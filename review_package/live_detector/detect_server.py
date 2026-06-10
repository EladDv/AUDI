#!/usr/bin/env python3
# ============================================================================
#  detect_server.py  -  LIVE drone detector for Raspberry Pi (phone UI)
# ----------------------------------------------------------------------------
#  Reads the 4-mic stream from the ESP32 (same serial framing as
#  record_server.py), runs the trained cascade in real time, and shows a big
#  live verdict on your phone:  NO-DRONE / DRONE (EVO) / DRONE (FPV).
#
#  Run on the Pi:
#      pip install flask pyserial numpy librosa torch
#      python detect_server.py --port /dev/ttyUSB0 --rate 16000
#
#  Open on your phone:  http://<pi-ip>:8766
#
#  It reuses the EXACT training code (config.py + features.py + models.py) and
#  the trained weights in experiments/<exp>/model.pt, plus the EVO/FPV head in
#  stage2_evo_fpv/stage2_model.npz - so live = what the model learned.
#
#  Copy to the Pi (minimum): config.py features.py models.py augment.py
#  detect_server.py, the chosen experiments/<exp>/ folder, and
#  stage2_evo_fpv/stage2_model.npz.
# ============================================================================
from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from config import CFG, CLASSES, EXPERIMENTS_DIR          # noqa: E402
from features import FEATURES                              # noqa: E402
from models import MODELS                                  # noqa: E402

NCH         = 4
FRAME_BYTES = NCH * 2
FULLSCALE   = 32768.0
TARGET      = CLASSES.index("target_drone")
OTHER       = CLASSES.index("other_drone")
# Detection fires on ANY drone (target OR other).  EVO/FPV only when confident.
TYPE_MIN_CONF = 0.30
WEAK_SIGNAL   = 0.75
STAGE2_FILE = HERE / "stage2_evo_fpv" / "stage2_model.npz"

app = Flask(__name__)

_state = {
    "connected": False, "port": "", "rate": 16000, "true_rate": 0.0,
    "p_drone": 0.0, "p_fpv": None, "drone_type": None, "verdict": "starting...",
    "color": "gray", "threshold": 0.5, "min_hits": 2, "min_level": -40.0,
    "hits": 0, "silence": False,
    "phys_gate": 0.0, "phys_dist": None, "phys_veto": False,
    "mic_db": [None] * NCH, "infer_ms": 0.0, "error": "",
    # model info (filled in main())
    "model_name": "", "model_date": "", "s2_date": "",
    # alarm log
    "alarm_count": 0, "alarm_log": [],   # list of {time, type, p_drone}
}
_last_alarm_state = False   # edge-detect: only log on rising edge

_ring: deque = deque(maxlen=400)     # recent 4-ch chunks (rolling window)
_ring_lock = threading.Lock()
_args = None
_model = None
_ck = None
_s2 = None
_fp = None       # fingerprint reference (9 physics cues) for the drone-ness gate


def load_fingerprint():
    """Reference centers + spread of the 9 physics cues (EVO & FPV)."""
    fp_path = HERE / "fingerprint_ref.npz"
    if not fp_path.exists():
        return None
    z = np.load(fp_path, allow_pickle=True)
    return {k: z[k] for k in z.files if k != "names"}


def physics_distance(clip):
    """Nearest z-scored distance of this clip's 9 cues to the EVO/FPV references.

    Real drones land ~0.2-0.5 sigma; speech/music land 2.5-4.5 sigma. So a small
    distance = drone-like physics, a large distance = NOT a drone (veto the alarm).
    Returns (nearest_dist, dist_evo, dist_fpv) or (None, ..) if no reference.
    """
    if _fp is None:
        return None, None, None
    from features import type_physics_stats
    cue = type_physics_stats(clip, CFG).astype(np.float32)
    gz = _fp["glob_std"]
    de = float(np.sqrt((((cue - _fp["evo_mean"]) / gz) ** 2).mean()))
    df = float(np.sqrt((((cue - _fp["fpv_mean"]) / gz) ** 2).mean()))
    return min(de, df), de, df


# --------------------------------------------------------------------------- #
def _db(v: float) -> float:
    return 20.0 * np.log10(max(float(v), 1e-12))


def load_stage1(exp: str):
    import torch
    ck = torch.load(EXPERIMENTS_DIR / exp / "model.pt",
                    map_location="cpu", weights_only=False)
    model = MODELS[ck["model"]](in_ch=ck["in_ch"], n_classes=len(ck["labels"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def load_stage2():
    if not STAGE2_FILE.exists():
        return None
    z = np.load(STAGE2_FILE, allow_pickle=True)
    bk = str(z["backbone"])
    if not (EXPERIMENTS_DIR / bk / "model.pt").exists():
        return None
    bmodel, _ = load_stage1(bk)
    if not hasattr(bmodel, "body"):
        return None
    files = set(z.files)
    # fused models carry use_scd/use_stats + feat_mean/feat_std; old emb-only ones
    # don't. Default to emb-only so a legacy 128-d file still loads correctly.
    use_scd   = bool(z["use_scd"])   if "use_scd"   in files else False
    use_stats = bool(z["use_stats"]) if "use_stats" in files else False
    feat_mean = z["feat_mean"] if "feat_mean" in files else None
    feat_std  = z["feat_std"]  if "feat_std"  in files else None
    print(f"Stage 2 head: coef={z['coef'].shape}  use_scd={use_scd} "
          f"use_stats={use_stats}")
    return dict(model=bmodel, backbone=bk, feature=str(z["feature"]),
                mean=z["mean"], std=z["std"], coef=z["coef"],
                intercept=z["intercept"], use_scd=use_scd, use_stats=use_stats,
                feat_mean=feat_mean, feat_std=feat_std)


# --------------------------------------------------------------------------- #
def _serial_reader() -> None:
    import serial  # type: ignore
    while True:
        try:
            _state["error"] = ""
            print(f"[serial] connecting {_state['port']} @ 921600 ...")
            ser = serial.Serial(_state["port"], 921600, timeout=1)
            try:
                ser.set_buffer_size(rx_size=1 << 18)
            except Exception:
                pass
            time.sleep(2.0)
            ser.reset_input_buffer()
            _state["connected"] = True
            t0, n_frames, leftover = time.time(), 0, b""
            while True:
                chunk = ser.read(8192)
                if not chunk:
                    continue
                chunk = leftover + chunk
                usable = (len(chunk) // FRAME_BYTES) * FRAME_BYTES
                leftover = chunk[usable:]
                if usable == 0:
                    continue
                arr = np.frombuffer(chunk[:usable], dtype=np.int16).reshape(-1, NCH)
                n_frames += arr.shape[0]
                with _ring_lock:
                    _ring.append(arr)
                el = time.time() - t0
                _state["true_rate"] = n_frames / max(el, 0.1)
        except Exception as exc:
            _state["connected"] = False
            _state["error"] = str(exc)
            print(f"[serial] error: {exc} - retry in 3 s")
            time.sleep(3)


# --------------------------------------------------------------------------- #
def _grab_window() -> np.ndarray | None:
    """Last clip_s seconds of 4-ch audio, or None if not enough yet."""
    with _ring_lock:
        snap = list(_ring)
    if not snap:
        return None
    a = np.concatenate(snap)                       # (N,4) int16
    need = int(CFG.clip_s * CFG.sr)
    if a.shape[0] < need:
        return None
    return a[-need:]


_last_window: np.ndarray | None = None   # (N,4) int16 – most recent raw window

def _detector_loop() -> None:
    global _last_alarm_state, _last_window
    import torch
    fn = FEATURES[_ck["features"]]["fn"]
    hop = max(0.25, float(_args.hop))
    hits = 0
    while True:
        seg = _grab_window()
        if seg is None:
            time.sleep(0.2)
            continue
        try:
            t0 = time.time()
            _last_window = seg          # save for /record_clip
            # live per-mic levels
            for c in range(NCH):
                rms = float(np.sqrt(np.mean((seg[:, c].astype(float) / FULLSCALE) ** 2)) + 1e-12)
                _state["mic_db"][c] = round(_db(rms), 1)

            # ---- silence gate ------------------------------------------------
            # PCEN normalises loudness so even a quiet room's noise floor can
            # look like a drone.  Skip inference when ALL mics are below the
            # minimum level - that's pure noise floor, not real audio.
            mono_rms = float(np.sqrt(np.mean((seg.astype(float).mean(axis=1) / FULLSCALE) ** 2)) + 1e-12)
            mono_db  = _db(mono_rms)
            if mono_db < float(_state["min_level"]):
                hits = 0
                _state["hits"] = 0
                _state["silence"] = True
                _state["p_drone"] = 0.0
                _state["verdict"] = f"NO DRONE  (too quiet {mono_db:.0f} dB)"
                _state["color"] = "gray"
                time.sleep(hop)
                continue
            _state["silence"] = False

            mono = seg.astype(np.float32).mean(axis=1) / FULLSCALE
            X = fn(mono, CFG)[None].astype(np.float32)
            Xn = (X - _ck["mean"]) / _ck["std"]
            with torch.no_grad():
                logits = _model(torch.tensor(Xn.astype(np.float32)))
                p = torch.softmax(logits, 1).numpy()[0]

            # P(any drone) = P(target_drone) + P(other_drone)
            p_drone = float(np.clip(p[TARGET] + p[OTHER], 0.0, 1.0))
            thr = float(_state["threshold"])

            # ---- PHYSICS GATE (OFF by default) ------------------------------
            # OPTIONAL safety net, disabled unless --phys-gate > 0. When OFF the
            # LIVE verdict is the PURE trained CNN, exactly as it came out of
            # training. Enable only if you want the 9-cue drone-ness veto back.
            gate = float(_state["phys_gate"])
            if gate > 0:
                nearest, de, df = physics_distance(mono)
                _state["phys_dist"] = None if nearest is None else round(nearest, 2)
                veto = (nearest is not None) and (nearest > gate)
            else:
                nearest, veto = None, False
                _state["phys_dist"] = None
            _state["phys_veto"] = bool(veto)

            over = (p_drone >= thr) and (not veto)
            hits = hits + 1 if over else 0
            _state["hits"] = hits
            alarm = hits >= int(_state["min_hits"])

            p_fpv = None
            dtype = None
            type_label = None
            if alarm and _s2 is not None:
                Xs = (FEATURES[_s2["feature"]]["fn"](mono, CFG)[None].astype(np.float32)
                      - _s2["mean"]) / _s2["std"]
                with torch.no_grad():
                    emb = _s2["model"].body(torch.tensor(Xs.astype(np.float32))).mean(dim=(2, 3)).numpy()
                # rebuild the SAME fused vector train_stage2 used: [emb | scd | 9 cues]
                parts = [emb]
                if _s2["use_scd"]:
                    from scd_probe import scd_alpha_profile
                    parts.append(scd_alpha_profile(mono, CFG.sr)[None].astype(np.float32))
                if _s2["use_stats"]:
                    from features import type_physics_stats
                    parts.append(type_physics_stats(mono, CFG)[None].astype(np.float32))
                feat = np.concatenate(parts, axis=1).astype(np.float32)
                if _s2["feat_mean"] is not None:
                    feat = (feat - _s2["feat_mean"]) / _s2["feat_std"]
                z = float((feat @ _s2["coef"].T + _s2["intercept"]).ravel()[0])
                p_fpv = float(1.0 / (1.0 + np.exp(-z)))
                conf  = abs(2 * p_fpv - 1)
                if conf >= TYPE_MIN_CONF:
                    dtype = "FPV" if p_fpv >= 0.5 else "EVO"
                    type_label = dtype
                else:
                    reason = "weak" if p_drone < WEAK_SIGNAL else "overlap"
                    type_label = f"type? ({reason})"

            _state["p_drone"]   = round(p_drone, 3)
            _state["p_fpv"]     = None if p_fpv is None else round(p_fpv, 3)
            _state["drone_type"] = dtype
            _state["infer_ms"]  = round((time.time() - t0) * 1000, 1)

            if not alarm:
                if veto and p_drone >= thr:
                    # CNN thought drone, but physics say otherwise (speech/music/etc.)
                    _state["verdict"] = f"NO DRONE  (not drone-like: {nearest:.1f} sigma)"
                    _state["color"] = "green"
                elif over:   # building toward alarm (hits < min_hits)
                    _state["verdict"] = f"possible... ({hits}/{int(_state['min_hits'])})"
                    _state["color"] = "orange"
                else:
                    _state["verdict"] = "NO DRONE"
                    _state["color"] = "green"
            elif dtype == "FPV":
                _state["verdict"], _state["color"] = "DRONE - FPV", "red"
            elif dtype == "EVO":
                _state["verdict"], _state["color"] = "DRONE - EVO", "orange"
            elif type_label:
                _state["verdict"] = f"DRONE ({type_label})"
                _state["color"] = "red"
            else:
                _state["verdict"], _state["color"] = "DRONE", "red"

            # --- alarm log: record rising edge only -------------------------
            alarm_now = alarm   # bool
            if alarm_now and not _last_alarm_state:
                entry = {
                    "time": time.strftime("%H:%M:%S"),
                    "type": _state.get("drone_type") or "DRONE",
                    "p": round(_state["p_drone"], 2),
                }
                _state["alarm_count"] += 1
                log = _state["alarm_log"]
                log.append(entry)
                if len(log) > 50:          # keep last 50 events
                    log.pop(0)
            _last_alarm_state = alarm_now

        except Exception as exc:
            _state["error"] = f"infer: {exc}"
        time.sleep(hop)


# --------------------------------------------------------------------------- #
_INDEX_HTML = r"""<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width,initial-scale=1"><meta charset="utf-8">
<title>Live Drone Detector</title><style>
body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;
padding:16px;max-width:480px;margin:0 auto}
h1{font-size:1.3rem;color:#38bdf8} .sub{font-size:.8rem;color:#64748b;margin-bottom:14px}
.banner{border-radius:14px;padding:30px 10px;text-align:center;font-size:2.2rem;
font-weight:800;margin-bottom:14px;transition:background .2s}
.card{background:#1e293b;border-radius:10px;padding:14px;margin-bottom:12px}
.row{display:flex;justify-content:space-between;margin:6px 0}
.label{color:#94a3b8;font-size:.85rem}.val{font-weight:700}
.bar{height:14px;border-radius:7px;background:#0f172a;overflow:hidden;margin-top:4px}
.fill{height:100%;background:#38bdf8;width:0%}
.mic-grid{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-top:10px}
.mic{text-align:center;background:#0f172a;border-radius:8px;padding:8px}
.mic-name{font-size:.72rem;color:#64748b}.mic-db{font-weight:700}
input[type=range]{width:100%}
.btn{background:#1d4ed8;color:#fff;border:none;border-radius:8px;padding:8px 16px;
font-size:.85rem;cursor:pointer;margin-top:8px;width:100%}
.btn:active{background:#1e40af}
.alarm-entry{font-size:.78rem;padding:3px 0;border-bottom:1px solid #334155;color:#94a3b8}
.alarm-entry .atype{color:#f87171;font-weight:700}
</style></head><body>
<h1>Live Drone Detector</h1><p class="sub" id="sub">connecting...</p>
<div class="banner" id="banner" style="background:#334155">...</div>

<div class="card">
 <div class="row"><span class="label">Model</span><span class="val" id="mname" style="color:#38bdf8">-</span></div>
 <div class="row"><span class="label">Trained</span><span class="val" id="mdate" style="font-size:.8rem;color:#94a3b8">-</span></div>
 <div class="row"><span class="label">Stage-2 trained</span><span class="val" id="s2date" style="font-size:.8rem;color:#94a3b8">-</span></div>
</div>

<div class="card">
 <div class="row"><span class="label">P(drone)</span><span class="val" id="pd">-</span></div>
 <div class="bar"><div class="fill" id="pdbar"></div></div>
 <div class="row" style="margin-top:10px"><span class="label">P(FPV) | type</span><span class="val" id="pf">-</span></div>
 <div class="row"><span class="label">Threshold</span><span class="val" id="thv">0.50</span></div>
 <input type="range" id="thr" min="0" max="1" step="0.01" value="0.5" oninput="setThr()">
 <div class="row" style="margin-top:8px"><span class="label">Min-hits (consecutive windows)</span><span class="val" id="mhv">5</span></div>
 <input type="range" id="mhr" min="1" max="20" step="1" value="5" oninput="setMH()">
 <div class="row" style="margin-top:8px"><span class="label">Inference</span><span class="val" id="ms">-</span></div>
 <div class="row"><span class="label">Hits / silence</span><span class="val" id="sg">-</span></div>
 <div class="row"><span class="label">Physics (drone-like?)</span><span class="val" id="ph">-</span></div>
</div>

<div class="card">
 <div class="row"><span class="label">Alarms this session</span><span class="val" id="acnt" style="color:#f87171">0</span></div>
 <div id="alog" style="margin-top:8px;max-height:140px;overflow-y:auto"></div>
 <button class="btn" onclick="recordClip()">📥 Save clip as false positive (for retraining)</button>
 <div id="recmsg" style="font-size:.75rem;color:#4ade80;margin-top:4px"></div>
</div>

<div class="card">
 <div class="row"><span class="label">Stream</span><span class="val" id="st">-</span></div>
 <div class="mic-grid">
  <div class="mic"><div class="mic-name">Mic 1</div><div class="mic-db" id="m0">-</div></div>
  <div class="mic"><div class="mic-name">Mic 2</div><div class="mic-db" id="m1">-</div></div>
  <div class="mic"><div class="mic-name">Mic 3</div><div class="mic-db" id="m2">-</div></div>
  <div class="mic"><div class="mic-name">Mic 4</div><div class="mic-db" id="m3">-</div></div>
 </div>
</div>
<script>
const C={green:'#16a34a',orange:'#ea580c',red:'#dc2626',gray:'#334155'};
async function setThr(){
 const v=document.getElementById('thr').value;
 document.getElementById('thv').textContent=Number(v).toFixed(2);
 await fetch('/threshold?v='+v,{method:'POST'});
}
async function setMH(){
 const v=document.getElementById('mhr').value;
 document.getElementById('mhv').textContent=v;
 await fetch('/min_hits?v='+v,{method:'POST'});
}
async function recordClip(){
 const r=await(await fetch('/record_clip',{method:'POST'})).json();
 const m=document.getElementById('recmsg');
 if(r.ok){m.textContent='Saved: '+r.file.split(/[\\/]/).pop();}
 else{m.style.color='#f87171';m.textContent='Error: '+r.error;}
 setTimeout(()=>{m.textContent='';},5000);
}
let _prevLog=0;
async function poll(){
 try{
  const d=await(await fetch('/status')).json();
  document.getElementById('sub').textContent=d.port+' | '+d.rate+' Hz';
  const b=document.getElementById('banner');
  b.textContent=d.verdict; b.style.background=C[d.color]||C.gray;
  document.getElementById('pd').textContent=(d.p_drone*100).toFixed(0)+'%';
  document.getElementById('pdbar').style.width=(d.p_drone*100)+'%';
  document.getElementById('pf').textContent=
     d.p_fpv==null?(d.drone_type||'-'):((d.p_fpv*100).toFixed(0)+'% -> '+d.drone_type);
  document.getElementById('ms').textContent=d.infer_ms+' ms';
  const sg=document.getElementById('sg');
  if(d.silence){sg.textContent='SILENT (gate active)';sg.style.color='#64748b';}
  else{sg.textContent='hits '+d.hits+'/'+d.min_hits;sg.style.color=d.hits>0?'#fb923c':'#4ade80';}
  const ph=document.getElementById('ph');
  if(d.phys_dist==null){ph.textContent='off';ph.style.color='#64748b';}
  else if(d.phys_veto){ph.textContent=d.phys_dist+' sigma -> NOT drone';ph.style.color='#4ade80';}
  else{ph.textContent=d.phys_dist+' sigma -> drone-like';ph.style.color='#fb923c';}
  const st=document.getElementById('st');
  st.textContent=d.connected?('CONNECTED '+d.true_rate.toFixed(0)+' Hz'):'DISCONNECTED';
  st.style.color=d.connected?'#4ade80':'#f87171';
  for(let i=0;i<4;i++){const v=d.mic_db[i];
   document.getElementById('m'+i).textContent=v!=null?v.toFixed(1)+' dB':'-';}
  if(d.error)st.textContent='ERR: '+d.error;
  // model info
  document.getElementById('mname').textContent=d.model_name||'-';
  document.getElementById('mdate').textContent=d.model_date||'-';
  document.getElementById('s2date').textContent=d.s2_date||'-';
  // alarm log
  document.getElementById('acnt').textContent=d.alarm_count||0;
  if(d.alarm_log&&d.alarm_log.length!==_prevLog){
   _prevLog=d.alarm_log.length;
   const box=document.getElementById('alog');
   box.innerHTML=d.alarm_log.slice().reverse().map(e=>
    '<div class="alarm-entry">'+e.time+' &nbsp;<span class="atype">'+e.type+'</span>&nbsp; P='+Math.round(e.p*100)+'%</div>'
   ).join('');
  }
  // keep sliders in sync with server values
  const tslider=document.getElementById('thr');
  if(Math.abs(tslider.value-d.threshold)>0.005){tslider.value=d.threshold;document.getElementById('thv').textContent=Number(d.threshold).toFixed(2);}
  const mslider=document.getElementById('mhr');
  if(parseInt(mslider.value)!==d.min_hits){mslider.value=d.min_hits;document.getElementById('mhv').textContent=d.min_hits;}
 }catch(e){}
}
setInterval(poll,400);poll();
</script></body></html>"""


@app.route("/")
def index():
    return _INDEX_HTML


@app.route("/status")
def status():
    return jsonify({k: v for k, v in _state.items() if k != "hits" or True})


@app.route("/threshold", methods=["POST"])
def set_threshold():
    try:
        _state["threshold"] = float(request.args.get("v", 0.5))
    except Exception:
        pass
    return jsonify(ok=True, threshold=_state["threshold"])


@app.route("/min_hits", methods=["POST"])
def set_min_hits():
    try:
        _state["min_hits"] = max(1, int(request.args.get("v", 5)))
    except Exception:
        pass
    return jsonify(ok=True, min_hits=_state["min_hits"])


@app.route("/record_clip", methods=["POST"])
def record_clip():
    """Save the most recent 1-second window as a WAV for hard-negative retraining."""
    import wave, struct
    seg = _last_window
    if seg is None:
        return jsonify(ok=False, error="no audio yet")
    out_dir = HERE / "false_positives"
    out_dir.mkdir(exist_ok=True)
    fname = out_dir / f"fp_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    mono = seg.astype(np.float32).mean(axis=1)
    mono_i16 = np.clip(mono, -32768, 32767).astype(np.int16)
    with wave.open(str(fname), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_state["rate"])
        wf.writeframes(mono_i16.tobytes())
    print(f"[record_clip] saved {fname}")
    return jsonify(ok=True, file=str(fname))


# --------------------------------------------------------------------------- #
def main() -> None:
    global _args, _model, _ck, _s2, _fp
    ap = argparse.ArgumentParser(description="Live drone detector for the Pi")
    ap.add_argument("--port", required=True, help="serial port, e.g. /dev/ttyUSB0")
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--exp", default="mel2_cnn", help="Stage-1 experiment to use")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-hits", type=int, default=2,
                    help="consecutive windows over threshold before ALARM (filters blips)")
    ap.add_argument("--min-level", type=float, default=-40.0,
                    help="silence gate dB: below this, skip inference (PCEN amplifies "
                         "a silent mic's hiss into a fake drone without this gate)")
    ap.add_argument("--phys-gate", type=float, default=0.0,
                    help="OPTIONAL physics gate (sigma), OFF by default (0=off). When "
                         ">0, vetoes the alarm if the window's 9 physics cues are "
                         "farther than this from BOTH drone fingerprints. With 0 the "
                         "LIVE verdict is the pure trained CNN.")
    ap.add_argument("--hop", type=float, default=0.5, help="seconds between inferences")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--webport", type=int, default=8766)
    _args = ap.parse_args()

    _state["port"]      = _args.port
    _state["rate"]      = _args.rate
    _state["threshold"] = _args.threshold
    _state["min_hits"]  = _args.min_hits
    _state["min_level"] = _args.min_level
    _state["phys_gate"] = _args.phys_gate

    _model, _ck = load_stage1(_args.exp)
    _s2 = load_stage2()
    _fp = load_fingerprint()

    # populate model info for the UI
    import os
    m1_path = EXPERIMENTS_DIR / _args.exp / "model.pt"
    m2_path = HERE / "stage2_evo_fpv" / "stage2_model.npz"
    _state["model_name"] = _args.exp
    _state["model_date"] = (time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(m1_path)))
                            if m1_path.exists() else "?")
    _state["s2_date"]    = (time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(m2_path)))
                            if m2_path.exists() else "?")

    print(f"Stage 1: {_args.exp} (feature={_ck['features']})")
    print(f"Stage 2: {'loaded ('+_s2['backbone']+')' if _s2 else 'NOT FOUND - EVO/FPV disabled'}")
    if _args.phys_gate > 0 and _fp is not None:
        print(f"Physics gate: ON ({_args.phys_gate} sigma)")
    else:
        print("Physics gate: OFF  (LIVE = pure trained CNN)")

    threading.Thread(target=_serial_reader, daemon=True).start()
    threading.Thread(target=_detector_loop, daemon=True).start()

    print("=" * 56)
    print("  LIVE Drone Detector")
    print(f"  Serial : {_args.port} @ 921600   Rate: {_args.rate} Hz")
    print(f"  Open phone:  http://<pi-ip>:{_args.webport}")
    print("=" * 56)
    app.run(host=_args.host, port=_args.webport, debug=False, threaded=True)


if __name__ == "__main__":
    main()
