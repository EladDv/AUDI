"""System health stats for web UI."""
import os


def _cpu_temp() -> float:
    """Try to read CPU temperature from common Linux paths."""
    paths = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/hwmon/hwmon0/temp1_input",
    ]
    for p in paths:
        try:
            with open(p) as f:
                val = int(f.read().strip())
                return val / 1000.0
        except (FileNotFoundError, ValueError):
            continue
    return 0.0


def _memory() -> dict:
    """Read memory info from /proc/meminfo."""
    mem = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    mem["total_kb"] = int(line.split()[1])
                elif "MemAvailable" in line:
                    mem["available_kb"] = int(line.split()[1])
    except Exception:
        pass
    total = mem.get("total_kb", 0)
    avail = mem.get("available_kb", 0)
    used_pct = round((1 - avail / total) * 100, 1) if total > 0 else 0
    return {"used_pct": used_pct, "total_kb": total, "available_kb": avail}


def _load() -> dict:
    try:
        a, b, c = os.getloadavg()
        return {"1min": round(a, 2), "5min": round(b, 2), "15min": round(c, 2)}
    except Exception:
        return {"1min": 0, "5min": 0, "15min": 0}


def _uptime() -> tuple:
    try:
        with open("/proc/uptime") as f:
            secs = float(f.read().split()[0])
    except Exception:
        secs = 0
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    return secs, f"{h}h {m}m"


def _disk() -> dict:
    try:
        stat = os.statvfs("/")
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bavail * stat.f_frsize
        used_pct = round((1 - free / total) * 100, 1) if total > 0 else 0
        return {"used_pct": used_pct, "free_gb": round(free / 1e9, 1)}
    except Exception:
        return {"used_pct": 0, "free_gb": 0}


def all_stats() -> dict:
    uptime_secs, uptime_str = _uptime()
    return {
        "cpu_temperature_c": _cpu_temp(),
        "memory": _memory(),
        "load": _load(),
        "uptime_seconds": uptime_secs,
        "uptime_str": uptime_str,
        "disk": _disk(),
    }
