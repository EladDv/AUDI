"""Attack run leaderboard — ranked model comparison at precision levels."""

from __future__ import annotations

import csv

import streamlit as st

from eval_app.model_utils import _CHECKPOINTS_DIR


def leaderboard_page() -> None:
    """Show ranked models from attack_run_precision_eval.csv."""
    st.title("🎯 Attack Run Leaderboard")

    csv_path = _CHECKPOINTS_DIR / "attack_run_precision_eval.csv"
    if not csv_path.exists():
        st.warning("No eval data. Run `uv run python scripts/evaluate.py attack-runs` first.")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    precisions = sorted(set(r["precision"] for r in rows), reverse=True)
    sweeps_all = sorted(set(r.get("sweep", "") for r in rows))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        precision_filter = st.selectbox(
            "Precision", precisions, index=precisions.index("P90") if "P90" in precisions else 0
        )
    with col2:
        bg_max = st.slider("Max bg detections", 0, 60, 60)
    with col3:
        min_cov = st.slider("Min coverage %", 0, 100, 0)
    with col4:
        sweep_filter = st.multiselect("Sweeps", sweeps_all, default=[])

    rows_filtered = [
        r
        for r in rows
        if r["precision"] == precision_filter
        and int(r["bg"]) <= bg_max
        and float(r["cov_pct"]) >= min_cov
        and (not sweep_filter or r.get("sweep") in sweep_filter)
    ]
    rows_filtered.sort(
        key=lambda r: (
            -float(r["cov_pct"]),
            float(r["first_pct"]),
            int(r["bg"]),
        )
    )

    # ── Load field eval data ──────────────────────────────────────────
    field_path = _CHECKPOINTS_DIR / "field_eval_all.csv"
    field_data: dict[str, dict] = {}
    if field_path.exists():
        with open(field_path) as f:
            for fr in csv.DictReader(f):
                ref = f"{fr['ref']}" if 'ref' in fr else f"{fr['sweep']}/{fr['model']}"
                p = fr.get("P", "")
                if p not in field_data:
                    field_data[p] = {}
                field_data[p][ref] = fr

    st.subheader(f"{len(rows_filtered)} models at {precision_filter}")
    st.caption(
        "cov% = mean % of attack segment windows above threshold · "
        "1st% = median % of segment before first detection (lower=earlier) · "
        "bg = background windows above threshold · "
        "field TP/FP = alert recordings with at least 1 detection"
    )

    if rows_filtered:
        # Header
        cols = st.columns([3, 1, 1, 1, 1, 1, 1, 1, 3])
        cols[0].write("**model**")
        cols[1].write("**σ**")
        cols[2].write("**cov%**")
        cols[3].write("**1st%**")
        cols[4].write("**bg**")
        cols[5].write("**TP**")
        cols[6].write("**FP**")
        cols[7].write("**FN**")
        cols[8].write("**sweep**")

        for r in rows_filtered:
            ref = f"{r.get('sweep','')}/{r['model']}"
            bg = int(r["bg"])
            bg_color = "green" if bg == 0 else "orange" if bg <= 3 else "red"

            # Field eval for this precision
            field_info = field_data.get(precision_filter, {}).get(ref, {})
            tp = field_info.get("alert_tp", "?")
            fp = field_info.get("alert_fp", "?")
            fn = field_info.get("alert_fn", "?")

            cols2 = st.columns([3, 1, 1, 1, 1, 1, 1, 1, 3])
            cols2[0].write(r["model"])
            cols2[1].write(f"{float(r['sigma']):.3f}")
            cols2[2].write(f"{r['cov_pct']}%")
            cols2[3].write(f"{r['first_pct']}%")
            cols2[4].markdown(
                f"<span style='color:{bg_color}'>{bg}/60</span>",
                unsafe_allow_html=True,
            )
            cols2[5].write(str(tp))
            cols2[6].write(str(fp))
            cols2[7].write(str(fn))
            cols2[8].write(r.get("sweep", ""))

        best = rows_filtered[0]
        st.info(
            f"**Best:** {best['model']} — "
            f"cov={best['cov_pct']}%, "
            f"1st={best['first_pct']}%, "
            f"bg={best['bg']}/60 at σ={float(best['sigma']):.3f}"
        )
    else:
        st.info("No models match the filters.")
