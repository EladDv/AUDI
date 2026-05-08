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

    st.subheader(f"{len(rows_filtered)} models at {precision_filter}")
    st.caption(
        "cov% = mean % of attack segment windows above threshold · "
        "1st% = median % of segment before first detection (lower=earlier) · "
        "bg = background windows above threshold"
    )

    if rows_filtered:
        for r in rows_filtered:
            bg = int(r["bg"])
            bg_color = "green" if bg == 0 else "orange" if bg <= 3 else "red"
            cols = st.columns([2, 1, 1, 1, 1, 2])
            cols[0].write(r["model"])
            cols[1].write(f"{float(r['sigma']):.3f}")
            cols[2].write(f"{r['cov_pct']}%")
            cols[3].write(f"{r['first_pct']}%")
            cols[4].markdown(
                f"<span style='color:{bg_color}'>{bg}/60</span>",
                unsafe_allow_html=True,
            )
            cols[5].write(r.get("sweep", ""))

        best = rows_filtered[0]
        st.info(
            f"**Best:** {best['model']} — "
            f"cov={best['cov_pct']}%, "
            f"1st={best['first_pct']}%, "
            f"bg={best['bg']}/60 at σ={float(best['sigma']):.3f}"
        )
    else:
        st.info("No models match the filters.")
