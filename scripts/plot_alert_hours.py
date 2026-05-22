#!/usr/bin/env python3
"""Histogram of alert times by hour (UTC+3)."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

DATA = Path("/home/elad/projects/AUDI/data/field_recordings_20260514")
scores = json.loads((DATA / "wd003_scores.json").read_text())

timestamps = []
for a in scores["alerts"]:
    ts = int(a["alert_dir"].split("_")[1])
    timestamps.append(ts)

# Remove last alert
timestamps = sorted(timestamps)[:-1]

# Hours relative to first alert (t=0)
t0 = timestamps[0]
hours_rel = [(ts - t0) / 3600 for ts in timestamps]

# Bin into integer hours
max_h = int(np.ceil(max(hours_rel)))
counts = Counter(int(h) for h in hours_rel)
all_hours = list(range(max_h + 1))
vals = [counts.get(h, 0) for h in all_hours]
pcts = [v / len(timestamps) * 100 for v in vals]

fig, ax = plt.subplots(figsize=(14, 5), dpi=140, facecolor="#0e1117")
ax.set_facecolor("#0e1117")

bars = ax.bar(all_hours, vals, color="#4ECDC4", edgecolor="#1a3a3a", linewidth=0.5)

for h, v, p in zip(all_hours, vals, pcts):
    if v > 0:
        ax.text(h, v + max(vals) * 0.03, f"{v}\n({p:.1f}%)", ha='center', va='bottom',
                color='#ccc', fontsize=7.5, fontweight='bold', linespacing=1.2)

ax.set_xlabel("Hours since first alert (t=0)", color="#ccc", fontsize=12)
ax.set_ylabel("Alert count", color="#ccc", fontsize=12)

d_start = datetime.fromtimestamp(t0, tz=timezone.utc).strftime('%b %d %H:%M')
d_end = datetime.fromtimestamp(max(timestamps), tz=timezone.utc).strftime('%b %d %H:%M')
ax.set_title(f"Field alerts — {len(timestamps)} total, {d_start} to {d_end} UTC  (first alert = t=0)",
             color="#ccc", fontsize=12, fontweight='bold')
ax.set_xticks(all_hours)
ax.set_xticklabels([f"+{h}h" for h in all_hours], color="#888", fontsize=8)
ax.tick_params(colors="#888", labelsize=9)
ax.set_ylim(0, max(vals) * 1.18)
ax.grid(axis='y', alpha=0.15, color='#444')
for spine in ax.spines.values():
    spine.set_color("#333")

top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
for h, _ in top3:
    bars[h].set_color("#FF6B6B")

fig.tight_layout(pad=1.2)
path = DATA / "alert_hour_histogram.png"
fig.savefig(str(path), dpi=140, facecolor="#0e1117", edgecolor="none")
plt.close(fig)

print(f"Saved {path}")
print()
for h in sorted(counts.keys()):
    if counts[h] > 0:
        print(f"  +{h:3d}h  {counts[h]:3d}  ({pcts[h]:.1f}%)")
print(f"\nTop 3: {', '.join(f'+{h}h ({counts[h]}, {pcts[h]:.1f}%)' for h, _ in top3)}")
