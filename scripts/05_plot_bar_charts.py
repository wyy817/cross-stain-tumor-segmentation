"""
05_plot_bar_charts.py
Generate cross-stain comparison bar charts for v6 (in-dist) and v7 (hold-out).
Metric: tumor_Dice (the meaningful indicator; mDice is inflated by background class).
Output: outputs/bar_v6.png, outputs/bar_v7.png
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load JSONs ──────────────────────────────────────────────────────────────

def load(name):
    path = os.path.join(ROOT, name)
    with open(path) as f:
        return json.load(f)

v6 = {
    "E1": load("eval_E1_v6.json"),
    "E2": load("eval_E2_v6.json"),
    "E3": load("eval_E3_v6.json"),
}
v7 = {
    "E1": load("eval_E1_v7.json"),
    "E2": load("eval_E2_v7.json"),
    "E3": load("eval_E3_v7.json"),
}

STAINS   = ["HE", "IHC", "NCR"]
EXPS     = ["E1", "E2", "E3"]
METRIC   = "tumor_Dice"

EXP_LABELS = {
    "E1": "E1  ResNet-50 + FPN",
    "E2": "E2  CONCH (frozen) + FPN",
    "E3": "E3  CONCH + FPN + StainAug",
}

# Colour palette: one colour per experiment (colour-blind friendly)
COLORS = {
    "E1": "#4C72B0",   # blue
    "E2": "#DD8452",   # orange
    "E3": "#55A868",   # green
}

STAIN_LABELS = {"HE": "H&E", "IHC": "IHC", "NCR": "NCR"}


# ── Plotting function ────────────────────────────────────────────────────────

def plot_bar(data, title, out_path, v7_note=False):
    """
    data: dict  {exp: {stain: value}}
    """
    n_stains = len(STAINS)
    n_exps   = len(EXPS)
    bar_w    = 0.22
    x        = np.arange(n_stains)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, exp in enumerate(EXPS):
        offsets = (i - (n_exps - 1) / 2) * bar_w
        vals = [data[exp][s][METRIC] for s in STAINS]
        bars = ax.bar(
            x + offsets, vals,
            width=bar_w,
            color=COLORS[exp],
            label=EXP_LABELS[exp],
            zorder=3,
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on top of each bar
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v:.1f}",
                ha="center", va="bottom",
                fontsize=8.5, color="#222222",
            )

    # NCR annotation for v7
    if v7_note:
        ax.annotate(
            "NCR: unseen\nduring training",
            xy=(x[2], 5), xytext=(x[2] + 0.45, 30),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
            fontsize=8, color="#555555",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([STAIN_LABELS[s] for s in STAINS], fontsize=12)
    ax.set_ylabel("Tumor Dice (%)", fontsize=11)
    ax.set_ylim(0, 107)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Extract values ───────────────────────────────────────────────────────────

def extract(raw):
    return {exp: {s: raw[exp][s] for s in STAINS} for exp in EXPS}

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_bar(
        extract(v6),
        title="Multi-Stain Training — In-Distribution Evaluation",
        out_path=os.path.join(OUT_DIR, "bar_v6.png"),
        v7_note=False,
    )
    plot_bar(
        extract(v7),
        title="Multi-Stain Training — Stain Hold-Out Evaluation (NCR unseen)",
        out_path=os.path.join(OUT_DIR, "bar_v7.png"),
        v7_note=True,
    )
    print("Done.")
