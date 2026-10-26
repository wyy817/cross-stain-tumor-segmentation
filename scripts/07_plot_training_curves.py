"""Plot mDice-vs-iteration training curves for E1/E2/E3 (v7 logs).

Usage:
    python scripts/07_plot_training_curves.py
    python scripts/07_plot_training_curves.py --out outputs/training_curves.pdf
"""
import re
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Log files ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent  # FYP_S2 root
LOGS = {
    "E1 (ResNet+FPN)":      BASE / "work_dirs/E1_resnet_fpn_v7/20260413_190517/20260413_190517.log",
    "E2 (CONCH+FPN)":       BASE / "work_dirs/E2_conch_fpn_v7/20260413_203949/20260413_203949.log",
    "E3 (CONCH+FPN+StainAug)": BASE / "work_dirs/E3_conch_fpn_aug_v7/20260413_215258/20260413_215258.log",
}

# Regex patterns — mmseg v7 logs: val batch total = 1508
VAL_SUMMARY_RE = re.compile(
    r"Iter\(val\)\s+\[(\d+)/(\d+)\].*?mDice:\s*([\d.]+)"
)
BEST_CKPT_RE = re.compile(
    r"best checkpoint with ([\d.]+) mDice at (\d+) iter"
)
TRAIN_RE = re.compile(
    r"Iter\(train\)\s+\[\s*(\d+)/\d+\].*?loss:\s*([\d.e+-]+)"
)


def parse_log(path: Path):
    """Return (val_iters, val_mdice, train_iters, train_loss, best_iter, best_mdice)."""
    val_iters, val_mdice = [], []
    train_iters, train_loss = [], []
    best_iter, best_mdice = None, None

    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Collect val summaries: only the last batch of each val pass
    pending_total = None
    for line in lines:
        # Train iter
        m = TRAIN_RE.search(line)
        if m:
            train_iters.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            continue

        # Val summary line: Iter(val) [N/N]
        m = VAL_SUMMARY_RE.search(line)
        if m:
            current, total, mdice = int(m.group(1)), int(m.group(2)), float(m.group(3))
            if current == total:          # last batch → full-epoch summary
                pending_total = (total, mdice)
            continue

        # Best checkpoint line carries the iteration number
        m = BEST_CKPT_RE.search(line)
        if m and pending_total is not None:
            mdice_val = float(m.group(1))
            ckpt_iter = int(m.group(2))
            val_iters.append(ckpt_iter)
            val_mdice.append(pending_total[1])   # use the mDice from summary line
            if best_mdice is None or mdice_val > best_mdice:
                best_mdice = mdice_val
                best_iter = ckpt_iter
            pending_total = None

    # If any val summary without a following best-ckpt line (non-improving),
    # we still need to pair it with an iter.  Use a second pass to collect all
    # val summaries with their preceding checkpoint-save iters.
    if not val_iters:
        # Fallback: pair val summaries in order with checkpoint iters
        ckpt_iters = [int(m.group(2)) for m in
                      (BEST_CKPT_RE.search(l) for l in lines) if m]
        val_mdice_raw = [float(m.group(3)) for m in
                         (VAL_SUMMARY_RE.search(l) for l in lines)
                         if m and int(m.group(1)) == int(m.group(2))]
        val_iters = ckpt_iters[:len(val_mdice_raw)]
        val_mdice = val_mdice_raw[:len(val_iters)]

    return val_iters, val_mdice, train_iters, train_loss, best_iter, best_mdice


def parse_log_v2(path: Path):
    """Robust two-pass parser.

    Pass 1: collect every val-summary mDice (final batch only).
    Pass 2: collect checkpoint save iters in order.
    Then zip them — mmseg always saves a checkpoint after each val.
    """
    val_mdice_list = []   # in order of occurrence
    ckpt_iter_list = []   # in order of occurrence
    train_iters, train_loss = [], []
    best_iter, best_mdice = None, None

    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Detect total val-batch count from first val line
    val_total = None
    for line in lines:
        m = VAL_SUMMARY_RE.search(line)
        if m:
            val_total = int(m.group(2))
            break

    if val_total is None:
        return [], [], [], [], None, None

    for line in lines:
        # Train
        m = TRAIN_RE.search(line)
        if m:
            train_iters.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            continue
        # Val final-batch summary
        m = VAL_SUMMARY_RE.search(line)
        if m and int(m.group(1)) == val_total:
            val_mdice_list.append(float(m.group(3)))
            continue
        # Checkpoint iter (from "best checkpoint ... at N iter" or "Saving checkpoint at N")
        m = re.search(r"Saving checkpoint at (\d+) iterations", line)
        if m:
            ckpt_iter_list.append(int(m.group(1)))

    # Pair: every checkpoint has a val run before it
    n = min(len(val_mdice_list), len(ckpt_iter_list))
    val_iters = ckpt_iter_list[:n]
    val_mdice = val_mdice_list[:n]

    # Best
    if val_mdice:
        idx = val_mdice.index(max(val_mdice))
        best_mdice = val_mdice[idx]
        best_iter  = val_iters[idx]

    return val_iters, val_mdice, train_iters, train_loss, best_iter, best_mdice


# ── Plotting ─────────────────────────────────────────────────────────────────
COLORS = {
    "E1 (ResNet+FPN)":          "#2196F3",   # blue
    "E2 (CONCH+FPN)":           "#FF9800",   # orange
    "E3 (CONCH+FPN+StainAug)":  "#4CAF50",   # green
}
MARKERS = {
    "E1 (ResNet+FPN)":          "o",
    "E2 (CONCH+FPN)":           "s",
    "E3 (CONCH+FPN+StainAug)":  "^",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/training_curves.png",
                        help="Output image path (png/pdf/svg)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_dice, ax_loss = axes

    for label, log_path in LOGS.items():
        if not log_path.exists():
            print(f"[WARN] Log not found: {log_path}")
            continue

        val_iters, val_mdice, tr_iters, tr_loss, best_iter, best_mdice = \
            parse_log_v2(log_path)

        color  = COLORS[label]
        marker = MARKERS[label]
        print(f"{label}: {len(val_iters)} val points, "
              f"best mDice={best_mdice:.2f} @ iter {best_iter}")

        # ── mDice curve ──
        if val_iters:
            ax_dice.plot(val_iters, val_mdice,
                         color=color, marker=marker, markersize=5,
                         linewidth=1.8, label=label)
            # Mark best point
            if best_iter is not None:
                ax_dice.axvline(best_iter, color=color,
                                linestyle="--", linewidth=0.8, alpha=0.5)
                ax_dice.scatter([best_iter], [best_mdice],
                                color=color, s=80, zorder=5)

        # ── Training loss curve (smoothed) ──
        if tr_iters:
            # Smooth with window=10
            window = 10
            smoothed = [
                sum(tr_loss[max(0, i - window):i + 1]) /
                len(tr_loss[max(0, i - window):i + 1])
                for i in range(len(tr_loss))
            ]
            ax_loss.plot(tr_iters, smoothed,
                         color=color, linewidth=1.5,
                         label=label, alpha=0.9)

    # ── Axes formatting ──
    ax_dice.set_xlabel("Iteration", fontsize=12)
    ax_dice.set_ylabel("Val mDice (%)", fontsize=12)
    ax_dice.set_title("Validation mDice vs. Iteration (v7, HE+IHC val)", fontsize=12)
    ax_dice.legend(fontsize=9)
    ax_dice.grid(True, alpha=0.3)
    ax_dice.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_dice.set_ylim(bottom=50)

    ax_loss.set_xlabel("Iteration", fontsize=12)
    ax_loss.set_ylabel("Training Loss", fontsize=12)
    ax_loss.set_title("Training Loss vs. Iteration (smoothed, w=10)", fontsize=12)
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
