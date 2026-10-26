"""
Script 04: Visualize model predictions on eval patches.

Generates side-by-side figures: [Input | Ground Truth | Prediction]
for N samples per stain from the eval split.

Usage:
    python scripts/04_visualize.py \
        --config  configs/E1_resnet_fpn.py \
        --ckpt    v6_pth/E1_best_mDice_iter_20000.pth \
        --stains  HE IHC NCR \
        --n       8 \
        --cs_csv  data/patch_index_private_v6.csv \
        --out_dir outputs/vis_E1

    # Only patches that contain tumor (better for paper figures):
    python scripts/04_visualize.py ... --tumor_only

    # Specific patients only:
    python scripts/04_visualize.py ... --patients PATIENT_1 PATIENT_2
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import models.register  # noqa: F401

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from mmengine.config import Config
from mmseg.apis import init_model, inference_model

# ── PyTorch 2.6 safe-load patch (same as 03_evaluate.py) ─────────────────────
_orig_torch_load = torch.load
def _torch_load_unsafe(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_unsafe


# ── Colours (BGR) ─────────────────────────────────────────────────────────────
COLOR_GT   = (0,  180,  0)   # green  — ground truth overlay
COLOR_PRED = (0,   0,  220)  # red    — prediction overlay
ALPHA      = 0.40            # overlay transparency
LABEL_W    = 130             # width of the left info bar per row
GAP_COL    = 4               # pixel gap between panels
GAP_ROW    = 3               # pixel gap between rows
HDR_H      = 34              # header bar height


def parse_args():
    p = argparse.ArgumentParser(description="Visualize segmentation predictions")
    p.add_argument("--config",   required=True,  help="Config file path")
    p.add_argument("--ckpt",     required=True,  help="Checkpoint .pth path")
    p.add_argument("--stains",   nargs="+", default=["HE", "IHC", "NCR"])
    p.add_argument("--n",        type=int, default=8,
                   help="Number of samples per stain")
    p.add_argument("--cs_csv",
                   default="/path/to/FYP_S2/data/patch_index_private_v6.csv",
                   help="CSV with patch index (must contain split=val rows)")
    p.add_argument("--out_dir",  default="outputs/vis")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--tumor_only", action="store_true",
                   help="Skip all-background patches (recommended for paper figures)")
    p.add_argument("--patients", nargs="+", default=None,
                   help="Restrict to specific patient IDs (e.g. PATIENT_1)")
    return p.parse_args()


# ── Image utilities ───────────────────────────────────────────────────────────

def overlay(img_bgr: np.ndarray, mask: np.ndarray,
            color_bgr: tuple, alpha: float) -> np.ndarray:
    """Blend a binary mask over img_bgr with the given colour and alpha."""
    out = img_bgr.astype(np.float32)
    c   = np.array(color_bgr, dtype=np.float32)
    where = mask == 1
    out[where] = out[where] * (1.0 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def put_text_center(img: np.ndarray, text: str, y: int,
                    scale: float = 0.50, thickness: int = 1,
                    fg=(255, 255, 255), bg=(30, 30, 30)):
    """Draw text centred horizontally on img."""
    w = img.shape[1]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max((w - tw) // 2, 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, bg,
                thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg,
                thickness, cv2.LINE_AA)


def build_header(total_w: int, panel_w: int) -> np.ndarray:
    """Top header bar with column titles."""
    hdr = np.full((HDR_H, total_w, 3), 235, dtype=np.uint8)
    labels = ["Input", "Ground Truth", "Prediction"]
    # panel centres relative to the full row (LABEL_W offset + gaps)
    starts = [
        LABEL_W + GAP_COL,
        LABEL_W + GAP_COL + panel_w + GAP_COL,
        LABEL_W + GAP_COL + 2 * (panel_w + GAP_COL),
    ]
    for lbl, x0 in zip(labels, starts):
        cx = x0 + panel_w // 2
        (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(hdr, lbl, (cx - tw // 2, HDR_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
    return hdr


def build_info_bar(h: int, patient_id: str, row_idx: int,
                   gt_pct: float, pred_pct: float) -> np.ndarray:
    """Left-side info bar for one row."""
    bar = np.full((h, LABEL_W, 3), 245, dtype=np.uint8)
    short = patient_id[-5:]   # e.g. "90023"
    lines = [
        (f"#{row_idx}  {short}", 0.40, (50, 50, 50)),
        (f"GT:   {gt_pct:5.1f}%", 0.38, (0, 120, 0)),
        (f"Pred: {pred_pct:5.1f}%", 0.38, (0, 0, 160)),
    ]
    y = 20
    for txt, sc, col in lines:
        cv2.putText(bar, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
        y += 20
    return bar


def make_row(img_path: str, seg_path: str,
             pred_mask: np.ndarray, patient_id: str,
             row_idx: int) -> np.ndarray:
    """Build one full row: info_bar | input | GT | pred."""
    img    = cv2.imread(img_path)
    gt_raw = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    if img is None or gt_raw is None:
        raise FileNotFoundError(f"Cannot read {img_path} or {seg_path}")
    gt = (gt_raw > 127).astype(np.uint8)

    p1 = img.copy()
    p2 = overlay(img, gt,        COLOR_GT,   ALPHA)
    p3 = overlay(img, pred_mask, COLOR_PRED, ALPHA)

    gap = np.full((img.shape[0], GAP_COL, 3), 180, dtype=np.uint8)
    panels = np.hstack([p1, gap, p2, gap, p3])

    info = build_info_bar(img.shape[0], patient_id,
                          row_idx,
                          gt.mean() * 100,
                          pred_mask.mean() * 100)
    gap_l = np.full((img.shape[0], GAP_COL, 3), 180, dtype=np.uint8)
    return np.hstack([info, gap_l, panels])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.cs_csv)
    df_eval = df[df["split"] == "val"].copy()
    if args.patients:
        df_eval = df_eval[df_eval["patient_id"].isin(args.patients)]

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.ckpt}")
    cfg = Config.fromfile(args.config)
    cfg.test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="PackSegInputs"),
    ]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = init_model(cfg, args.ckpt, device=device)
    model.eval()
    print(f"Model ready on {device}.\n")

    # ── Per-stain visualisation ───────────────────────────────────────────────
    for stain in args.stains:
        stain_df = df_eval[df_eval["stain"] == stain].copy()
        if stain_df.empty:
            print(f"[WARN] No eval patches for stain={stain}, skipping.")
            continue

        # Shuffle for random sampling
        stain_df = stain_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

        collected_rows = []
        skipped = 0

        for _, row in stain_df.iterrows():
            if len(collected_rows) >= args.n:
                break

            img_path = row["img_path"]
            seg_path = row["seg_path"]
            patient  = row["patient_id"]

            # Optional: skip all-background patches
            if args.tumor_only:
                gt_raw = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                if gt_raw is None or gt_raw.max() == 0:
                    skipped += 1
                    continue

            # Run inference
            result    = inference_model(model, img_path)
            pred_mask = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)

            idx = len(collected_rows) + 1
            panel = make_row(img_path, seg_path, pred_mask, patient, idx)
            collected_rows.append(panel)

            tumor_gt   = (cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE) > 127).mean() * 100
            tumor_pred = pred_mask.mean() * 100
            print(f"  [{stain}] {idx}/{args.n}  {Path(img_path).name}"
                  f"  GT={tumor_gt:.1f}%  Pred={tumor_pred:.1f}%")

        if not collected_rows:
            print(f"  [WARN] No suitable patches found for {stain}.")
            continue

        if skipped:
            print(f"  (skipped {skipped} background-only patches)")

        # ── Assemble figure ───────────────────────────────────────────────────
        panel_w = collected_rows[0].shape[1] - LABEL_W - GAP_COL  # total panels width
        # panel_w includes 3 × 512 + 2 × GAP_COL
        total_w = collected_rows[0].shape[1]

        header  = build_header(total_w, 512)
        h_gap   = np.full((GAP_ROW, total_w, 3), 160, dtype=np.uint8)

        parts = [header]
        for r in collected_rows:
            parts.append(h_gap)
            parts.append(r)

        figure = np.vstack(parts)

        # ── Save ──────────────────────────────────────────────────────────────
        suffix = "_tumor" if args.tumor_only else ""
        out_path = out_dir / f"{stain}_n{len(collected_rows)}{suffix}.png"
        cv2.imwrite(str(out_path), figure)
        print(f"  => Saved: {out_path}  ({figure.shape[1]}×{figure.shape[0]} px)\n")


if __name__ == "__main__":
    main()
