"""
Script 01b: Extract HE patches from private dataset (SHE.kfb + ASHE.kfb).
Run once before training (Phase 2).

Usage:
    python scripts/01_extract_patches_private.py
    python scripts/01_extract_patches_private.py --smoke_test   # 1 patient, ~60 patches
    python scripts/01_extract_patches_private.py --fold 0       # default: LOOCV fold 0 (val=023)

Outputs:
    patches/private/images/{patient}_{x}_{y}.png  — RGB 512x512
    patches/private/masks/{patient}_{x}_{y}.png   — binary 512x512 (0/255)
    data/patch_index_private.csv                  — img_path, seg_path, patient_id, stain, split
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'data'))  # for reader.py's internal: zero.kfbparser, utils

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from reader import KfbReader, OpenSlideReader

# ── Constants ──────────────────────────────────────────────────────────────────

PATIENTS = [
    'PATIENT_1', 'PATIENT_2', 'PATIENT_3',
    'PATIENT_4', 'PATIENT_5',
]

# ASHE top-left corner in SHE 40x pixel space (confirmed by Phase 1 / 00b_find_ashe_offset.py)
ASHE_OFFSET_40X = {
    'PATIENT_1': (22688, 51872),
    'PATIENT_2': (0, 0),
    'PATIENT_3': (0, 0),
    'PATIENT_4': (0, 0),
    'PATIENT_5': (0, 0),
}

# LOOCV: fold index → val patient
LOOCV_FOLDS = [
    'PATIENT_1',  # fold 0
    'PATIENT_2',  # fold 1
    'PATIENT_3',  # fold 2
    'PATIENT_4',  # fold 3
    'PATIENT_5',  # fold 4
]

RATIO_PATCH         = 0.5    # 20x (relative to 40x base)
PATCH_SZ            = 512    # pixels at 20x
STRIDE              = 256    # pixels at 20x (50% overlap)
TISSUE_THR          = 0.30   # min tissue ratio for tumor patches
BG_TISSUE_THR       = 0.10   # relaxed threshold for bg candidates (more bg from slide edges)
RATIO_THUMB         = 0.025  # for pre-filtering (~2190×5177 for full SHE 40x)
BG_RATIO            = 2.0    # target background patches per tumor patch
MAX_TUMOR_PER_PATIENT = 8000  # cap per train patient to prevent 99% tumor imbalance


# ── Thumbnail helpers ──────────────────────────────────────────────────────────

def read_tissue_map(she_reader) -> np.ndarray:
    """Read SHE thumbnail → float32 tissue mask (1=tissue, 0=white bg)."""
    w = she_reader.width(RATIO_THUMB)
    h = she_reader.height(RATIO_THUMB)
    thumb = she_reader.read(0, 0, w, h, RATIO_THUMB)
    if thumb.ndim == 3 and thumb.shape[2] == 4:
        # Composite RGBA over white background so transparent=white (not tissue)
        alpha = thumb[:, :, 3:4].astype(np.float32) / 255.0
        rgb   = thumb[:, :, :3].astype(np.float32)
        thumb = (alpha * rgb + (1.0 - alpha) * 255.0).astype(np.uint8)
    if thumb.ndim == 3:
        gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    else:
        gray = thumb  # already grayscale (low-res kfb tile)
    return (gray < 220).astype(np.float32)


def read_tumor_map(ashe_reader: KfbReader) -> np.ndarray:
    """Read ASHE thumbnail → float32 tumor mask (1=tumor annotation, 0=none)."""
    w = ashe_reader.width(RATIO_THUMB)
    h = ashe_reader.height(RATIO_THUMB)
    thumb = ashe_reader.read(0, 0, w, h, RATIO_THUMB)
    if thumb.ndim == 3:
        return (thumb.max(axis=-1) > 0).astype(np.float32)
    return (thumb > 0).astype(np.float32)


def safe_region(mask2d: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """Extract region from 2D mask with zero-padding for out-of-bounds coords."""
    mh, mw = mask2d.shape
    out = np.zeros((h, w), dtype=np.float32)
    sy = max(0, -y0);  ey = min(h, mh - y0)
    sx = max(0, -x0);  ex = min(w, mw - x0)
    if ey > sy and ex > sx:
        out[sy:ey, sx:ex] = mask2d[y0 + sy: y0 + ey, x0 + sx: x0 + ex]
    return out


# ── Patch-level helpers ────────────────────────────────────────────────────────

def read_patch_and_mask(
    she_reader: KfbReader,
    ashe_reader: KfbReader,
    ashe_offset_40x: tuple,
    x0_20x: int,
    y0_20x: int,
) -> tuple:
    """
    Read SHE RGB patch and binary tumor mask, both 512×512 uint8 at 20x.

    Coordinate mapping:
      ASHE is stored with its own (0,0) origin.
      Its top-left corner sits at (ox_40x, oy_40x) in SHE 40x space.
      At 20x (ratio=0.5): ashe_x0 = she_x0 - ox_40x * 0.5
    """
    # SHE patch (RGB or RGBA) — composite over white for OpenSlide transparency
    she = she_reader.read(x0_20x, y0_20x, PATCH_SZ, PATCH_SZ, RATIO_PATCH)
    if she.ndim == 3 and she.shape[2] == 4:
        alpha = she[:, :, 3:4].astype(np.float32) / 255.0
        she = (alpha * she[:, :, :3].astype(np.float32) + (1.0 - alpha) * 255.0).astype(np.uint8)

    # ASHE patch — coords in ASHE 20x space
    ox_40x, oy_40x = ashe_offset_40x
    ax0 = x0_20x - int(ox_40x * RATIO_PATCH)
    ay0 = y0_20x - int(oy_40x * RATIO_PATCH)
    ashe = ashe_reader.read(ax0, ay0, PATCH_SZ, PATCH_SZ, RATIO_PATCH)
    if ashe.ndim == 3:
        ashe_gray = ashe.max(axis=-1)
    else:
        ashe_gray = ashe

    mask = (ashe_gray > 0).astype(np.uint8) * 255
    return she.astype(np.uint8), mask


def tissue_ratio(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float((gray < 220).sum()) / gray.size


# ── Per-patient extraction ─────────────────────────────────────────────────────

def process_patient(
    patient: str,
    data_dir: Path,
    seg_dir: Path,
    img_out: Path,
    msk_out: Path,
    split: str,
    smoke_test: bool = False,
    seed: int = 42,
) -> list:
    rng = random.Random(seed)

    he_path   = str(data_dir / patient / f'{patient}-HE.mrxs')
    ashe_path = str(seg_dir / f'{patient}-ASHE.kfb')
    offset    = ASHE_OFFSET_40X[patient]

    print(f'  Opening HE:   {he_path}')
    print(f'  Opening ASHE: {ashe_path}')
    she_r  = OpenSlideReader(he_path)
    ashe_r = KfbReader(ashe_path, inter='nearest', fill=0)

    she_w = she_r.width(RATIO_PATCH)
    she_h = she_r.height(RATIO_PATCH)
    print(f'  SHE 20x: {she_w} × {she_h}')

    print('  Building tissue map (thumbnail)...')
    tissue_map = read_tissue_map(she_r)
    print('  Building tumor map (thumbnail)...')
    tumor_map  = read_tumor_map(ashe_r)
    print(f'  tissue_map {tissue_map.shape}, tumor_map {tumor_map.shape}')
    print(f'  tissue coverage: {tissue_map.mean():.3f}, tumor coverage: {tumor_map.mean():.3f}')

    # Thumbnail block size for one PATCH_SZ patch at 20x:
    #   PATCH_SZ at 20x = PATCH_SZ*2 at 40x; in thumbnail: PATCH_SZ*2*RATIO_THUMB
    blk = max(1, int(PATCH_SZ * 2 * RATIO_THUMB))  # ~25 px at thumb scale

    tumor_cands = []
    bg_cands    = []

    xs = range(0, she_w - PATCH_SZ + 1, STRIDE)
    ys = range(0, she_h - PATCH_SZ + 1, STRIDE)

    for y0 in ys:
        for x0 in xs:
            # Tissue pre-filter (thumbnail) — use relaxed threshold to find bg candidates
            th_x = int(x0 * 2 * RATIO_THUMB)
            th_y = int(y0 * 2 * RATIO_THUMB)
            t_reg = tissue_map[th_y: th_y + blk, th_x: th_x + blk]
            t_mean = t_reg.mean() if t_reg.size > 0 else 0.0
            if t_mean < BG_TISSUE_THR:
                continue  # completely empty, skip

            # Tumor check in ASHE thumbnail
            ox_40x, oy_40x = offset
            ax_th = int((x0 * 2 - ox_40x) * RATIO_THUMB)
            ay_th = int((y0 * 2 - oy_40x) * RATIO_THUMB)
            tm_reg = safe_region(tumor_map, ay_th, ax_th, blk, blk)

            if tm_reg.mean() > 0:
                if t_mean >= TISSUE_THR:  # tumor patches require strict tissue threshold
                    tumor_cands.append((x0, y0))
            else:
                bg_cands.append((x0, y0))  # bg uses relaxed threshold

    print(f'  Candidates (thumbnail): {len(tumor_cands)} tumor, {len(bg_cands)} bg')

    if smoke_test:
        tumor_cands = tumor_cands[:20]
        bg_cands    = bg_cands[:40]

    # Cap tumor patches per patient (prevents train patients with 99% tumor)
    if split == 'train' and len(tumor_cands) > MAX_TUMOR_PER_PATIENT:
        rng.shuffle(tumor_cands)
        tumor_cands = tumor_cands[:MAX_TUMOR_PER_PATIENT]
        print(f'  Tumor capped at {MAX_TUMOR_PER_PATIENT}')

    # Downsample background to BG_RATIO × tumor count
    max_bg = int(len(tumor_cands) * BG_RATIO)
    if len(bg_cands) > max_bg:
        rng.shuffle(bg_cands)
        bg_cands = bg_cands[:max_bg]

    rows = []

    def save_patch(x0: int, y0: int) -> dict:
        she_rgb, mask = read_patch_and_mask(she_r, ashe_r, offset, x0, y0)
        if tissue_ratio(she_rgb) < TISSUE_THR:
            return None
        fname = f'{patient}_{x0}_{y0}.png'
        cv2.imwrite(str(img_out / fname), cv2.cvtColor(she_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(msk_out / fname), mask)
        return {
            'img_path':   str(img_out / fname),
            'seg_path':   str(msk_out / fname),
            'patient_id': patient,
            'stain':      'HE',
            'split':      split,
        }

    print('  Extracting tumor patches...')
    n_tumor = 0
    for x0, y0 in tumor_cands:
        row = save_patch(x0, y0)
        if row:
            rows.append(row)
            n_tumor += 1

    print(f'  Extracting bg patches (target ≤{max_bg})...')
    n_bg = 0
    for x0, y0 in bg_cands:
        row = save_patch(x0, y0)
        if row:
            rows.append(row)
            n_bg += 1

    print(f'  Saved: {n_tumor} tumor + {n_bg} bg = {len(rows)} total')
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='/path/to/FYP_S2/data')
    parser.add_argument('--out_dir',    default='/path/to/FYP_S2/patches/private_v2')
    parser.add_argument('--csv_out',    default='/path/to/FYP_S2/data/patch_index_private_v2.csv')
    parser.add_argument('--fold',       type=int, default=0,
                        help='LOOCV fold index 0-4 (0=val:023, 1=val:025, ...)')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Process val patient only, 20 tumor + 40 bg patches max')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    seg_dir  = data_dir / 'seg'
    out_dir  = Path(args.out_dir)
    img_out  = out_dir / 'images'
    msk_out  = out_dir / 'masks'
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    val_patient = LOOCV_FOLDS[args.fold]
    train_patients = [p for p in PATIENTS if p != val_patient]
    print(f'Fold {args.fold}: val={val_patient}')
    print(f'             train={train_patients}')

    process_order = [val_patient] + train_patients
    if args.smoke_test:
        process_order = [val_patient]

    all_rows = []
    for patient in process_order:
        split = 'val' if patient == val_patient else 'train'
        print(f'\n[{patient}]  split={split}')
        rows = process_patient(
            patient=patient,
            data_dir=data_dir,
            seg_dir=seg_dir,
            img_out=img_out,
            msk_out=msk_out,
            split=split,
            smoke_test=args.smoke_test,
            seed=args.seed,
        )
        all_rows.extend(rows)

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['img_path', 'seg_path', 'patient_id', 'stain', 'split']
    with open(str(csv_path), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'\n{"="*50}')
    print(f'Total patches saved: {len(all_rows)}')
    print(f'CSV: {csv_path}')
    for (split, stain), cnt in sorted(Counter((r['split'], r['stain']) for r in all_rows).items()):
        print(f'  {split:5s} | {stain}: {cnt}')


if __name__ == '__main__':
    main()
