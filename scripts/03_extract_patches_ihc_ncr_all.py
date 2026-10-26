"""
Script 03: Extract IHC/NCR patches for ALL train patients (Phase 7 multi-stain training).

Reads all train rows from patch_index_private_v4.csv, extracts the corresponding
IHC and NCR patches via affine warp (HE.vp), then writes patch_index_private_v5.csv
= original HE rows + new IHC rows + new NCR rows.

Coordinate math: identical to 02_extract_patches_ihc_ncr.py.
  HE.vp gives M_he_40x (3×3): world_xy = M_he_40x @ [x_he_40x, y_he_40x, 1]
  IHC.vp = identity → IHC pixel space = world space
  NCR.vp assumed identity (same as IHC)
  At 20x: M_20x = M_he_40x with translation /2
  Warp: dst local (px,py) → IHC crop via T_crop_inv @ M_20x @ T_patch

Requires openslide-python and opencv.

Usage:
    python scripts/03_extract_patches_ihc_ncr_all.py
    python scripts/03_extract_patches_ihc_ncr_all.py --smoke_test     # 10 patches per patient
    python scripts/03_extract_patches_ihc_ncr_all.py --stains IHC     # skip NCR

Outputs:
    patches/private/ihc_images/{patient}_{x}_{y}.png   (appended to existing)
    patches/private/ncr_images/{patient}_{x}_{y}.png
    data/patch_index_private_v5.csv
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'data'))

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from reader import OpenSlideReader

# ── Constants ──────────────────────────────────────────────────────────────────

PATCH_SZ    = 512
RATIO_PATCH = 0.5   # 20x (relative to 40x base)
PAD         = 64    # extra margin when reading IHC/NCR crop


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_vp(path: str) -> np.ndarray:
    """Load .vp JSON → 3×3 float64 pose matrix."""
    with open(path) as f:
        data = json.load(f)
    pose = data[0]['pose']
    return np.array(pose, dtype=np.float64).reshape(3, 3)


def build_M_20x(M_40x: np.ndarray) -> np.ndarray:
    M = M_40x.copy()
    M[0, 2] /= 2.0
    M[1, 2] /= 2.0
    return M


def composite_rgba(rgba: np.ndarray) -> np.ndarray:
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        rgb   = rgba[:, :, :3].astype(np.float32)
        return (alpha * rgb + (1 - alpha) * 255).astype(np.uint8)
    return rgba[:, :, :3].astype(np.uint8) if rgba.ndim == 3 else rgba.astype(np.uint8)


def warp_patch(reader: OpenSlideReader, M_20x: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """Return RGB uint8 (512, 512, 3) for the IHC/NCR region aligned to HE patch."""
    corners_he = np.array([
        [x0,            y0           ],
        [x0 + PATCH_SZ, y0           ],
        [x0 + PATCH_SZ, y0 + PATCH_SZ],
        [x0,            y0 + PATCH_SZ],
    ], dtype=np.float64)
    corners_ihc = (M_20x @ np.hstack([corners_he, np.ones((4, 1))]).T).T[:, :2]

    x_min = int(np.floor(corners_ihc[:, 0].min())) - PAD
    y_min = int(np.floor(corners_ihc[:, 1].min())) - PAD
    x_max = int(np.ceil(corners_ihc[:, 0].max()))  + PAD
    y_max = int(np.ceil(corners_ihc[:, 1].max()))  + PAD
    crop_w = max(1, x_max - x_min)
    crop_h = max(1, y_max - y_min)

    raw      = reader.read(x_min, y_min, crop_w, crop_h, ratio=RATIO_PATCH)
    crop_rgb = composite_rgba(raw)

    T_patch    = np.array([[1, 0, x0    ], [0, 1, y0    ], [0, 0, 1]], dtype=np.float64)
    T_crop_inv = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    M_warp     = T_crop_inv @ M_20x @ T_patch

    return cv2.warpAffine(
        crop_rgb,
        M_warp[:2].astype(np.float32),
        (PATCH_SZ, PATCH_SZ),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract IHC/NCR patches for all train patients (Phase 7).')
    parser.add_argument('--data_dir',   default='/path/to/FYP_S2/data')
    parser.add_argument('--out_dir',    default='/path/to/FYP_S2/patches/private')
    parser.add_argument('--csv_in',     default='/path/to/FYP_S2/data/patch_index_private_v4.csv')
    parser.add_argument('--csv_out',    default='/path/to/FYP_S2/data/patch_index_private_v5.csv')
    parser.add_argument('--smoke_test', action='store_true', help='Only first 10 patches per patient')
    parser.add_argument('--stains',     nargs='+', default=['IHC', 'NCR'], choices=['IHC', 'NCR'])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    csv_in = Path(args.csv_in)
    if not csv_in.exists():
        sys.exit(f'ERROR: {csv_in} not found.')

    # Read ALL train rows, group by patient
    train_rows_by_patient: dict[str, list] = {}
    with open(str(csv_in), newline='') as f:
        for row in csv.DictReader(f):
            if row['split'] == 'train':
                train_rows_by_patient.setdefault(row['patient_id'], []).append(row)

    print(f'Train patches per patient: { {p: len(v) for p, v in train_rows_by_patient.items()} }')
    print(f'Stains to extract: {args.stains}')

    stain_dirs = {
        'IHC': out_dir / 'ihc_images',
        'NCR': out_dir / 'ncr_images',
    }
    for s in args.stains:
        stain_dirs[s].mkdir(parents=True, exist_ok=True)

    # Collect all new IHC/NCR rows
    new_cs_rows: list[dict] = []

    patients = sorted(train_rows_by_patient.keys())
    for patient in patients:
        he_rows = train_rows_by_patient[patient]
        patient_dir = data_dir / patient
        vp_path  = str(patient_dir / f'{patient}-HE.vp')
        ihc_path = str(patient_dir / f'{patient}-IHC.mrxs')
        ncr_path = str(patient_dir / f'{patient}-NCR.mrxs')

        print(f'\n[{patient}]  {len(he_rows)} train HE patches')

        M_40x = load_vp(vp_path)
        M_20x = build_M_20x(M_40x)
        print(f'  HE→IHC M_20x (2×3):\n    {M_20x[:2]}')

        readers = {}
        if 'IHC' in args.stains:
            readers['IHC'] = OpenSlideReader(ihc_path)
            print(f'  IHC size 40x: {readers["IHC"].width(1.0)} × {readers["IHC"].height(1.0)}')
        if 'NCR' in args.stains:
            readers['NCR'] = OpenSlideReader(ncr_path)
            print(f'  NCR size 40x: {readers["NCR"].width(1.0)} × {readers["NCR"].height(1.0)}')

        rows_to_process = he_rows[:10] if args.smoke_test else he_rows
        n_saved = {s: 0 for s in args.stains}

        for i, he_row in enumerate(rows_to_process):
            # Parse x0, y0 from filename: {patient}_{x0}_{y0}.png
            fname = Path(he_row['img_path']).name
            parts = fname.replace('.png', '').split('_')
            x0    = int(parts[-2])
            y0    = int(parts[-1])
            out_fn = f'{patient}_{x0}_{y0}.png'

            for stain in args.stains:
                out_path = stain_dirs[stain] / out_fn
                if out_path.exists():
                    # Already extracted (e.g. from previous run for val); reuse
                    pass
                else:
                    patch = warp_patch(readers[stain], M_20x, x0, y0)
                    cv2.imwrite(str(out_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                n_saved[stain] += 1
                new_cs_rows.append({
                    'img_path':   str(out_path).replace('\\', '/'),
                    'seg_path':   he_row['seg_path'].replace('\\', '/'),
                    'patient_id': patient,
                    'stain':      stain,
                    'split':      'train',
                })

            if (i + 1) % 500 == 0:
                print(f'  ... {i+1}/{len(rows_to_process)} done')

        print(f'  Saved/reused: { {s: n_saved[s] for s in args.stains} }')

    # ── Write combined v5 CSV: original HE rows + new IHC/NCR rows ──────────────
    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['img_path', 'seg_path', 'patient_id', 'stain', 'split']

    # Read original HE rows (v4)
    he_rows_all = []
    with open(str(csv_in), newline='') as f:
        for row in csv.DictReader(f):
            he_rows_all.append({
                'img_path':   row['img_path'].replace('\\', '/'),
                'seg_path':   row['seg_path'].replace('\\', '/'),
                'patient_id': row['patient_id'],
                'stain':      row['stain'],
                'split':      row['split'],
            })

    all_rows = he_rows_all + new_cs_rows
    with open(str(csv_out), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'\n=== CSV written: {csv_out} ===')
    print(f'  HE train:  {sum(1 for r in all_rows if r["stain"] == "HE")}')
    for s in args.stains:
        print(f'  {s} train:  {sum(1 for r in new_cs_rows if r["stain"] == s)}')
    print(f'  Total:     {len(all_rows)}')


if __name__ == '__main__':
    main()
