"""
Script 02: Extract cross-stain (IHC, NCR) patches for val patients.
Uses HE patch positions from patch_index_private.csv + HE.vp affine transform.

Coordinate math:
  HE.vp gives M_he_40x (3x3): world_xy = M_he_40x @ [x_he_40x, y_he_40x, 1]
  IHC.vp = identity → IHC pixel space = world space
  NCR.vp absent → assumed identity (same as IHC)
  At 20x: M_20x = M_he_40x with translation halved
  Warp: dst local (px,py) → IHC crop = T_crop_inv @ M_20x @ T_patch @ [px,py,1]

Requires openslide-python.

Usage:
    python scripts/02_extract_patches_ihc_ncr.py
    python scripts/02_extract_patches_ihc_ncr.py --thumbnail_only
    python scripts/02_extract_patches_ihc_ncr.py --smoke_test    # 10 patches per patient
    python scripts/02_extract_patches_ihc_ncr.py --stains IHC    # skip NCR

Outputs:
    patches/private/ihc_images/{patient}_{x}_{y}.png
    patches/private/ncr_images/{patient}_{x}_{y}.png
    outputs/02_thumbnails/{patient}_ihc_vs_ncr.png
    data/patch_index_private_cs.csv
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

PATCH_SZ     = 512
RATIO_PATCH  = 0.5    # 20x (relative to 40x base)
PAD          = 64     # extra margin when reading IHC/NCR crop (accounts for rotation)
THUMB_RATIO  = 0.005  # 1/200 of 40x for quick side-by-side thumbnail


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_vp(path: str) -> np.ndarray:
    """Load .vp JSON file → 3×3 float64 pose matrix (row-major)."""
    with open(path) as f:
        data = json.load(f)
    pose = data[0]['pose']  # [r00,r01,tx, r10,r11,ty, 0,0,1]
    return np.array(pose, dtype=np.float64).reshape(3, 3)


def build_M_20x(M_40x: np.ndarray) -> np.ndarray:
    """Convert 40x pose matrix to 20x: same rotation, translation halved."""
    M = M_40x.copy()
    M[0, 2] /= 2.0
    M[1, 2] /= 2.0
    return M


def composite_rgba(rgba: np.ndarray) -> np.ndarray:
    """Composite RGBA over white background → RGB uint8."""
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        rgb   = rgba[:, :, :3].astype(np.float32)
        out   = alpha * rgb + (1.0 - alpha) * 255.0
        return out.astype(np.uint8)
    if rgba.ndim == 3 and rgba.shape[2] == 3:
        return rgba.astype(np.uint8)
    return rgba.astype(np.uint8)


def warp_patch(
    reader: OpenSlideReader,
    M_20x: np.ndarray,
    x0: int,
    y0: int,
) -> np.ndarray:
    """
    Read the IHC/NCR region that corresponds to HE 20x patch (x0, y0, 512×512).
    Returns RGB uint8 (512, 512, 3).
    """
    # 4 corners of HE 20x patch
    corners_he = np.array([
        [x0,           y0           ],
        [x0 + PATCH_SZ, y0           ],
        [x0 + PATCH_SZ, y0 + PATCH_SZ],
        [x0,           y0 + PATCH_SZ],
    ], dtype=np.float64)
    ones = np.ones((4, 1), dtype=np.float64)

    # Transform corners to IHC 20x space
    corners_ihc = (M_20x @ np.hstack([corners_he, ones]).T).T[:, :2]

    x_min = int(np.floor(corners_ihc[:, 0].min())) - PAD
    y_min = int(np.floor(corners_ihc[:, 1].min())) - PAD
    x_max = int(np.ceil(corners_ihc[:, 0].max()))  + PAD
    y_max = int(np.ceil(corners_ihc[:, 1].max()))  + PAD
    crop_w = max(1, x_max - x_min)
    crop_h = max(1, y_max - y_min)

    # Read IHC/NCR crop at 20x
    raw = reader.read(x_min, y_min, crop_w, crop_h, ratio=RATIO_PATCH)
    crop_rgb = composite_rgba(raw)

    # Warp matrix: dst local (px,py) → IHC crop coords
    #   M_warp = T_crop_inv @ M_20x @ T_patch
    T_patch    = np.array([[1, 0, x0    ], [0, 1, y0    ], [0, 0, 1]], dtype=np.float64)
    T_crop_inv = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    M_warp = T_crop_inv @ M_20x @ T_patch

    warped = cv2.warpAffine(
        crop_rgb,
        M_warp[:2].astype(np.float32),
        (PATCH_SZ, PATCH_SZ),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return warped  # RGB


def save_thumbnail(
    ihc_r: OpenSlideReader,
    ncr_r: OpenSlideReader,
    patient: str,
    out_dir: Path,
) -> None:
    """Save side-by-side IHC vs NCR thumbnail for visual alignment check."""
    ihc_w = ihc_r.width(THUMB_RATIO)
    ihc_h = ihc_r.height(THUMB_RATIO)
    ihc_rgba = ihc_r.read(0, 0, ihc_w, ihc_h, THUMB_RATIO)
    ihc_rgb  = composite_rgba(ihc_rgba)

    ncr_w = ncr_r.width(THUMB_RATIO)
    ncr_h = ncr_r.height(THUMB_RATIO)
    ncr_rgba = ncr_r.read(0, 0, ncr_w, ncr_h, THUMB_RATIO)
    ncr_rgb  = composite_rgba(ncr_rgba)

    # Resize NCR to same height as IHC for comparison
    if ncr_h != ihc_h:
        sc = ihc_h / max(ncr_h, 1)
        ncr_rgb = cv2.resize(ncr_rgb, (max(1, int(ncr_w * sc)), ihc_h))

    sep = np.full((ihc_h, 10, 3), 180, dtype=np.uint8)
    canvas = np.hstack([ihc_rgb, sep, ncr_rgb])

    out_path = out_dir / f'{patient}_ihc_vs_ncr.png'
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f'  Thumbnail saved: {out_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract cross-stain IHC/NCR patches for val patients.')
    parser.add_argument('--data_dir',       default='/path/to/FYP_S2/data')
    parser.add_argument('--out_dir',        default='/path/to/FYP_S2/patches/private')
    parser.add_argument('--csv_in',         default='/path/to/FYP_S2/data/patch_index_private.csv')
    parser.add_argument('--csv_out',        default='/path/to/FYP_S2/data/patch_index_private_cs.csv')
    parser.add_argument('--thumb_dir',      default='/path/to/FYP_S2/outputs/02_thumbnails')
    parser.add_argument('--thumbnail_only', action='store_true',
                        help='Only generate thumbnail comparisons, no patch extraction')
    parser.add_argument('--smoke_test',     action='store_true',
                        help='Only first 10 patches per patient')
    parser.add_argument('--stains',         nargs='+', default=['IHC', 'NCR'],
                        choices=['IHC', 'NCR'])
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    out_dir   = Path(args.out_dir)
    thumb_dir = Path(args.thumb_dir)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    # Load HE patch index
    csv_in = Path(args.csv_in)
    if not csv_in.exists():
        sys.exit(f'ERROR: {csv_in} not found. Run 01_extract_patches_private.py first.')

    # Group val patches by patient
    val_patches: dict = {}
    with open(str(csv_in), newline='') as f:
        for row in csv.DictReader(f):
            if row['split'] == 'val':
                p = row['patient_id']
                val_patches.setdefault(p, []).append(row)

    print(f'Val patches per patient: { {p: len(v) for p, v in val_patches.items()} }')
    print(f'Stains to extract: {args.stains}')

    # Output dirs for each stain
    stain_dirs = {
        'IHC': out_dir / 'ihc_images',
        'NCR': out_dir / 'ncr_images',
    }
    if not args.thumbnail_only:
        for s in args.stains:
            stain_dirs[s].mkdir(parents=True, exist_ok=True)

    all_cs_rows = []

    for patient, he_rows in val_patches.items():
        patient_dir = data_dir / patient
        vp_path  = str(patient_dir / f'{patient}-HE.vp')
        ihc_path = str(patient_dir / f'{patient}-IHC.mrxs')
        ncr_path = str(patient_dir / f'{patient}-NCR.mrxs')

        print(f'\n[{patient}]  {len(he_rows)} val HE patches')

        M_40x = load_vp(vp_path)
        M_20x = build_M_20x(M_40x)
        print(f'  HE→IHC M_20x (2×3):\n    {M_20x[:2]}')

        # Open both readers (thumbnail always; patch extraction per stain)
        ihc_r = OpenSlideReader(ihc_path)
        ncr_r = OpenSlideReader(ncr_path)
        print(f'  IHC size 40x: {ihc_r.width(1.0)} × {ihc_r.height(1.0)}')
        print(f'  NCR size 40x: {ncr_r.width(1.0)} × {ncr_r.height(1.0)}')

        save_thumbnail(ihc_r, ncr_r, patient, thumb_dir)

        if args.thumbnail_only:
            continue

        rows_to_process = he_rows[:10] if args.smoke_test else he_rows

        n_saved = {'IHC': 0, 'NCR': 0}
        for i, he_row in enumerate(rows_to_process):
            # Parse x0, y0 from filename: {patient}_{x0}_{y0}.png
            fname  = Path(he_row['img_path']).name
            parts  = fname.replace('.png', '').split('_')
            x0     = int(parts[-2])
            y0     = int(parts[-1])
            out_fn = f'{patient}_{x0}_{y0}.png'

            for stain in args.stains:
                reader   = ihc_r if stain == 'IHC' else ncr_r
                out_path = stain_dirs[stain] / out_fn
                patch    = warp_patch(reader, M_20x, x0, y0)
                cv2.imwrite(str(out_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                n_saved[stain] += 1
                all_cs_rows.append({
                    'img_path':   str(out_path),
                    'seg_path':   he_row['seg_path'],
                    'patient_id': patient,
                    'stain':      stain,
                    'split':      'val',
                })

            if (i + 1) % 100 == 0:
                print(f'  ... {i+1}/{len(rows_to_process)} done')

        print(f'  Saved: { {s: n_saved[s] for s in args.stains} }')

    if not args.thumbnail_only:
        csv_out = Path(args.csv_out)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ['img_path', 'seg_path', 'patient_id', 'stain', 'split']
        with open(str(csv_out), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_cs_rows)
        print(f'\nCSV written: {csv_out}  ({len(all_cs_rows)} rows)')
        for stain in args.stains:
            n = sum(1 for r in all_cs_rows if r['stain'] == stain)
            print(f'  {stain}: {n} patches')


if __name__ == '__main__':
    main()
