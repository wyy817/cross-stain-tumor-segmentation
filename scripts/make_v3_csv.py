"""
Generate patch_index_private_v3.csv from v2 patches (no re-extraction needed).

Strategy:
  - train tumor: cap to MAX_TUMOR_PER_PATIENT=2000 per patient (random sample)
  - train bg:    repeat rows until bg count ≈ tumor count (oversample)
  - val:         unchanged from v2

Usage:
    python scripts/make_v3_csv.py
    python scripts/make_v3_csv.py --max_tumor 2000 --seed 42
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_in',    default='/path/to/FYP_S2/data/patch_index_private_v2.csv')
    parser.add_argument('--csv_out',   default='/path/to/FYP_S2/data/patch_index_private_v3.csv')
    parser.add_argument('--max_tumor', type=int, default=2000, help='Max tumor patches per train patient')
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── Read v2 CSV ────────────────────────────────────────────────────────────
    with open(args.csv_in, newline='') as f:
        rows = list(csv.DictReader(f))

    val_rows   = [r for r in rows if r['split'] == 'val']
    train_rows = [r for r in rows if r['split'] == 'train']

    # Separate train tumor / bg by patient
    tumor_by_patient = defaultdict(list)
    bg_rows = []
    for r in train_rows:
        # A patch is "bg" if its mask is all-zero; we infer from existing label counts.
        # We re-use the stain field trick: all rows are HE, so check by
        # whether patient has bg candidates (we label by sampling logic).
        # Actually we don't have a "label" column — proxy: read mask and check.
        # Simpler: re-derive from original extraction logic via filename.
        # Since we can't easily check mask content here, we'll classify by
        # looking at whether the patient+position was a bg candidate.
        # Best proxy: just use patient to bucket tumor, then compare to bg.
        pass

    # ── Classify train rows as tumor vs bg ────────────────────────────────────
    # We need to know which rows are tumor vs bg.
    # The CSV doesn't have a label column, but we can check mask pixel values.
    # To avoid reading 32k images, we'll do a fast filename-based approach:
    # re-read a sample and classify all rows by checking the mask file.
    # For speed: check mask max value (0=bg, 255=tumor exists).

    print(f'Classifying {len(train_rows)} train rows (reading mask headers)...')
    tumor_by_patient = defaultdict(list)
    bg_all = []

    import cv2
    import numpy as np

    for i, r in enumerate(train_rows):
        if i % 2000 == 0:
            print(f'  {i}/{len(train_rows)}')
        mask = cv2.imread(r['seg_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.max() > 0:
            tumor_by_patient[r['patient_id']].append(r)
        else:
            bg_all.append(r)

    print(f'\nBefore cap:')
    for pid, rows_p in tumor_by_patient.items():
        print(f'  {pid}: {len(rows_p)} tumor')
    print(f'  bg total: {len(bg_all)}')

    # ── Cap tumor per patient ──────────────────────────────────────────────────
    train_tumor_final = []
    for pid, rows_p in tumor_by_patient.items():
        if len(rows_p) > args.max_tumor:
            rng.shuffle(rows_p)
            rows_p = rows_p[:args.max_tumor]
            print(f'  {pid}: capped to {args.max_tumor}')
        train_tumor_final.extend(rows_p)

    total_tumor = len(train_tumor_final)
    print(f'\nAfter cap: {total_tumor} tumor rows')

    # ── Oversample bg to match tumor count ────────────────────────────────────
    if len(bg_all) == 0:
        print('WARNING: no bg rows found, v3 CSV = same as tumor-capped only')
        bg_final = []
    else:
        repeat = max(1, total_tumor // len(bg_all))
        remainder = total_tumor - repeat * len(bg_all)
        bg_final = bg_all * repeat
        if remainder > 0:
            rng.shuffle(bg_all)
            bg_final += bg_all[:remainder]
        rng.shuffle(bg_final)
        print(f'bg: {len(bg_all)} rows × {repeat}+partial = {len(bg_final)} rows')

    tumor_pct = 100 * total_tumor / (total_tumor + len(bg_final)) if (total_tumor + len(bg_final)) > 0 else 0
    print(f'Train balance: {total_tumor} tumor + {len(bg_final)} bg = {total_tumor+len(bg_final)} total ({tumor_pct:.1f}% tumor)')

    # ── Write v3 CSV ──────────────────────────────────────────────────────────
    all_out = val_rows + train_tumor_final + bg_final
    fieldnames = ['img_path', 'seg_path', 'patient_id', 'stain', 'split']

    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(csv_out), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_out)

    print(f'\nWritten: {csv_out}')
    print(f'Total rows: {len(all_out)} (val={len(val_rows)}, train={total_tumor+len(bg_final)})')


if __name__ == '__main__':
    main()
