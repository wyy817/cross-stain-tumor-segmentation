"""Parse mmseg training log (.out) into CSV files.

Usage:
    python scripts/parse_training_log.py 1352766.out
    python scripts/parse_training_log.py 1352766.out --out_dir logs/

Outputs:
    <name>_train.csv   -- per-iteration training metrics
    <name>_val.csv     -- per-checkpoint validation metrics
"""

import re
import csv
import sys
import os
import argparse

# Regex patterns
TRAIN_RE = re.compile(
    r'Iter\(train\)\s+\[\s*(\d+)/\d+\]'
    r'.*?lr:\s*([\d.e+-]+)'
    r'.*?loss:\s*([\d.e+-]+)'
    r'.*?decode\.loss_ce:\s*([\d.e+-]+)'
    r'.*?decode\.loss_dice:\s*([\d.e+-]+)'
    r'.*?decode\.acc_seg:\s*([\d.e+-]+)'
)

VAL_RE = re.compile(
    r'Saving checkpoint at (\d+) iterations'
)

VAL_METRIC_RE = re.compile(
    r'Iter\(val\)\s+\[1076/1076\]'
    r'.*?aAcc:\s*([\d.]+)'
    r'.*?mIoU:\s*([\d.]+)'
    r'.*?mAcc:\s*([\d.]+)'
    r'.*?mDice:\s*([\d.]+)'
)

TIMESTAMP_RE = re.compile(r'^(\d{2}/\d{2} \d{2}:\d{2}:\d{2})')


def parse_log(log_path):
    train_rows = []
    val_rows = []

    # We need to pair "Saving checkpoint at N" with the subsequent val summary
    pending_ckpt_iter = None

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # Training iter
            m = TRAIN_RE.search(line)
            if m:
                ts_m = TIMESTAMP_RE.match(line)
                train_rows.append({
                    'timestamp': ts_m.group(1) if ts_m else '',
                    'iter': int(m.group(1)),
                    'lr': float(m.group(2)),
                    'loss': float(m.group(3)),
                    'loss_ce': float(m.group(4)),
                    'loss_dice': float(m.group(5)),
                    'acc_seg': float(m.group(6)),
                })
                continue

            # Checkpoint saved → remember iter number
            m = VAL_RE.search(line)
            if m:
                pending_ckpt_iter = int(m.group(1))
                continue

            # Val summary line
            m = VAL_METRIC_RE.search(line)
            if m:
                ts_m = TIMESTAMP_RE.match(line)
                val_rows.append({
                    'timestamp': ts_m.group(1) if ts_m else '',
                    'ckpt_iter': pending_ckpt_iter if pending_ckpt_iter is not None else '',
                    'aAcc': float(m.group(1)),
                    'mIoU': float(m.group(2)),
                    'mAcc': float(m.group(3)),
                    'mDice': float(m.group(4)),
                })
                pending_ckpt_iter = None  # reset

    return train_rows, val_rows


def write_csv(rows, path, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log', help='Path to .out log file')
    parser.add_argument('--out_dir', default=None, help='Output directory (default: same as log)')
    args = parser.parse_args()

    log_path = args.log
    base = os.path.splitext(os.path.basename(log_path))[0]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(log_path))
    os.makedirs(out_dir, exist_ok=True)

    train_rows, val_rows = parse_log(log_path)

    train_csv = os.path.join(out_dir, f'{base}_train.csv')
    val_csv   = os.path.join(out_dir, f'{base}_val.csv')

    write_csv(train_rows, train_csv, ['timestamp', 'iter', 'lr', 'loss', 'loss_ce', 'loss_dice', 'acc_seg'])
    write_csv(val_rows,   val_csv,   ['timestamp', 'ckpt_iter', 'aAcc', 'mIoU', 'mAcc', 'mDice'])

    # Quick summary
    if val_rows:
        best = max(val_rows, key=lambda r: r['mDice'])
        print(f"\nBest val: mDice={best['mDice']:.2f}  mIoU={best['mIoU']:.2f}  @ iter {best['ckpt_iter']}")


if __name__ == '__main__':
    main()
