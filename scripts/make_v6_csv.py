"""
Generate patch_index_private_v6.csv from v5 (multi-stain).

Strategy:
  - For each (patient, stain) group: 80% train / 20% eval (random, seed=42)
  - Applies to HE, IHC, and NCR equally across all 5 patients
  - No data leakage: eval patches are strictly excluded from train

Usage:
    python scripts/make_v6_csv.py
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

CSV_IN  = '/path/to/FYP_S2/data/patch_index_private_v5.csv'
CSV_OUT = '/path/to/FYP_S2/data/patch_index_private_v6.csv'
EVAL_RATIO = 0.20
SEED = 42


def main():
    rng = random.Random(SEED)

    with open(CSV_IN, newline='') as f:
        rows = list(csv.DictReader(f))

    # Group by (patient_id, stain)
    groups = defaultdict(list)
    for r in rows:
        groups[(r['patient_id'], r['stain'])].append(r)

    train_rows, eval_rows = [], []

    print(f'{"Patient":<20} {"Stain":<6} {"Total":>6} {"Train":>6} {"Eval":>6}')
    print('-' * 50)

    for (patient, stain), group_rows in sorted(groups.items()):
        rng.shuffle(group_rows)
        n_eval = round(len(group_rows) * EVAL_RATIO)
        n_train = len(group_rows) - n_eval

        for r in group_rows[:n_train]:
            r['split'] = 'train'
            train_rows.append(r)
        for r in group_rows[n_train:]:
            r['split'] = 'val'
            eval_rows.append(r)

        print(f'{patient:<20} {stain:<6} {len(group_rows):>6} {n_train:>6} {n_eval:>6}')

    print('-' * 50)
    print(f'{"TOTAL":<20} {"":6} {len(rows):>6} {len(train_rows):>6} {len(eval_rows):>6}')

    # Summary by stain
    print()
    print('=== By stain ===')
    from collections import Counter
    train_stain = Counter(r['stain'] for r in train_rows)
    eval_stain  = Counter(r['stain'] for r in eval_rows)
    for stain in ['HE', 'IHC', 'NCR']:
        print(f'  {stain}: train={train_stain[stain]}, eval={eval_stain[stain]}')

    all_out = train_rows + eval_rows
    fieldnames = ['img_path', 'seg_path', 'patient_id', 'stain', 'split']

    out_path = Path(CSV_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_out)

    print(f'\nWritten: {out_path}')
    print(f'Total: {len(all_out)} (train={len(train_rows)}, eval={len(eval_rows)})')


if __name__ == '__main__':
    main()
