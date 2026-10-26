"""
Script 01: Extract patches from ACROBAT WSI slides.
Run this ONCE before training.

Usage (activate fyp_new first):
    python scripts/01_extract_patches.py

Local test mode (1 slide only):
    python scripts/01_extract_patches.py --test_run

Outputs:
    patches/images/*.png  — RGB patches
    patches/masks/*.png   — binary tissue masks (0/255)
    data/patch_index.csv  — index CSV for training
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wsi_dir",
        default="D:/XJTLU/Cross_Stain_Tumor_Seg_for_Pathology_Images/datasets/ACROBAT/raw/extracted/valid",
        help="Directory containing *_val.tif files",
    )
    parser.add_argument("--out_dir", default="D:/XJTLU/FYP_S2/patches")
    parser.add_argument("--csv_out", default="D:/XJTLU/FYP_S2/data/patch_index.csv")
    parser.add_argument("--level",    type=int,   default=2,    help="WSI level (2=1/4 res)")
    parser.add_argument("--patch_sz", type=int,   default=512)
    parser.add_argument("--stride",   type=int,   default=256)
    parser.add_argument("--thr",      type=float, default=0.30, help="Min tissue ratio")
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Process only 1 HE slide for quick local verification",
    )
    args = parser.parse_args()

    from data.acrobat_patcher import (
        parse_filename, patient_split, process_slide, read_tif_level
    )
    import csv
    from pathlib import Path
    from collections import Counter

    wsi_dir = Path(args.wsi_dir)
    out_dir = Path(args.out_dir)
    img_out = out_dir / "images"
    msk_out = out_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(wsi_dir.glob("*.tif"))
    slides = []
    pids_he = []
    for tf in tif_files:
        pid, stain = parse_filename(tf.name)
        if pid is None:
            continue
        slides.append((tf, pid, stain))
        if stain == "HE":
            pids_he.append(pid)

    print(f"Found {len(slides)} slides, {len(set(pids_he))} HE patients")

    if args.test_run:
        # Take only the first HE slide for quick smoke test
        slides = [(tf, pid, stain) for tf, pid, stain in slides if stain == "HE"][:1]
        print(f"[test_run] Processing only: {slides[0][0].name}")

    splits = patient_split(list(set(pids_he)), seed=42)

    all_rows = []
    for tf, pid, stain in slides:
        split = splits.get(pid, "test")
        print(f"\nProcessing [{stain}] patient {pid} (split={split}): {tf.name}")
        rows = process_slide(
            tif_path=str(tf),
            pid=pid,
            stain=stain,
            split=split,
            img_out=img_out,
            msk_out=msk_out,
            level=args.level,
            patch_sz=args.patch_sz,
            stride=args.stride,
            tissue_thr=args.thr,
        )
        all_rows.extend(rows)
        print(f"  → {len(rows)} patches")

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["img_path", "seg_path", "patient_id", "stain", "split"]
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*50}")
    print(f"Total patches: {len(all_rows)}")
    print(f"CSV: {csv_path}")
    counter = Counter((r["stain"], r["split"]) for r in all_rows)
    for (stain, split), cnt in sorted(counter.items()):
        print(f"  {stain:5s} | {split:5s} : {cnt}")


if __name__ == "__main__":
    main()
