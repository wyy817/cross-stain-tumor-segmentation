"""
Script 03: Cross-stain evaluation.
Evaluates a trained checkpoint separately on each IHC stain
and prints a summary table.

Usage:
    python scripts/03_evaluate.py \
        --config  configs/E3_conch_fpn_aug.py \
        --ckpt    work_dirs/E3_conch_fpn_aug/best_mDice.pth \
        --stains  HE ER PGR HER2 KI67

Output:
    Stain   | mDice | mIoU
    --------|-------|------
    HE      | 0.xxx | 0.xxx
    ER      | 0.xxx | 0.xxx
    ...
    MEAN    | 0.xxx | 0.xxx
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# Register custom modules
import models.register  # noqa: F401

import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import METRICS
from mmseg.evaluation import IoUMetric


@METRICS.register_module()
class DetailedIoUMetric(IoUMetric):
    """IoUMetric that also returns per-class Dice/IoU in the result dict."""

    def compute_metrics(self, results):
        summary = super().compute_metrics(results)
        # results: list of (area_intersect, area_union, area_pred, area_label) tensors
        import torch as _torch
        n = len(results[0][0])  # num_classes from tensor shape
        total_intersect = _torch.zeros(n, dtype=_torch.float64)
        total_union     = _torch.zeros(n, dtype=_torch.float64)
        total_pred      = _torch.zeros(n, dtype=_torch.float64)
        total_label     = _torch.zeros(n, dtype=_torch.float64)
        for r in results:
            total_intersect += r[0]
            total_union     += r[1]
            total_pred      += r[2]
            total_label     += r[3]
        dice = (2 * total_intersect / (total_pred + total_label).clamp(min=1e-7) * 100)
        iou  = (total_intersect / total_union.clamp(min=1e-7) * 100)
        classes = ((self.dataset_meta or {}).get("classes") or [f"class_{i}" for i in range(n)])
        for i, name in enumerate(classes):
            summary[f"Dice.{name}"] = round(dice[i].item(), 4)
            summary[f"IoU.{name}"]  = round(iou[i].item(), 4)
        return summary

# PyTorch 2.6 changed weights_only default to True, but mmengine checkpoints
# contain numpy/HistoryBuffer objects that are blocked. Force weights_only=False
# globally so mmengine's checkpoint loader can deserialize saved state dicts.
_orig_torch_load = torch.load
def _torch_load_unsafe(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_unsafe


STAIN_SPLIT = {
    "HE":  "val",
    "IHC": "val",
    "NCR": "val",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-stain evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt",   required=True, help="Checkpoint path (.pth)")
    parser.add_argument(
        "--stains",
        nargs="+",
        default=["HE", "IHC", "NCR"],
        help="Stains to evaluate",
    )
    parser.add_argument(
        "--cs_csv",
        default="/path/to/FYP_S2/data/patch_index_private_cs.csv",
        help="CSV for cross-stain (IHC/NCR) patches",
    )
    parser.add_argument(
        "--out_json",
        default=None,
        help="Save results to JSON file (optional)",
    )
    return parser.parse_args()


def eval_one_stain(cfg, ckpt_path: str, stain: str, split: str, cs_csv: str) -> dict:
    """Run test loop for one stain, return metrics dict."""
    val_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations"),
        dict(type="RemapMask", mapping={255: 1}),
        dict(type="PackSegInputs"),
    ]
    # HE uses train CSV (v4); IHC/NCR use cross-stain CSV
    csv_file = cfg.train_dataloader.dataset.csv_file if stain == "HE" else cs_csv
    cfg.test_dataloader = dict(
        batch_size=1,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=dict(
            type="PrivateDataset",
            csv_file=csv_file,
            split=split,
            stains=[stain],
            pipeline=val_pipeline,
        ),
    )
    cfg.test_evaluator = dict(type="DetailedIoUMetric", iou_metrics=["mIoU", "mDice"])
    cfg.load_from = ckpt_path
    cfg.work_dir = str(Path(cfg.work_dir) / f"eval_{stain}")

    # Disable AMP on CPU
    if not torch.cuda.is_available():
        if hasattr(cfg, "optim_wrapper") and cfg.optim_wrapper.get("type") == "AmpOptimWrapper":
            cfg.optim_wrapper.type = "OptimWrapper"

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    return metrics


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    results = {}
    for stain in args.stains:
        split = STAIN_SPLIT.get(stain, "test")
        print(f"\n{'='*50}")
        print(f"Evaluating stain: {stain} (split={split})")
        print("="*50)
        try:
            metrics = eval_one_stain(cfg.copy(), args.ckpt, stain, split, args.cs_csv)
            results[stain] = {
                "mDice":      round(metrics.get("mDice", 0.0), 4),
                "mIoU":       round(metrics.get("mIoU",  0.0), 4),
                "bg_Dice":    round(metrics.get("Dice.background", metrics.get("Dice.Background", 0.0)), 4),
                "tumor_Dice": round(metrics.get("Dice.tumor",      metrics.get("Dice.Tumor",      0.0)), 4),
                "bg_IoU":     round(metrics.get("IoU.background",  metrics.get("IoU.Background",  0.0)), 4),
                "tumor_IoU":  round(metrics.get("IoU.tumor",       metrics.get("IoU.Tumor",       0.0)), 4),
            }
        except Exception as e:
            print(f"  [ERROR] {stain}: {e}")
            results[stain] = {"mDice": None, "mIoU": None,
                              "bg_Dice": None, "tumor_Dice": None,
                              "bg_IoU": None, "tumor_IoU": None}

    # Print summary table
    print(f"\n{'='*65}")
    print(f"{'Stain':<8} | {'mDice':>7} | {'mIoU':>7} | {'bg_Dice':>8} | {'tumor_Dice':>10} | {'tumor_IoU':>9}")
    print(f"{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*10}-+-{'-'*9}")
    dice_vals, iou_vals = [], []
    for stain in args.stains:
        r = results[stain]
        d = r["mDice"]
        u = r["mIoU"]
        bg  = r.get("bg_Dice")
        tum = r.get("tumor_Dice")
        tum_iou = r.get("tumor_IoU")
        print(f"{stain:<8} | {str(d):>7} | {str(u):>7} | {str(bg):>8} | {str(tum):>10} | {str(tum_iou):>9}")
        if d is not None:
            dice_vals.append(d)
            iou_vals.append(u)
    if dice_vals:
        print(f"{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*10}-+-{'-'*9}")
        print(
            f"{'MEAN':<8} | {sum(dice_vals)/len(dice_vals):>7.4f} | "
            f"{sum(iou_vals)/len(iou_vals):>7.4f}"
        )
    print("="*65)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.out_json}")


if __name__ == "__main__":
    main()
