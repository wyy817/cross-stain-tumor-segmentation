"""
Script 02: Train a segmentation model using mmseg Runner.

Usage:
    # Experiment 2 (CONCH + FPN, no aug):
    python scripts/02_train.py --config configs/E2_conch_fpn.py

    # Experiment 3 (CONCH + FPN + stain aug):
    python scripts/02_train.py --config configs/E3_conch_fpn_aug.py

    # Experiment 1 (ResNet baseline):
    python scripts/02_train.py --config configs/E1_resnet_fpn.py

    # Local smoke test (override to 200 iters, batch=1):
    python scripts/02_train.py --config configs/E2_conch_fpn.py --smoke_test

Server notes:
    [SERVER] Set --num_gpus for multi-GPU DDP:
    torchrun --nproc_per_node=2 scripts/02_train.py --config configs/E2_conch_fpn.py

Parameter overrides for server vs local are marked in configs/_base_.py.
"""

import sys
import os



# Ensure project root is in path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# ── CRITICAL: torch must be imported first on Windows so it registers its DLL
# directory before mmcv's _ext.pyd is loaded ────────────────────────────────
import torch  # noqa: E402

# ── CRITICAL: register custom modules BEFORE importing mmengine/mmseg Runner ─
import models.register  # noqa: F401

import argparse
from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="FYP train script")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Override config for a 200-iter local smoke test (batch=1, no AMP)",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options: e.g. --cfg-options train_dataloader.batch_size=4",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # Apply command-line overrides
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # Local smoke test: minimal settings to verify code runs
    if args.smoke_test:
        print("[smoke_test] Overriding to minimal settings for local testing...")
        cfg.train_cfg.max_iters = 200
        cfg.train_cfg.val_interval = 50
        cfg.train_dataloader.batch_size = 1
        cfg.train_dataloader.num_workers = 0
        cfg.train_dataloader.persistent_workers = False
        cfg.val_dataloader.num_workers = 0
        cfg.val_dataloader.persistent_workers = False
        # Disable AMP on CPU
        if not torch.cuda.is_available():
            cfg.optim_wrapper.type = "OptimWrapper"
            if "clip_grad" in cfg.optim_wrapper:
                pass  # keep clip_grad
        # Short schedulers for smoke test: linear 0→20, cosine 20→200
        if hasattr(cfg, "param_scheduler"):
            for sched in cfg.param_scheduler:
                if sched.get("type") == "CosineAnnealingLR":
                    sched["begin"] = 20
                    sched["end"] = 200
                elif sched.get("type") == "LinearLR":
                    sched["begin"] = 0
                    sched["end"] = 20

    if args.resume:
        cfg.resume = True

    # Build and run
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
