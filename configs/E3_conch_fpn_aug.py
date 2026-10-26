"""
Experiment 3: CONCH (frozen) + FPNHead + OD-Space Stain Augmentation
=======================================================================
Main contribution: adds Macenko OD-space stain augmentation to E2.
Trains on H&E with synthetic stain variation → better cross-stain transfer.

Training: H&E + IHC patches (v7 split) + StainAugment(sigma=0.05); evaluated on H&E, IHC and the held-out NCR stain
"""

_base_ = "./_base_.py"

# ── Paths — change for server ─────────────────────────────────────────────────
work_dir         = "/path/to/FYP_S2/work_dirs/E3_conch_fpn_aug_v7"
conch_checkpoint = "/path/to/FYP_S2/weights/pytorch_model.bin"
# [LOCAL 4090]: paths above
# v7 = Phase 9 Plan A: train HE+IHC, test NCR cross-stain

# ── Model (same as E2) ────────────────────────────────────────────────────────
model = dict(
    type="EncoderDecoder",
    data_preprocessor={{_base_.data_preprocessor}},
    backbone=dict(
        type="ConchBackbone",
        checkpoint_path=conch_checkpoint,
        frozen=True,
        input_size=512,
        patch_size=16,
        pyramid_channels=(384, 192, 96),
    ),
    decode_head={{_base_.decode_head}},
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

# ── Optimizer ─────────────────────────────────────────────────────────────────
optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=2e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0),
)

# ── Training pipeline WITH StainAugment ───────────────────────────────────────
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RemapMask", mapping={255: 1}),
    dict(type="RandomFlip", prob=0.5, direction=["horizontal", "vertical"]),
    dict(type="RandomRotate", prob=0.3, degree=(-90, 90), pad_val=0, seg_pad_val=0),
    # ── OD-space Macenko stain augmentation (main contribution) ──
    dict(
        type="StainAugment",
        sigma=0.05,   # perturbation magnitude
        prob=0.5,     # applied to 50% of patches
    ),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=16,
        contrast_range=(0.9, 1.1),
        saturation_range=(0.9, 1.1),
        hue_delta=5,
    ),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=8,    # [LOCAL: 2]  [SERVER: 8]
    num_workers=4,   # [LOCAL: 0]  [SERVER: 4]
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="PrivateDataset",
        csv_file={{_base_.csv_file}},
        split="train",
        stains=["HE", "IHC", "NCR"],
        pipeline=train_pipeline,
    ),
)

# ── Extended schedule (CONCH frozen head needs more iters to converge) ────────
train_cfg = dict(type="IterBasedTrainLoop", max_iters=20000, val_interval=1000)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-3,
        end_factor=1.0,
        begin=0,
        end=500,
        by_epoch=False,
    ),
    dict(
        type="CosineAnnealingLR",
        begin=500,
        end=20000,
        eta_min=1e-6,
        by_epoch=False,
    ),
]
