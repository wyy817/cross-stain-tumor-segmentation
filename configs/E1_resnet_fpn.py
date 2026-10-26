"""
Experiment 1: ResNet-50 + FPNHead Baseline
============================================
Traditional CNN baseline — no CONCH, no stain augmentation.
Serves as the lower-bound reference for cross-stain generalization.

Training: H&E + IHC patches (v7 split); evaluated on H&E, IHC and the held-out NCR stain
"""

_base_ = "./_base_.py"

# ── Work directory ────────────────────────────────────────────────────────────
work_dir = "/path/to/FYP_S2/work_dirs/E1_resnet_fpn_v7"
# [LOCAL 4090]: /path/to/FYP_S2/work_dirs/E1_resnet_fpn_v7
# v7 = Phase 9 Plan A: train HE+IHC, test NCR cross-stain

# ── Model ─────────────────────────────────────────────────────────────────────
model = dict(
    type="EncoderDecoder",
    data_preprocessor={{_base_.data_preprocessor}},
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,    # -1 = nothing frozen (full fine-tune)
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
        init_cfg=dict(
            type="Pretrained",
            checkpoint="torchvision://resnet50",  # ImageNet pretrain
        ),
    ),
    decode_head=dict(
        type="FPNHead",
        in_channels=[256, 512, 1024, 2048],   # ResNet-50 FPN channels
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],       # ResNet-50 strides for stages 0-3
        channels=256,
        dropout_ratio=0.1,
        num_classes={{_base_.num_classes}},
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0,
                 class_weight=[5.0, 1.0]),  # bg weight×5: counter train-set tumor dominance (98%)
            dict(type="DiceLoss", loss_weight=1.0),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

# ── Training pipeline (no stain augmentation) ─────────────────────────────────
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RemapMask", mapping={255: 1}),
    dict(type="RandomFlip", prob=0.5, direction=["horizontal", "vertical"]),
    dict(
        type="RandomRotate",
        prob=0.3,
        degree=(-90, 90),
        pad_val=0,
        seg_pad_val=0,
    ),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10,
    ),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=16,   # [LOCAL 4090: 16]  [SERVER: 8]
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
