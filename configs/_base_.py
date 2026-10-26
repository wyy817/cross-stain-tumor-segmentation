"""
Shared base configuration for all FYP experiments.
=========================================================
Parameters marked with:
  # [LOCAL]  → value for local testing (RTX 5070, Windows)
  # [SERVER] → value for A40 server

All paths use the default project layout:
  /path/to/FYP_S2/    (local Windows)
  /path/to/FYP_S2/    (server — change PROJECT_ROOT in each config)
"""

# ── Common model settings ────────────────────────────────────────────────────
num_classes = 2          # 0: background, 1: tissue
img_size    = 512

# ── Data settings ────────────────────────────────────────────────────────────
csv_file = "/path/to/FYP_S2/data/patch_index_private_v7.csv"
# [LOCAL 4090]: /path/to/FYP_S2/data/patch_index_private_v7.csv
# v6 = v5 with proper 80/20 per-patient per-stain split (no data leakage)
# v7 = Phase 9 Plan A: train on HE+IHC only, val on HE+IHC+NCR (NCR = cross-stain test)

# ── Dataset preprocessing ─────────────────────────────────────────────────────
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

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

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RemapMask", mapping={255: 1}),
    dict(type="PackSegInputs"),
]

# ── Dataloader settings ──────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=8,          # [LOCAL: 2]  [SERVER: 8]
    num_workers=4,         # [LOCAL: 0]  [SERVER: 4]
    persistent_workers=True,   # [LOCAL: False]  [SERVER: True]
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="PrivateDataset",
        csv_file=csv_file,
        split="train",
        stains=["HE", "IHC"],   # Phase 9: NCR held out
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="PrivateDataset",
        csv_file=csv_file,
        split="val",
        stains=["HE", "IHC", "NCR"],
        pipeline=val_pipeline,
    ),
)
val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice"])
val_cfg       = dict(type="ValLoop")

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="PrivateDataset",
        csv_file=csv_file,
        split="val",
        stains=["HE"],
        pipeline=val_pipeline,
    ),
)
test_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice"])

# ── Optimizer & Scheduler ────────────────────────────────────────────────────
optim_wrapper = dict(
    type="AmpOptimWrapper",      # AMP (fp16) — works on CUDA; auto-disables on CPU
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=0.01,
    ),
    clip_grad=dict(max_norm=1.0),
)

# ── Training schedule ─────────────────────────────────────────────────────────
train_cfg = dict(type="IterBasedTrainLoop", max_iters=20000, val_interval=1000)
# val_interval=1000: val on HE+IHC+NCR every 1000 iters; best mDice checkpoint saved
# [LOCAL]  max_iters=200, val_interval=100  (for smoke test)
# [SERVER] max_iters=20000, val_interval=1000

test_cfg = dict(type="TestLoop")

# ── Logging & Checkpointing ───────────────────────────────────────────────────
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=5,
        save_best="mDice",
        rule="greater",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook", draw=False),
)

# Param scheduler: cosine LR decay
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
        end=20000,   # [SERVER: 20000]  [LOCAL: 200]
        eta_min=1e-6,
        by_epoch=False,
    ),
]

# ── Environment ───────────────────────────────────────────────────────────────
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="spawn", opencv_num_threads=0),
    dist_cfg=dict(backend="gloo"),   # gloo for Windows; nccl for Linux GPU
)

# Logging
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="SegLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)

log_processor = dict(by_epoch=False)
log_level = "INFO"
load_from = None
resume = False
default_scope = "mmseg"

# ── Shared model skeleton (decode head, data_preprocessor) ───────────────────
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,   # required by SegDataPreProcessor; 512 is divisible by 32
)

decode_head = dict(
    type="FPNHead",
    in_channels=[768, 384, 192, 96],   # matches ConchBackbone.out_channels
    in_index=[0, 1, 2, 3],
    feature_strides=[16, 32, 64, 128], # strides of feat0..feat3 w.r.t. 512 input
    channels=256,
    dropout_ratio=0.1,
    num_classes=num_classes,
    norm_cfg=dict(type="BN", requires_grad=True),
    align_corners=False,
    loss_decode=[
        dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        dict(type="DiceLoss", loss_weight=1.0),
    ],
)
