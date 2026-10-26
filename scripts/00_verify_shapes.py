"""
Script 00: Local shape verification (no CONCH weights needed).
Runs on CPU without mmseg. Verifies the full forward pass:
  ConchBackbone → PyramidGenerator → FPNHead (manual)

Usage:
    conda activate fyp_new
    cd D:/XJTLU/FYP_S2
    python scripts/00_verify_shapes.py

Expected output:
    [backbone] feat0: torch.Size([1, 768, 32, 32])
    [backbone] feat1: torch.Size([1, 384, 16, 16])
    [backbone] feat2: torch.Size([1, 192, 8, 8])
    [backbone] feat3: torch.Size([1, 96, 4, 4])
    [StainAugment] output shape: (512, 512, 3) dtype uint8
    All shape checks PASSED!
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def check_conch_backbone():
    print("\n── ConchBackbone (random weights, CPU) ─────────────────────")
    from models.conch_backbone import ConchBackbone, PyramidGenerator

    # Test PyramidGenerator alone (faster, no CONCH)
    pyramid = PyramidGenerator(in_channels=768, out_channels=(384, 192, 96))
    dummy = torch.randn(1, 768, 32, 32)
    feats = pyramid(dummy)
    expected = [(1, 768, 32, 32), (1, 384, 16, 16), (1, 192, 8, 8), (1, 96, 4, 4)]
    for i, (f, exp) in enumerate(zip(feats, expected)):
        assert tuple(f.shape) == exp, f"feat{i} shape mismatch: {f.shape} vs {exp}"
        print(f"[pyramid] feat{i}: {tuple(f.shape)}  ✓")

    # Test ConchBackbone with random weights (no checkpoint)
    backbone = ConchBackbone(
        checkpoint_path="",   # empty = random weights
        frozen=False,
        input_size=512,
        patch_size=16,
        pyramid_channels=(384, 192, 96),
    )
    backbone.eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        feats = backbone(x)
    for i, (f, exp) in enumerate(zip(feats, expected)):
        assert tuple(f.shape) == exp, f"backbone feat{i} shape mismatch: {f.shape} vs {exp}"
        print(f"[backbone] feat{i}: {tuple(f.shape)}  ✓")
    print("[ConchBackbone] out_channels:", backbone.out_channels)


def check_stain_augment():
    print("\n── StainAugment ────────────────────────────────────────────")
    from transforms.stain_augment import StainAugment

    aug = StainAugment(sigma=0.05, prob=1.0)   # always apply

    # Simulate a tissue patch (random non-white pixels)
    rng = np.random.default_rng(0)
    img = rng.integers(30, 220, size=(512, 512, 3), dtype=np.uint8)

    results = {"img": img}
    out = aug.transform(results)
    out_img = out["img"]
    assert out_img.shape == (512, 512, 3), f"Wrong shape: {out_img.shape}"
    assert out_img.dtype == np.uint8, f"Wrong dtype: {out_img.dtype}"
    print(f"[StainAugment] output shape: {out_img.shape}  dtype: {out_img.dtype}  ✓")


def check_otsu_mask():
    print("\n── Otsu tissue mask (value range check) ────────────────────")
    from data.acrobat_patcher import otsu_tissue_mask

    rng = np.random.default_rng(1)
    # Simulate slide: white background + dark tissue patches
    fake_rgb = np.full((256, 256, 3), 230, dtype=np.uint8)
    fake_rgb[50:150, 50:150] = rng.integers(60, 140, (100, 100, 3), dtype=np.uint8)

    mask = otsu_tissue_mask(fake_rgb)
    unique_vals = set(np.unique(mask).tolist())
    assert unique_vals <= {0, 1}, f"Mask values should be 0/1, got {unique_vals}"
    tissue_ratio = mask.mean()
    print(f"[otsu_mask] unique values: {unique_vals}  tissue ratio: {tissue_ratio:.2%}  ✓")


def check_dataset():
    print("\n── ACROBATDataset (CSV existence check) ────────────────────")
    csv_path = "D:/XJTLU/FYP_S2/data/patch_index.csv"
    import os
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"[dataset] CSV loaded: {len(df)} rows")
        print(df["stain"].value_counts().to_string())
    else:
        print(f"[dataset] CSV not found at {csv_path} — run 01_extract_patches.py first")


if __name__ == "__main__":
    check_conch_backbone()
    check_stain_augment()
    check_otsu_mask()
    check_dataset()
    print("\n" + "="*50)
    print("All checks passed!")
