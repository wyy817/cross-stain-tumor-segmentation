"""
CONCH ViT Backbone for mmsegmentation
=======================================
Wraps the CONCH CoCa visual encoder (ViT-B/16) as an mmseg backbone.
Outputs a 4-level feature pyramid via a lightweight PyramidGenerator.

Architecture:
    Input  [B, 3, 512, 512]
      ↓ CONCH ViT trunk (frozen)
    tokens [B, 1025, 768]  (1 CLS + 32×32 patches)
      ↓ reshape
    feat0  [B, 768, 32, 32]  (stride 16)
      ↓ PyramidGenerator (3× stride-2 conv)
    feat1  [B, 384, 16, 16]  (stride 32)
    feat2  [B, 192,  8,  8]  (stride 64)
    feat3  [B,  96,  4,  4]  (stride 128)

Returns: tuple(feat0, feat1, feat2, feat3)  ← mmseg FPNHead convention
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS

# ── add CONCH package to path ────────────────────────────────────────────────
_THIS = Path(__file__).resolve().parent.parent   # FYP_S2/
_CONCH_DIR = _THIS / "CONCH"
if str(_CONCH_DIR) not in sys.path:
    sys.path.insert(0, str(_CONCH_DIR))


class PyramidGenerator(nn.Module):
    """
    Lightweight 3-stage down-sampling pyramid.
    Converts a single feature map into 3 additional lower-resolution scales.

    in_channels  → mid_ch[0] → mid_ch[1] → mid_ch[2]
    each stage: Conv3×3(stride=2) + BN + ReLU
    """

    def __init__(self, in_channels: int = 768, out_channels: Tuple = (384, 192, 96)):
        super().__init__()
        ch = [in_channels] + list(out_channels)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(ch[i + 1]),
                    nn.ReLU(inplace=True),
                )
                for i in range(len(out_channels))
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = [x]
        for stage in self.stages:
            feats.append(stage(feats[-1]))
        return feats  # [feat0, feat1, feat2, feat3]


@MODELS.register_module()
class ConchBackbone(BaseModule):
    """
    CONCH ViT-B/16 backbone for mmsegmentation.

    Args:
        checkpoint_path (str): Path to 'pytorch_model.bin'.
            Set to empty string to skip weight loading (for testing).
        frozen (bool): Freeze the CONCH ViT trunk (recommended for small datasets).
        input_size (int): Expected spatial resolution of input images (default 512).
        patch_size (int): ViT patch size (must match CONCH = 16).
        pyramid_channels (tuple): Output channels for each pyramid stage.
        init_cfg: mmseg init_cfg (unused here, weights loaded manually).
    """

    def __init__(
        self,
        checkpoint_path: str = "",
        frozen: bool = True,
        input_size: int = 512,
        patch_size: int = 16,
        pyramid_channels: Tuple = (384, 192, 96),
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)

        from conch.open_clip_custom import create_model

        # CONCH model config (ViT-B/16, input overridden to input_size)
        model_cfg = "conch_ViT-B-16"
        ckpt = checkpoint_path if checkpoint_path else None

        # Build the full CONCH model (visual + text)
        # force_image_size ensures position embeddings are sized for input_size
        print(f"[ConchBackbone] Loading CONCH model (input_size={input_size}) ...")
        if ckpt and Path(ckpt).exists():
            conch = create_model(
                model_cfg,
                checkpoint_path=ckpt,
                device="cpu",
                force_image_size=input_size,
            )
        else:
            # No checkpoint: build model with random weights for local shape testing
            print(f"[ConchBackbone] WARNING: no valid checkpoint_path, using random weights!")
            from conch.open_clip_custom.coca_model import CoCa
            from conch.open_clip_custom.factory import read_state_dict
            import json
            cfg_path = _CONCH_DIR / "conch/open_clip_custom/model_configs/conch_ViT-B-16.json"
            with open(cfg_path) as f:
                cfg = json.load(f)
            cfg["vision_cfg"]["image_size"] = input_size
            cfg.pop("custom_text", None)
            conch = CoCa(**cfg)

        # Extract the ViT trunk (trunk.forward = trunk.forward_features, see CONCH factory.py)
        self.vit = conch.visual.trunk
        del conch   # free memory (text, poolers, etc.)

        if frozen:
            for p in self.vit.parameters():
                p.requires_grad_(False)
            print("[ConchBackbone] ViT trunk FROZEN (only pyramid + head trained)")
        else:
            print("[ConchBackbone] ViT trunk NOT frozen")

        self.patch_size = patch_size
        self.input_size = input_size
        self.h = self.w = input_size // patch_size   # 32 for 512/16

        vit_dim = self.vit.num_features  # 768 for ViT-B
        self.pyramid = PyramidGenerator(vit_dim, pyramid_channels)

        # Expose for FPNHead in_channels
        self.out_channels = (vit_dim,) + tuple(pyramid_channels)  # (768, 384, 192, 96)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: [B, 3, H, W] — expected H=W=input_size (default 512)

        Returns:
            tuple of 4 feature maps:
                [B, 768, 32, 32], [B, 384, 16, 16],
                [B, 192,  8,  8], [B,  96,  4,  4]
        """
        # CONCH trunk: forward_features → [B, N+1, C] (CLS + patch tokens)
        tokens = self.vit(x)  # trunk.forward == trunk.forward_features

        # Handle timm API differences (0.9.x returns tensor, 1.x may return dict)
        if isinstance(tokens, dict):
            tokens = tokens.get("x", tokens.get("x_norm_patchtokens", None))
            if tokens is None:
                raise RuntimeError("Unexpected timm forward_features output format")

        # Remove CLS token if present
        expected_n = self.h * self.w
        if tokens.shape[1] == expected_n + 1:
            tokens = tokens[:, 1:]  # [B, N, C]
        elif tokens.shape[1] != expected_n:
            # Dynamic size: compute h, w from actual token count
            n = tokens.shape[1]
            import math
            h = w = int(math.isqrt(n))
            assert h * w == n, f"Token count {n} is not a perfect square"
            self.h, self.w = h, w

        B, N, C = tokens.shape
        # Reshape to spatial feature map
        feat0 = tokens.reshape(B, self.h, self.w, C).permute(0, 3, 1, 2).contiguous()
        # [B, C, h, w] = [B, 768, 32, 32]

        feats = self.pyramid(feat0)
        return tuple(feats)
