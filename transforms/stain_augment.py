"""
OD-Space Macenko Stain Augmentation
=====================================
mmseg-compatible transform that perturbs stain concentrations in
optical-density space to simulate cross-stain appearance variation.

Algorithm (simplified Macenko):
  1. RGB → OD (optical density)
  2. Remove background pixels (low OD)
  3. SVD on OD matrix → 2 principal stain vectors
  4. Project all pixels onto stain plane → stain concentrations
  5. Perturb concentrations by multiplicative noise: α ~ Uniform(1-σ, 1+σ)
  6. Reconstruct OD → RGB

Usage in mmseg pipeline:
    dict(type='StainAugment', sigma=0.05, prob=0.5)
"""

import numpy as np
from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class StainAugment(BaseTransform):
    """
    OD-space Macenko stain augmentation.

    Args:
        sigma (float): Amplitude of stain perturbation.
            Each concentration channel is multiplied by Uniform(1-σ, 1+σ).
            Typical range: 0.02–0.10.
        prob (float): Probability to apply augmentation (default 0.5).
        io (float): Transmitted light intensity (default 255.0).
        beta (float): OD threshold for background removal (default 0.15).
    """

    def __init__(
        self,
        sigma: float = 0.05,
        prob: float = 0.5,
        io: float = 255.0,
        beta: float = 0.15,
    ):
        self.sigma = sigma
        self.prob = prob
        self.io = io
        self.beta = beta

    # ── core Macenko methods ─────────────────────────────────────────────────

    def _rgb2od(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB [H,W,3] uint8 → OD [H,W,3] float32."""
        rgb = rgb.astype(np.float32)
        rgb = np.clip(rgb, 1, self.io)
        return -np.log(rgb / self.io)

    def _od2rgb(self, od: np.ndarray) -> np.ndarray:
        """Convert OD [H,W,3] float32 → RGB [H,W,3] uint8."""
        rgb = self.io * np.exp(-od)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _get_stain_matrix(self, od: np.ndarray) -> np.ndarray:
        """
        Estimate 2×3 stain matrix via SVD of tissue-only OD pixels.
        Returns stain_matrix [2, 3] where each row is a stain vector.
        """
        # Flatten to [N, 3]
        od_flat = od.reshape(-1, 3)
        # Keep tissue pixels (OD norm > beta)
        mask = od_flat.sum(axis=1) > self.beta
        od_tissue = od_flat[mask]
        if od_tissue.shape[0] < 10:
            # Not enough tissue: return identity-like matrix
            return np.eye(2, 3, dtype=np.float32)

        # SVD: covariance → top-2 singular vectors
        _, _, Vt = np.linalg.svd(od_tissue, full_matrices=False)
        V = Vt[:2].T   # [3, 2]

        # Project tissue OD onto the 2D stain plane
        proj = od_tissue @ V  # [N, 2]

        # Compute angles in stain plane
        angles = np.arctan2(proj[:, 1], proj[:, 0])

        # Stain vectors at robust percentiles
        a1 = np.percentile(angles, 1)
        a2 = np.percentile(angles, 99)

        # Two stain directions in OD space
        s1 = (V @ np.array([np.cos(a1), np.sin(a1)])).reshape(1, 3)
        s2 = (V @ np.array([np.cos(a2), np.sin(a2)])).reshape(1, 3)

        stain_matrix = np.concatenate([s1, s2], axis=0)  # [2, 3]
        # Normalize each row
        norms = np.linalg.norm(stain_matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        stain_matrix = stain_matrix / norms
        return stain_matrix.astype(np.float32)

    def _augment(self, rgb: np.ndarray) -> np.ndarray:
        """Apply stain augmentation to a single RGB patch."""
        H, W = rgb.shape[:2]
        od = self._rgb2od(rgb)  # [H, W, 3]
        od_flat = od.reshape(-1, 3)  # [N, 3]

        stain_matrix = self._get_stain_matrix(od)  # [2, 3]

        # Solve for concentrations: od_flat ≈ C @ stain_matrix  [N, 2]
        # Use least-squares (NNLS-free approximation via pseudo-inverse)
        SM_pinv = np.linalg.pinv(stain_matrix)  # [3, 2]
        C = od_flat @ SM_pinv  # [N, 2]
        C = np.clip(C, 0, None)  # concentrations >= 0

        # Perturb: each stain channel scaled by random α in [1-σ, 1+σ]
        alpha = 1.0 + np.random.uniform(-self.sigma, self.sigma, size=(1, 2)).astype(np.float32)
        C_aug = C * alpha  # [N, 2]

        # Reconstruct OD
        od_aug = (C_aug @ stain_matrix).reshape(H, W, 3)  # [H, W, 3]
        od_aug = np.clip(od_aug, 0, 10)

        return self._od2rgb(od_aug)  # [H, W, 3] uint8

    # ── mmseg BaseTransform API ──────────────────────────────────────────────

    def transform(self, results: dict) -> dict:
        if np.random.random() >= self.prob:
            return results   # skip augmentation

        img = results.get("img")
        if img is None:
            return results

        # img in mmseg is typically numpy uint8 [H, W, 3]
        if img.dtype != np.uint8:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img_u8 = img

        try:
            img_aug = self._augment(img_u8)
        except Exception as e:
            # Graceful fallback: return original
            import warnings
            warnings.warn(f"[StainAugment] failed: {e}; returning original image")
            return results

        results["img"] = img_aug.astype(img.dtype)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(sigma={self.sigma}, prob={self.prob})"
