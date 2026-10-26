from typing import Any

import math
import numpy as np

import cv2

# =========================
# Math helpers (2D pose)
# =========================
def translate(tx: float, ty: float) -> np.ndarray:
    m = np.eye(3, dtype=np.float64)
    m[0, 2] = float(tx)
    m[1, 2] = float(ty)
    return m


def rotate(theta_rad: float) -> np.ndarray:
    c = float(math.cos(theta_rad))
    s = float(math.sin(theta_rad))
    m = np.eye(3, dtype=np.float64)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def scale2d(sx: float, sy: float) -> np.ndarray:
    m = np.eye(3, dtype=np.float64)
    m[0, 0] = float(sx)
    m[1, 1] = float(sy)
    return m


def apply_pose(pose_3x3: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """
    pose: (3,3)
    xy: (...,2)
    return: (...,2)
    """
    xy = np.asarray(xy, dtype=np.float64)
    shp = xy.shape
    if shp[-1] != 2:
        raise ValueError("xy last dim must be 2")

    flat = xy.reshape(-1, 2)
    ones = np.ones((flat.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([flat, ones], axis=1)  # (N,3)
    out = (pose_3x3 @ homo.T).T  # (N,3)
    return out[:, :2].reshape(shp)


def cv2interpolation(inter: str) -> int:
    t = (inter or "").strip().lower()
    if t in ("nearest", "nn", "nearest_neighbor", "nearest-neighbor"):
        return cv2.INTER_NEAREST
    if t in ("linear", "bilinear"):
        return cv2.INTER_LINEAR
    if t in ("cubic", "bicubic"):
        return cv2.INTER_CUBIC
    if t in ("area",):
        return cv2.INTER_AREA
    if t in ("lanczos", "lanczos4"):
        return cv2.INTER_LANCZOS4
    # default
    return cv2.INTER_LINEAR

