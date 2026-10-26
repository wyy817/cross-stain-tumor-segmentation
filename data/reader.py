# reader.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Tuple, Type, Any
from zero import kfbparser
import math
import numpy as np
import cv2
import openslide
from utils import cv2interpolation



class WSIReader:
    def width(self, ratio: float = 1.0) -> int:
        ...

    def height(self, ratio: float = 1.0) -> int:
        ...

    def channels(self) -> int:
        ...

    def inter(self) -> str:
        """Interpolation method string."""
        ...

    def dtype(self) -> Any:
        ...

    def fill(self) -> Any:
        """Fill pixel value like 0 or 255 (scalar or sequence)."""
        ...

    def read(self, x: int, y: int, w: int, h: int, ratio: float = 1.0) -> np.ndarray:
        """
        Return numpy array for region at given ratio.
        Shape: (H,W) or (H,W,C)
        """
        ...


class OpenSlideReader(WSIReader):
    """
    OpenSlide-backed WSIReader.

    Coordinate convention:
      - width(ratio), height(ratio): dimensions under the requested ratio
      - read(x,y,w,h,ratio): x,y,w,h are in the coordinate system of that ratio
        (i.e., same as indexing an image of size width(ratio) x height(ratio))
    """

    def __init__(self, path: str):
        if openslide is None:
            raise ImportError("openslide-python is not available.")
        self._slide = openslide.OpenSlide(path)

        # OpenSlide read_region returns RGBA uint8
        self._channels = 4
        self._dtype = np.dtype(np.uint8)

        # Default interpolation and fill
        self._inter = "linear"
        self._fill = 0

        # Cache
        self._level_downsamples = [float(x) for x in self._slide.level_downsamples]
        self._level_dimensions = [(int(w), int(h)) for (w, h) in self._slide.level_dimensions]
        self._w0, self._h0 = self._level_dimensions[0]

    # -------------------------
    # Basic properties
    # -------------------------
    def channels(self) -> int:
        return self._channels

    def dtype(self):
        # Return something np.dtype can parse
        return self._dtype

    def inter(self) -> str:
        return self._inter

    def fill(self):
        return self._fill

    # -------------------------
    # Ratio <-> OpenSlide level mapping
    # -------------------------
    def _ratio_to_level(self, ratio: float) -> int:
        """
        ratio=1.0 -> level 0
        ratio=0.5 -> level 1 (downsample ~ 2)
        general: choose nearest downsample to (1/ratio) in log space
        """
        r = float(ratio)
        if not math.isfinite(r) or r <= 0:
            r = 1.0

        target_down = 1.0 / r
        downs = self._level_downsamples

        # Use log distance (more stable across pyramid)
        log_t = math.log(max(target_down, 1e-12))
        best_i = 0
        best_d = float("inf")
        for i, d in enumerate(downs):
            dd = abs(math.log(max(float(d), 1e-12)) - log_t)
            if dd < best_d:
                best_d = dd
                best_i = i
        return int(best_i)

    def _level_ratio(self, level: int) -> float:
        # ratio = 1 / downsample
        down = float(self._level_downsamples[int(level)])
        return 1.0 / max(down, 1e-12)

    # -------------------------
    # Dimensions
    # -------------------------
    def width(self, ratio: float = 1.0) -> int:
        r = float(ratio)
        if not math.isfinite(r) or r <= 0:
            r = 1.0

        lvl = self._ratio_to_level(r)
        lvl_ratio = self._level_ratio(lvl)

        # If ratio matches a level closely, return that level's native dims
        if abs(lvl_ratio - r) <= 1e-6:
            return int(self._level_dimensions[lvl][0])

        # Otherwise approximate by base dimensions * ratio
        return int(max(1, round(self._w0 * r)))

    def height(self, ratio: float = 1.0) -> int:
        r = float(ratio)
        if not math.isfinite(r) or r <= 0:
            r = 1.0

        lvl = self._ratio_to_level(r)
        lvl_ratio = self._level_ratio(lvl)

        if abs(lvl_ratio - r) <= 1e-6:
            return int(self._level_dimensions[lvl][1])

        return int(max(1, round(self._h0 * r)))

    # -------------------------
    # Read region
    # -------------------------
    def read(self, x: int, y: int, w: int, h: int, ratio: float = 1.0) -> np.ndarray:
        """
        Return region at requested ratio with shape:
          - (h, w, 4) RGBA uint8
        """
        r = float(ratio)
        if not math.isfinite(r) or r <= 0:
            r = 1.0

        w = int(w)
        h = int(h)
        if w <= 0 or h <= 0:
            return np.zeros((max(h, 0), max(w, 0), 4), dtype=np.uint8)

        # Choose nearest level
        level = self._ratio_to_level(r)
        down = float(self._level_downsamples[level])
        lvl_ratio = self._level_ratio(level)

        # Map requested ratio-space coords -> chosen level-space coords
        # x0 in level0 = x / r
        # xL in level  = x0 / down = x * (lvl_ratio / r)
        scale = lvl_ratio / r

        xL = float(x) * scale
        yL = float(y) * scale
        wL = float(w) * scale
        hL = float(h) * scale

        # Convert to integer region in chosen level
        xL0 = int(math.floor(xL))
        yL0 = int(math.floor(yL))
        xL1 = int(math.ceil(xL + wL))
        yL1 = int(math.ceil(yL + hL))

        bw = max(1, xL1 - xL0)
        bh = max(1, yL1 - yL0)

        # Convert chosen level origin -> level0 origin for OpenSlide
        # location in level0 pixels
        x0 = int(math.floor(xL0 * down))
        y0 = int(math.floor(yL0 * down))

        # Read patch at chosen level
        pil_img = self._slide.read_region((x0, y0), int(level), (int(bw), int(bh)))
        patch = np.asarray(pil_img, dtype=np.uint8)  # RGBA uint8, shape (bh,bw,4)

        # If chosen level ratio != requested ratio, resize to exactly (h,w)
        if patch.shape[0] != h or patch.shape[1] != w:
            interp = cv2interpolation(self._inter)
            patch = cv2.resize(patch, (w, h), interpolation=interp)

        return patch


    def close(self) -> None:
        # OpenSlide has close() on newer versions
        try:
            self._slide.close()
        except Exception:
            pass


class KfbReader(WSIReader):
    """
    WSIReader wrapper for kfbparser.Reader.

    Conventions:
      - ratio=1.0 => base resolution (max scale)
      - ratio=0.5 => half resolution
      - read(x,y,w,h,ratio): x,y,w,h are in the coordinate system of that ratio
        (same as indexing an image of size width(ratio) x height(ratio))

    Mapping:
      kfb scale = base_scale * ratio
      and kfb coordinates for that scale are:
        x_kfb = x / ratio, y_kfb = y / ratio, w_kfb = w / ratio, h_kfb = h / ratio
      because kfbparser.Reader.get() uses coordinates in its own scale space,
      while our WSIReader uses ratio-space coordinates.
    """

    def __init__(
        self,
        path: str,
        inter: str = "cubic",
        method: str | int = "disk",
        channels: int = 3,
        dtype: str|np.dtype = np.uint8,
        fill: Optional[int | Tuple[int, int, int]] = 0,
    ) -> None:
        self._r = kfbparser.Reader(path, inter=inter, method=method, fill=fill)

        # base scale: scales sorted descending, scales[0] is max (e.g. 40.0)
        scales = self._r.scales
        if not scales:
            raise ValueError("kfbparser.Reader has no scales")
        self._scale = float(scales[0])

        # info.width / info.height are at base scale (40x)
        info = self._r.info
        self._w0 = int(info.width)
        self._h0 = int(info.height)

        # protocol defaults
        self._channels = channels
        self._dtype = dtype
        self._inter = inter
        self._fill = fill
        if not isinstance(fill, int):
            assert  len(fill) == self._channels

    # -------------------------
    # WSIReader protocol
    # -------------------------
    def width(self, ratio: float = 1.0) -> int:
        r = float(ratio)
        if not np.isfinite(r) or r <= 0:
            r = 1.0
        return int(max(1, round(self._w0 * r)))

    def height(self, ratio: float = 1.0) -> int:
        r = float(ratio)
        if not np.isfinite(r) or r <= 0:
            r = 1.0
        return int(max(1, round(self._h0 * r)))

    def channels(self) -> int:
        return self._channels

    def inter(self) -> str:
        return str(self._inter)

    def dtype(self) -> Any:
        return self._dtype

    def fill(self) -> Any:
        return self._fill

    def read(self, x: int, y: int, w: int, h: int, ratio: float = 1.0) -> np.ndarray:
        """
        Return RGB uint8 image (H,W,3).
        """
        if not np.isfinite(ratio) or ratio <= 0:
            ratio = 1.0

        w = int(w)
        h = int(h)
        if w <= 0 or h <= 0:
            return np.zeros((max(h, 0), max(w, 0), self._channels), dtype=self._dtype)
        scale = self._scale * ratio
        img = self._r.get(x, y, w, h, scale)
        if img is None:
            # fill if missing
            out = np.empty((h, w, self._channels), dtype=self._dtype)
            out[...] = self._fill
            return out
        # Normalise channel count: some kfb levels store grayscale tiles
        if self._channels == 3:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
        elif self._channels == 1:
            if img.ndim == 3:
                img = img[:, :, :1]
        return img

    # optional
    @property
    def info(self) -> Any:
        return self._r.info

    @property
    def scales(self) -> list[float]:
        return list(self._r.scales)


class InstanceReader(KfbReader):
    def __init__(self, path: str, id: int):
        super().__init__(path, "nearest", channels=4, dtype=np.uint8, fill=0)

    def read(self, x: int, y: int, w: int, h: int, ratio: float = 1.0) -> np.ndarray:
        image = self.read(x, y, w, h, ratio)



class MaskReader(KfbReader):
    def __init__(self, path: str, id: int):
        super().__init__(path, "nearest", channels=1, dtype=np.uint8, fill=0)
        self._id = id

    def read(self, x: int, y: int, w: int, h: int, ratio: float = 1.0) -> np.ndarray:
        image = super().read(x, y, w, h, ratio)
        if self._channels == 1:
            image = np.expand_dims(image, axis=-1)
        return (image == self._id).astype('uint8') * 255

