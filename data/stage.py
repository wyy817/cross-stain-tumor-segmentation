import math
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
from zero import kfbparser
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from reader import WSIReader, OpenSlideReader, MaskReader, KfbReader
from utils import apply_pose, scale2d, translate, rotate, cv2interpolation


# ============================================================
# Camera
# ============================================================

@dataclass
class Camera:
    x: float = 0.0
    y: float = 0.0
    z: float = 1.0   # zoom factor in world->screen mapping, constraint: z >= 1
    w: int = 1024
    h: int = 1024

    def __post_init__(self) -> None:
        self.z = max(float(self.z), 1.0)

    @property
    def center(self) -> Tuple[float, float]:
        # pixel-space center of the viewport
        return (self.w * 0.5, self.h * 0.5)

    def set(self, w: int, h: int) -> None:
        self.w = int(w)
        self.h = int(h)

    def move(self, x: float, y: float, z: Optional[float] = None) -> None:
        self.x = float(x)
        self.y = float(y)
        if z is not None:
            self.z = max(float(z), 1.0)


# ============================================================
# Actor (NO z)
# pose includes: translate + rotate + scale(fx, fy) in world space
# ============================================================

class Actor:
    """
    Generic 2D actor:
      - readers: WSIReader or list[WSIReader]
      - pose: 3x3 local->world (includes translation, rotation, scale)
      - info: persistence dict used by reset()

    IMPORTANT:
      - Actor has NO z.
      - Camera.z controls global zoom.
      - Per-actor zoom is controlled by pose scaling (fx, fy).
    """

    def __init__(self, readers: Union[WSIReader, List[WSIReader]]) -> None:
        if isinstance(readers, list):
            if len(readers) == 0:
                raise ValueError("readers list cannot be empty")
            self.readers: List[WSIReader] = readers
        else:
            self.readers = [readers]

        self._pose = np.eye(3, dtype=np.float64)

        # persistence cache (for reset)
        self.info: Optional[dict] = None

    # ---------------- pose ----------------

    def pose(self) -> np.ndarray:
        return self._pose.copy()

    def set(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (3, 3):
            raise ValueError("pose must be 3x3")
        self._pose = pose

    # ---------------- geometry ----------------

    def center(self, type: str = "world") -> np.ndarray:
        """
        type:
          - 'local': local center (before pose)
          - 'world': apply pose to local center (default)

        NOTE:
          - 'world' MUST call 'local' first (as you requested).
        """
        t = str(type).lower().strip()
        if t not in ("local", "world"):
            raise ValueError("center(type=...) must be 'local' or 'world'")

        # local center
        ws, hs = [], []
        for r in self.readers:
            ws.append(float(r.width(1.0)))
            hs.append(float(r.height(1.0)))
        w = float(np.max(ws))
        h = float(np.max(hs))
        c_local = np.array([w * 0.5, h * 0.5], dtype=np.float64)

        if t == "local":
            return c_local

        # world center: derived from local center
        return apply_pose(self._pose, c_local)

    def bbox(self) -> Tuple[float, float, float, float]:
        w = float(self.readers[0].width(1.0))
        h = float(self.readers[0].height(1.0))
        corners = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float64)
        wc = apply_pose(self._pose, corners)
        minx, miny = wc.min(axis=0)
        maxx, maxy = wc.max(axis=0)
        return float(minx), float(miny), float(maxx), float(maxy)

    def _pose_scale_xy(self) -> Tuple[float, float]:
        A = self._pose[:2, :2].astype(np.float64)
        sx = float(np.linalg.norm(A[:, 0]))
        sy = float(np.linalg.norm(A[:, 1]))
        sx = max(sx, 1e-12)
        sy = max(sy, 1e-12)
        return sx, sy

    # ---------------- transforms ----------------

    def move(self, x: float, y: float) -> None:
        c = self.center("world")
        self._pose = translate(float(x) - float(c[0]), float(y) - float(c[1])) @ self._pose

    def rotate(self, angle_deg: float, axis: Optional[Tuple[float, float]] = None) -> None:
        """
        Rotate in world space around axis.
        Default axis: center(type='world'), and 'world' is derived via local center.
        """
        theta = math.radians(float(angle_deg))
        if axis is None:
            p = self.center("world")
            ax, ay = float(p[0]), float(p[1])
        else:
            ax, ay = float(axis[0]), float(axis[1])
        self._pose = translate(ax, ay) @ rotate(theta) @ translate(-ax, -ay) @ self._pose

    def flip(self, flip_type: str) -> None:
        t = flip_type.lower().strip()
        if t in ("lr", "h", "horizontal"):
            sx, sy = -1.0, 1.0
        elif t in ("ud", "v", "vertical"):
            sx, sy = 1.0, -1.0
        elif t in ("both", "hv", "vh"):
            sx, sy = -1.0, -1.0
        else:
            raise ValueError(f"Unknown flip_type: {flip_type}")

        c = self.center("world")
        cx, cy = float(c[0]), float(c[1])
        self._pose = translate(cx, cy) @ scale2d(sx, sy) @ translate(-cx, -cy) @ self._pose

    def scale(self, fx: float = 1.0, fy: float = 1.0, axis: Optional[Tuple[float, float]] = None) -> None:
        """
        Apply additional scaling in world space around axis.
        Default axis: center(type='world').
        """
        fx = float(fx)
        fy = float(fy)

        if axis is None:
            c = self.center("world")
            ax, ay = float(c[0]), float(c[1])
        else:
            ax, ay = float(axis[0]), float(axis[1])

        self._pose = translate(ax, ay) @ scale2d(fx, fy) @ translate(-ax, -ay) @ self._pose

    # ---------------- persistence ----------------

    def dump(self) -> Dict[str, Any]:
        data = {
            "version": 1,
            "pose": self._pose.reshape(-1).tolist(),
        }
        return data

    def load(self, info: Dict[str, Any]) -> bool:
        try:
            pose_flat = info.get("pose", None)
            pose = np.array(pose_flat, dtype=np.float64).reshape(3, 3)
            self._pose = pose
            self.info = dict(info)
            return True
        except Exception:
            return False

    def reset(self) -> None:
        if self.info is not None:
            if self.load(self.info):
                return
        self._pose = np.eye(3, dtype=np.float64)

    # ---------------- rendering core ----------------

    def read(self, camera: Camera) -> List[np.ndarray]:
        """
        KEY DESIGN:
          - Screen->World depends ONLY on camera (x,y,z).
          - Actor scaling is fully inside pose.
          - No actor-dependent drift when panning camera.

        Sampling resolution (ratio):
          - effective_zoom ~ camera.z / max(scale_x, scale_y)
          - ratio = 1 / effective_zoom = max(scale) / camera.z, clamped to (0,1]
        """
        cam = camera
        W, H = int(cam.w), int(cam.h)
        cx_px, cy_px = cam.center

        cam_z = max(float(cam.z), 1e-6)

        sx, sy = self._pose_scale_xy()
        smax = max(abs(sx), abs(sy), 1e-12)

        ratio = smax / cam_z
        ratio = float(np.clip(ratio, 1e-6, 1.0))

        pose_inv = np.linalg.inv(self.pose())

        uu, vv = np.meshgrid(
            np.arange(W, dtype=np.float64),
            np.arange(H, dtype=np.float64),
        )

        # Screen -> World
        world_x = float(cam.x) + (uu - float(cx_px)) * cam_z
        world_y = float(cam.y) + (vv - float(cy_px)) * cam_z
        world_xy = np.stack([world_x, world_y], axis=-1)  # (H,W,2)

        # World -> Local(base)
        local_xy = apply_pose(pose_inv, world_xy.reshape(-1, 2)).reshape(H, W, 2)

        # Local(base) -> Local(ratio space)
        local_lvl = local_xy * ratio
        x = local_lvl[..., 0]
        y = local_lvl[..., 1]

        outs: List[np.ndarray] = []

        for rdr in self.readers:
            lvl_w = int(rdr.width(ratio))
            lvl_h = int(rdr.height(ratio))

            valid = (x >= 0) & (y >= 0) & (x < (lvl_w - 1)) & (y < (lvl_h - 1))

            ch = int(rdr.channels())
            try:
                out_dtype = np.dtype(rdr.dtype())
            except Exception:
                out_dtype = np.uint8

            if not np.any(valid):
                fill_val = rdr.fill()
                img = np.empty((H, W, ch), dtype=out_dtype)
                img[...] = fill_val
                outs.append(img)
                continue

            xv = x[valid]
            yv = y[valid]

            x0 = int(np.floor(xv.min()))
            y0 = int(np.floor(yv.min()))
            x1 = int(np.ceil(xv.max())) + 2
            y1 = int(np.ceil(yv.max())) + 2

            x0c = max(0, x0)
            y0c = max(0, y0)
            x1c = min(lvl_w, x1)
            y1c = min(lvl_h, y1)

            bw = max(1, x1c - x0c)
            bh = max(1, y1c - y0c)

            patch = np.asarray(rdr.read(int(x0c), int(y0c), int(bw), int(bh), ratio=ratio))
            if patch.ndim == 2:
                patch = patch[..., np.newaxis]
            if patch.shape[-1] > ch:
                patch = patch[..., :ch]

            map_x = (x - float(x0c)).astype(np.float32)
            map_y = (y - float(y0c)).astype(np.float32)

            interp = cv2interpolation(rdr.inter())
            border_val = rdr.fill()

            warped = cv2.remap(
                np.ascontiguousarray(patch),
                map_x,
                map_y,
                interpolation=interp,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_val,
            )

            if warped.ndim == 2:
                warped = warped[..., np.newaxis]

            outs.append(warped)

        return outs

    def render(self, camera):
        raise NotImplementedError


# ============================================================
# ColorActor (RGBA)
# ============================================================

class ColorActor(Actor):
    def render(self, camera):
        buffer = self.read(camera)
        color = buffer[0]
        if color.ndim == 2:
            color = color[..., np.newaxis]

        if len(buffer) == 1:
            mask = np.expand_dims(np.zeros(color.shape[:2], dtype="uint8"), axis=-1) + 255
        else:
            mask = buffer[1]
            if mask.ndim == 2:
                mask = mask[..., np.newaxis]

        if color.shape[-1] == 4:
            mask = mask & color[..., [3]]

        return np.concatenate((color[..., :3], mask), axis=-1)

class SegmenticActor(Actor):
    def render(self, camera):
        buffer, mask = self.read(camera)
        buffer[mask == 0] = 0
        return buffer


# ============================================================
# Team
# ============================================================

class Team:
    def __init__(self):
        self._actors: List[Actor] = []

    def __len__(self):
        return len(self._actors)

    def __getitem__(self, i):
        return self._actors[i]

    def load(self, path: str) -> None:
        datas = json.load(open(path, "r"))
        if isinstance(datas, dict):
            for actor in self._actors:
                actor.load(datas)
        elif isinstance(datas, list):
            assert len(datas) == len(self._actors)
            for i in range(len(self._actors)):
                self._actors[i].load(datas[i])
        else:
            raise ValueError("vp must be a dict or a list")

    def save(self, path: str) -> None:
        datas = []
        for actor in self._actors:
            datas.append(actor.dump())
        json.dump(datas, open(path, "w"))

    def center(self) -> np.ndarray:
        centers = []
        for actor in self._actors:
            centers.append(actor.center("world"))
        return np.mean(centers, axis=0)

    def render(self, camera):
        raise NotImplementedError


# ============================================================
# ColorTeam
# ============================================================

class ColorTeam(Team):
    def __init__(self, color_path: str, mask_path: str | None = None):
        super().__init__()
        color = OpenSlideReader(color_path)

        if mask_path is None:
            self._actors.append(ColorActor([color]))
        else:
            reader = KfbReader(mask_path, "nearest", channels=1, dtype=np.uint8, fill=0)
            mask_preview = reader.read(
                0,
                0,
                int(reader.width(0.05)),
                int(reader.height(0.05)),
                ratio=0.05,
            )
            if mask_preview.ndim == 3:
                mask_preview = mask_preview[..., 0]

            for i in range(1, int(mask_preview.max()) + 1):
                mask = MaskReader(mask_path, i)
                self._actors.append(ColorActor([color, mask]))

    def render(self, camera):
        images, alphas = [], []
        for actor in self._actors:
            image = actor.render(camera)  # RGBA
            alphas.append(image[..., [3]].astype("float32"))
            images.append(image[..., :3].astype("float32"))

        images = np.stack(images, axis=0)
        alphas = np.stack(alphas, axis=0)

        summary = alphas.sum(axis=0, keepdims=True)
        bg = np.squeeze(summary <= 0, axis=0).astype("float32")
        summaries = np.repeat(summary, len(images), axis=0)

        fg = summaries > 0
        weights = np.zeros_like(alphas)
        weights[fg] = alphas[fg] / summaries[fg]

        rgb = (weights * images).sum(axis=0) + bg * 255.0
        return rgb.clip(0, 255).astype(np.uint8)

class PropsTeam(Team):
    def __init__(self, prop_path: str, mask_path: str | None = None):
        super().__init__()
        prop = KfbReader(prop_path, "nearest", channels=1, dtype=np.uint8, fill=0)
        mask_path = mask_path if mask_path else None
        reader = KfbReader(mask_path, "nearest", channels=1, dtype=np.uint8, fill=0)
        mask_preview = reader.read(
            0,
            0,
            int(reader.width(0.05)),
            int(reader.height(0.05)),
            ratio=0.05,
        )
        if mask_preview.ndim == 3:
            mask_preview = mask_preview[..., 0]
        for i in range(1, int(mask_preview.max()) + 1):
            #if i >= 5:
            #    continue
            mask = MaskReader(mask_path, i)
            self._actors.append(SegmenticActor([prop, mask]))

    def render(self, camera):
        result = None
        for actor in self._actors:
            buffers = actor.read(camera)
            if result is None:
                result = np.zeros_like(buffers[0])
            if len(buffers) > 1:
                mask = buffers[1] > 0
            else:
                mask = np.ones_like(buffers[0], dtype='uint8')
            result[mask] =  buffers[0][mask]
        return result



def test():
    team = ColorTeam(
        "/path/to/dataset/PATIENT_X/PATIENT_X-HE.mrxs",
        "/path/to/dataset/PATIENT_X/PATIENT_X-FHE.kfb",
    )
    team.center()
    team.load("/path/to/dataset/PATIENT_X/PATIENT_X-HE.vp")

    camera = Camera(0, 0, 40, 4096, 4096)
    camera.move(29984 + 4096 * 4, 79776 + 4096*4, z=1)
    patch = team.render(camera)
    cv2.imwrite('E:/1.png', patch)

if __name__ == "__main__":
    # test()
    input_root = '/path/to/dataset'
    output_root = '/path/to/dataset3'
    for folder in os.listdir(input_root):
        input_dir = f'{input_root}/{folder}'
        output_dir = f'{output_root}/{folder}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/{folder}/-SIHC.kfb'
        if os.path.exists(output_path):
            continue
        writer = kfbparser.Writer(output_path)
        for filename in os.listdir(input_dir):
            stem, ext = os.path.splitext(filename)
            if ext == '.png':
                try:
                    x, y = [int(_) for _ in stem.split("-")]
                except ValueError:
                    continue
                image = Image.open(f'{input_dir}/{filename}')
                w, h = image.size
                camera = Camera(0, 0, 1, w, h)
                # camera = Camera(0, 0, 1, 4096, 4096)
                camera.move(x + w / 2, y + h / 2)

                team = PropsTeam(
                    f"{input_root}/{folder}/{folder}-SHE.kfb",
                    f"{input_root}/{folder}/{folder}-FHE.kfb"
                )
                team.load(f"{input_root}/{folder}/{folder}-HE.vp")

                # camera = Camera(0, 0, 40, 4096, 4096)
                # camera.move(29984 + 4096 * 4, 79776 + 4096 * 4, z=1)
                patch = team.render(camera)
                writer.roi(x, y, patch)
        writer.close()




