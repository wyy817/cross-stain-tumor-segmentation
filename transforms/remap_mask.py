"""
RemapMask transform — remaps specific values in gt_seg_map.

Used to convert binary masks saved as 0/255 to 0/1 class indices,
since mmseg's default ignore_index=255 would otherwise silently
discard all tumor pixels during training.
"""

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RemapMask(BaseTransform):
    """Remap pixel values in gt_seg_map.

    Args:
        mapping (dict): {old_value: new_value}, e.g. {255: 1}
    """

    def __init__(self, mapping: dict):
        self.mapping = {int(k): int(v) for k, v in mapping.items()}

    def transform(self, results: dict) -> dict:
        seg = results["gt_seg_map"]
        for old_val, new_val in self.mapping.items():
            seg[seg == old_val] = new_val
        results["gt_seg_map"] = seg
        return results
