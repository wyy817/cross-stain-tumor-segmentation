"""
mmseg Registry Registrations
==============================
Import this module before building any mmseg Runner/model to ensure
custom modules (ConchBackbone, ACROBATDataset, StainAugment) are registered.
"""

# Register all built-in mmseg modules (visualizers, heads, etc.) first
from mmseg.utils import register_all_modules as _mmseg_register
_mmseg_register(init_default_scope=False)

# Custom backbone
from models.conch_backbone import ConchBackbone  # noqa: F401

# Custom dataset (optional — only available when acrobat_dataset.py is present)
try:
    from data.acrobat_dataset import ACROBATDataset  # noqa: F401
except ImportError:
    pass

# Private dataset (supervisor-provided WSI data)
try:
    from data.private_dataset import PrivateDataset  # noqa: F401
except ImportError:
    pass

# Custom transforms
from transforms.stain_augment import StainAugment  # noqa: F401
from transforms.remap_mask import RemapMask         # noqa: F401

__all__ = ["ConchBackbone", "ACROBATDataset", "StainAugment"]
