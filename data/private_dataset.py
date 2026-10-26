"""
PrivateDataset — mmseg dataset for the supervisor-provided private WSI patch data.

CSV format (patch_index_private.csv / patch_index_private_cs.csv):
    img_path, seg_path, patient_id, stain, split

Usage in config:
    dataset=dict(
        type="PrivateDataset",
        csv_file="/path/to/patch_index_private.csv",
        split="train",          # or "val"
        stains=["HE"],          # filter by stain type
        pipeline=train_pipeline,
    )
"""

import os
import pandas as pd

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class PrivateDataset(BaseSegDataset):
    METAINFO = dict(
        classes=("background", "tumor"),
        palette=[[0, 0, 0], [255, 0, 0]],
    )

    def __init__(self, csv_file, split, stains=None, **kwargs):
        self._csv_file = csv_file
        self._split = split
        self._stains = stains or ["HE"]
        # data_root='' so that absolute paths in CSV are used as-is
        kwargs.setdefault("data_root", "")
        super().__init__(img_suffix=".png", seg_map_suffix=".png", **kwargs)

    def load_data_list(self):
        df = pd.read_csv(self._csv_file)
        df = df[df["split"] == self._split]
        if self._stains:
            df = df[df["stain"].isin(self._stains)]
        data_list = []
        for _, row in df.iterrows():
            data_list.append(
                dict(
                    img_path=row["img_path"],
                    seg_map_path=row["seg_path"],
                    reduce_zero_label=False,
                    seg_fields=[],
                )
            )
        return data_list
