"""
make_v7_csv.py — Phase 9 Plan A: Stain-level hold-out experiment
================================================================
Train: HE + IHC (all 5 patients, from v6 train split)
Val:   HE + IHC + NCR (all 5 patients, from v6 val split)

During training, only HE+IHC are seen. NCR val is the cross-stain test.
HE+IHC val serves as in-distribution reference.
"""
import pandas as pd

v6_path = "/path/to/FYP_S2/data/patch_index_private_v6.csv"
v7_path = "/path/to/FYP_S2/data/patch_index_private_v7.csv"

df = pd.read_csv(v6_path)

train = df[(df["split"] == "train") & (df["stain"].isin(["HE", "IHC"]))]
val   = df[df["split"] == "val"]   # keep all 3 stains for eval

v7 = pd.concat([train, val], ignore_index=True)
v7.to_csv(v7_path, index=False)

print(f"v7 saved to {v7_path}")
print(v7.groupby(["split", "stain"]).size().to_string())
print(f"\nTotal: {len(v7)} rows")
print(f"  Train: {len(train)} (HE+IHC only)")
print(f"  Val:   {len(val)} (HE+IHC+NCR, NCR is cross-stain test)")
