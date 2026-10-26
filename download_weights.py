"""
Download CONCH weights from HuggingFace.

Steps before running:
  1. Accept terms at: https://huggingface.co/MahmoodLab/CONCH
  2. Generate a token at: https://huggingface.co/settings/tokens
  3. Run: python download_weights.py --token hf_xxxx

The weight file (pytorch_model.bin, ~400MB) will be saved to:
  weights/pytorch_model.bin
"""
import argparse
import os
from pathlib import Path


def download_conch_weights(hf_token: str, save_dir: str = "weights"):
    from huggingface_hub import hf_hub_download

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    repo_id = "MahmoodLab/CONCH"
    filenames = ["pytorch_model.bin", "config.json"]

    for fname in filenames:
        out_path = save_dir / fname
        if out_path.exists():
            print(f"[skip] {fname} already exists at {out_path}")
            continue
        print(f"[download] {fname} from {repo_id} ...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            token=hf_token,
            local_dir=str(save_dir),
        )
        print(f"[done] saved to {local_path}")

    weight_path = save_dir / "pytorch_model.bin"
    print(f"\n[OK] CONCH weights ready at: {weight_path.resolve()}")
    print(f"     Use this path in configs: checkpoint_path='{weight_path.resolve()}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True,
                        help="HuggingFace access token (from settings/tokens)")
    parser.add_argument("--save_dir", type=str, default="weights",
                        help="Directory to save weights (default: weights/)")
    args = parser.parse_args()
    download_conch_weights(args.token, args.save_dir)
