"""Download CholecSeg8k from Kaggle into data/raw/cholecseg8k.

Uses kagglehub, which handles caching and resuming. For public datasets no
API token is usually needed; if you get an auth error, create a token at
kaggle.com -> Settings -> API -> Create New Token and place kaggle.json in
C:\\Users\\<you>\\.kaggle\\
"""

import shutil
from pathlib import Path

import kagglehub

from sam3_seg.utils.config import load_config
from sam3_seg.utils.env import load_env

# inside main(), first line:
load_env(required=["KAGGLE_USERNAME", "KAGGLE_KEY"])

DATASET_ID = "newslab/cholecseg8k"


def main() -> None:
    cfg = load_config()
    target: Path = cfg["paths"]["raw_data"]          # data/raw/cholecseg8k

    if target.exists() and any(p.name.startswith("video") for p in target.iterdir()):
        print(f"Dataset already present at {target} — nothing to do.")
        return

    print(f"Downloading {DATASET_ID} (~3-4 GB, be patient)...")
    cache_path = Path(kagglehub.dataset_download(DATASET_ID))
    print(f"Downloaded to kagglehub cache: {cache_path}")

    target.mkdir(parents=True, exist_ok=True)
    # The archive may unpack with a wrapper folder; normalize either way.
    video_dirs = sorted(cache_path.rglob("video*"))
    top_level = {p for p in video_dirs if p.is_dir() and p.name.count("_") == 0}
    print(f"Copying {len(top_level)} video folders into {target} ...")
    for vd in sorted(top_level):
        dest = target / vd.name
        if not dest.exists():
            shutil.copytree(vd, dest)
            print("  +", vd.name)

    print("Done.")


if __name__ == "__main__":
    main()