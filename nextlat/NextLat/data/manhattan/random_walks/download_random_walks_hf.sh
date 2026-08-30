#!/bin/bash
set -euo pipefail

TARGET_DIR="data/manhattan/random_walks"
REPO_ID="JaydenTeoh/manhattan"

mkdir -p "${TARGET_DIR}"

python - <<'PY'
import shutil
from huggingface_hub import hf_hub_download

repo_id = "JaydenTeoh/manhattan"
repo_type = "dataset"
target_dir = "data/manhattan/random_walks"
filenames = [
    "all_pairs.pkl",
    "valid_turns.pkl",
    "node_and_direction_to_neighbor.pkl",
    "tokenizer.pkl",
    "eval_pairs_dist50.pkl",
    "shortest_paths.pkl",
]

for filename in filenames:
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
    )
    dest_path = f"{target_dir}/{filename}"
    shutil.copy2(local_path, dest_path)
    print(f"Downloaded {filename} -> {dest_path}")
PY
