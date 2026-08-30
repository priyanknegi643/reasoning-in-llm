import argparse
import os

from huggingface_hub import HfApi, hf_hub_download


def _download_file(repo_id: str, filename: str, local_dir: str) -> None:
    if os.path.exists(os.path.join(local_dir, filename)):
        return
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir,
    )


def download_fineweb10b(num_chunks: int, output_root: str) -> None:
    """
    Download GPT-2-tokenized FineWeb10B bins.

    This is the original behavior used by `data/fineweb.py`.
    """
    local_dir = os.path.join(output_root, "fineweb10B")
    os.makedirs(local_dir, exist_ok=True)

    _download_file("kjj0/fineweb10B-gpt2", "fineweb_val_%06d.bin" % 0, local_dir)
    for i in range(1, num_chunks + 1):
        _download_file("kjj0/fineweb10B-gpt2", "fineweb_train_%06d.bin" % i, local_dir)


def download_fineweb_edu_100bt(output_root: str, max_files: int | None = None) -> None:
    """
    Download the FineWeb-Edu sample-100BT subset from Hugging Face.

    This fetches parquet shards from:
      - HuggingFaceFW/fineweb-edu (sample-100BT)
    """
    repo_id = "HuggingFaceFW/fineweb-edu"
    local_dir = os.path.join(output_root, "fineweb-edu-sample-100BT")
    os.makedirs(local_dir, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    # HF repos can store this subset under either path convention.
    candidates = [
        f for f in files if f.startswith("sample/100BT/") and f.endswith(".parquet")
    ]
    if not candidates:
        candidates = [
            f
            for f in files
            if f.startswith("data/sample-100BT/") and f.endswith(".parquet")
        ]

    if not candidates:
        raise RuntimeError(
            "Could not locate FineWeb-Edu sample-100BT parquet files in repo "
            f"{repo_id}. Repo layout may have changed."
        )

    candidates = sorted(candidates)
    if max_files is not None:
        candidates = candidates[:max_files]

    for filename in candidates:
        _download_file(repo_id, filename, local_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FineWeb/FineWeb-Edu data.")
    parser.add_argument(
        "--dataset",
        default="fineweb10B",
        choices=["fineweb10B", "fineweb-edu-100BT"],
        help="Which dataset to download.",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=103,
        help="Number of fineweb10B train chunks (ignored for fineweb-edu-100BT).",
    )
    parser.add_argument(
        "--output_root",
        default=os.path.dirname(__file__),
        help="Base directory where downloaded data will be stored.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional cap on number of files to download (mainly for smoke tests).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "fineweb10B":
        download_fineweb10b(num_chunks=args.num_chunks, output_root=args.output_root)
    else:
        download_fineweb_edu_100bt(
            output_root=args.output_root,
            max_files=args.max_files,
        )
