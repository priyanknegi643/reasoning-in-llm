import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast

MAGIC = 20240520
VERSION = 1
HEADER_SIZE_INT32 = 256
_WORKER_TOKENIZER = None
_WORKER_EOS_ID = None


def _extract_text(row: dict) -> Optional[str]:
    for key in ("text", "content", "raw_content"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _write_bin(path: Path, token_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.zeros(HEADER_SIZE_INT32, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = int(token_ids.size)
    with path.open("wb") as f:
        f.write(header.tobytes(order="C"))
        f.write(token_ids.astype(np.uint16, copy=False).tobytes(order="C"))


def _init_worker_tokenizer(tokenizer_name: str) -> None:
    global _WORKER_TOKENIZER, _WORKER_EOS_ID
    _WORKER_TOKENIZER = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    _WORKER_EOS_ID = _WORKER_TOKENIZER.eos_token_id
    if _WORKER_EOS_ID is None:
        raise RuntimeError("GPT2 tokenizer eos_token_id is None")


def _tokenize_text_batch(text_batch: list[str]) -> list[np.ndarray]:
    if _WORKER_TOKENIZER is None or _WORKER_EOS_ID is None:
        raise RuntimeError("Tokenizer worker is not initialized")
    encoded = _WORKER_TOKENIZER(
        text_batch,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    token_arrays: list[np.ndarray] = []
    for ids in encoded["input_ids"]:
        if not ids:
            ids = [_WORKER_EOS_ID]
        else:
            ids.append(_WORKER_EOS_ID)
        token_arrays.append(np.asarray(ids, dtype=np.uint16))
    return token_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pretokenize FineWeb-Edu into NextLat .bin shards "
            "(fineweb_train_*.bin and fineweb_val_000000.bin)."
        )
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset id.",
    )
    parser.add_argument(
        "--subset",
        default="sample-100BT",
        help="Dataset config/subset name (e.g. sample-100BT).",
    )
    parser.add_argument(
        "--output_dir",
        default="data/datasets/finewebedu/fineweb-edu-sample-100BT-pretokenized",
        help="Output directory for pretokenized shards.",
    )
    parser.add_argument(
        "--shard_tokens",
        type=int,
        default=100_000_000,
        help="Number of tokens per shard. Keep 100M for compatibility.",
    )
    parser.add_argument(
        "--max_documents",
        type=int,
        default=None,
        help="Optional limit of source documents for smoke tests.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Optional cap on total written tokens (val + train).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of tokenization worker processes (CPU).",
    )
    parser.add_argument(
        "--text_batch_size",
        type=int,
        default=1024,
        help="Number of documents per tokenization batch.",
    )
    parser.add_argument(
        "--expected_total_chunks",
        type=int,
        default=None,
        help=(
            "Optional strict check for total number of output .bin chunks "
            "(val + train), e.g. 1000 for sample-100BT at 100M/chunk."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done_marker = out_dir / "_PRETOKENIZE_DONE.json"
    if done_marker.exists() and not args.overwrite:
        print(f"[pretokenize] found done marker: {done_marker}. Skipping.")
        return

    # Use streaming to avoid loading the full corpus into memory.
    dataset = load_dataset(
        args.dataset,
        name=args.subset,
        split="train",
        streaming=True,
    )

    shard_tokens = int(args.shard_tokens)
    if shard_tokens <= 0:
        raise ValueError("--shard_tokens must be > 0")
    if int(args.num_workers) <= 0:
        raise ValueError("--num_workers must be > 0")
    if int(args.text_batch_size) <= 0:
        raise ValueError("--text_batch_size must be > 0")

    current = np.empty((shard_tokens,), dtype=np.uint16)
    token_count = 0
    shard_index = 0
    docs_seen = 0
    total_tokens_written = 0
    train_shards_written = 0

    def flush_chunk(tokens_np: np.ndarray) -> None:
        nonlocal shard_index, total_tokens_written, train_shards_written
        split = "val" if shard_index == 0 else "train"
        if split == "val":
            filename = out_dir / "fineweb_val_000000.bin"
        else:
            filename = out_dir / f"fineweb_train_{shard_index:06d}.bin"
            train_shards_written += 1
        _write_bin(filename, tokens_np)
        total_tokens_written += int(tokens_np.size)
        print(
            f"[pretokenize] wrote {filename.name} "
            f"tokens={tokens_np.size:,} split={split}"
        )
        shard_index += 1

    def _consume_tokens(tokens: np.ndarray) -> bool:
        nonlocal token_count
        pos = 0
        while pos < tokens.size:
            token_budget = None
            if args.max_tokens is not None:
                token_budget = int(args.max_tokens) - (
                    total_tokens_written + token_count
                )
                if token_budget <= 0:
                    return True
            remaining = shard_tokens - token_count
            take = min(remaining, tokens.size - pos)
            if token_budget is not None:
                take = min(take, token_budget)
            current[token_count : token_count + take] = tokens[pos : pos + take]
            token_count += take
            pos += take

            if token_count == shard_tokens:
                flush_chunk(current)
                token_count = 0
        return args.max_tokens is not None and (
            total_tokens_written + token_count >= int(args.max_tokens)
        )

    def _iter_text_batches():
        nonlocal docs_seen
        text_batch: list[str] = []
        for row in dataset:
            if args.max_tokens is not None and (
                total_tokens_written + token_count >= int(args.max_tokens)
            ):
                break
            docs_seen += 1
            if args.max_documents is not None and docs_seen > args.max_documents:
                break
            text = _extract_text(row)
            if not text:
                continue
            text_batch.append(text)
            if len(text_batch) >= int(args.text_batch_size):
                yield text_batch
                text_batch = []
        if text_batch:
            yield text_batch

    reached_token_cap = False
    if int(args.num_workers) == 1:
        _init_worker_tokenizer("gpt2")
        for text_batch in _iter_text_batches():
            for tokens in _tokenize_text_batch(text_batch):
                reached_token_cap = _consume_tokens(tokens)
                if reached_token_cap:
                    break
            if reached_token_cap:
                break
    else:
        with mp.Pool(
            processes=int(args.num_workers),
            initializer=_init_worker_tokenizer,
            initargs=("gpt2",),
        ) as pool:
            for token_batch in pool.imap(
                _tokenize_text_batch, _iter_text_batches(), chunksize=1
            ):
                for tokens in token_batch:
                    reached_token_cap = _consume_tokens(tokens)
                    if reached_token_cap:
                        break
                if reached_token_cap:
                    break

    if token_count > 0:
        flush_chunk(current[:token_count].copy())

    chunk_files = sorted(out_dir.glob("fineweb_*.bin"))
    total_chunk_files = len(chunk_files)
    if total_chunk_files != shard_index:
        raise RuntimeError(
            f"Output chunk count mismatch: files={total_chunk_files}, "
            f"shards_written={shard_index}"
        )
    if args.expected_total_chunks is not None:
        expected_chunks = int(args.expected_total_chunks)
        if total_chunk_files != expected_chunks:
            raise RuntimeError(
                f"Expected {expected_chunks} chunks, found {total_chunk_files} in {out_dir}"
            )
        expected_tokens = expected_chunks * shard_tokens
        if total_tokens_written != expected_tokens:
            raise RuntimeError(
                f"Expected {expected_tokens:,} tokens from {expected_chunks} chunks, "
                f"but wrote {total_tokens_written:,}."
            )

    summary = {
        "dataset": args.dataset,
        "subset": args.subset,
        "output_dir": str(out_dir),
        "shard_tokens": shard_tokens,
        "documents_seen": docs_seen,
        "total_tokens_written": total_tokens_written,
        "total_shards_written": shard_index,
        "train_shards_written": train_shards_written,
        "num_workers": int(args.num_workers),
        "text_batch_size": int(args.text_batch_size),
        "expected_total_chunks": args.expected_total_chunks,
    }
    done_marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[pretokenize] complete. Summary written to {done_marker}")


if __name__ == "__main__":
    main()
