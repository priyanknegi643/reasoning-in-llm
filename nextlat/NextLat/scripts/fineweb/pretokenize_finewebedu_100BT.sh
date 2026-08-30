#!/bin/bash
set -eo pipefail

python data/fineweb/pretokenize_finewebedu.py \
  --subset sample-100BT \
  --output_dir data/fineweb/pretokenized-100BT \
  --shard_tokens 100000000 \
  --num_workers 96 \
  --text_batch_size 1024 \
  --expected_total_chunks 1000
