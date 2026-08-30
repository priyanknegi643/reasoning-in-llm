#!/bin/bash
set -eo pipefail

# Local defaults (override via env vars if needed)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
LOCAL_DEVICES="${LOCAL_DEVICES:-4}"
DATA_DIR="${DATA_DIR:-data/fineweb/pretokenized-10BT}"
CKPT_DIR="${CKPT_DIR:-output/nextlat-finewebedu-100M/NextLat-100M-seed1234}"


CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" fabric run \
  --strategy ddp \
  --devices "${LOCAL_DEVICES}" \
  --precision bf16-mixed \
  train.py \
  --config config/fineweb/100M/nextlat_finewebedu.yaml

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --nproc_per_node="${LOCAL_DEVICES}" eval/eval_speculative_checkpoints.py \
  --train_config "config/fineweb/100M/nextlat_finewebedu.yaml" \
  --checkpoint_dir "${CKPT_DIR}" \
  --last_checkpoint_only \
  --num_samples_per_dataset 64 \
  --prompt_tokens 512 \
  --max_new_tokens 512 \
  --gamma 6 \
  --top_k 1 \
  --wandb_project finewebspec-100M \
  --wandb_run_name nextlat-h1-100M-spec-g6

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --nproc_per_node="${LOCAL_DEVICES}" eval/eval_checkpoints.py \
  --train_config "config/fineweb/100M/nextlat_finewebedu.yaml" \
  --checkpoint_dir "${CKPT_DIR}" \
  --last_checkpoint_only \
  --eval_config config/fineweb/lm_eval_fineweb.yaml