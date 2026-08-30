#!/bin/bash
set -eo pipefail

NUM_GPUS="${NUM_GPUS:-8}"


fabric run --strategy ddp --devices "${NUM_GPUS}" --precision bf16-mixed train.py --config config/fineweb/1B/gpt_finewebedu_1b_100b.yaml trainer.experiment_name="GPT-1B"
torchrun --nproc_per_node="${NUM_GPUS}" eval/eval_checkpoints.py --train_config config/fineweb/1B/gpt_finewebedu_1b_100b.yaml --checkpoint_dir "data/nextlat-finewebedu/GPT-1B-seed1234" --eval_config config/fineweb/lm_eval_fineweb.yaml --last_checkpoint_only --wandb_project fineweb-lmeval
