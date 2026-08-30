#!/bin/bash
set -eo pipefail

NUM_GPUS="${NUM_GPUS:-8}"


fabric run --strategy ddp --devices "${NUM_GPUS}" --precision bf16-mixed train.py --config config/fineweb/1B/jtp_finewebedu_1b_100b_horizon1.yaml trainer.experiment_name="JTP-1B-horizon1"
torchrun --nproc_per_node="${NUM_GPUS}" eval/eval_checkpoints.py --train_config config/fineweb/1B/jtp_finewebedu_1b_100b_horizon1.yaml --checkpoint_dir "data/nextlat-finewebedu/JTP-1B-horizon1-seed1234" --eval_config config/fineweb/lm_eval_fineweb.yaml --last_checkpoint_only --wandb_project fineweb-lmeval
python3 -m pip install -q "datasets==4.6.1"
torchrun --nproc_per_node="${NUM_GPUS}" eval/eval_speculative_checkpoints.py --train_config config/fineweb/1B/jtp_finewebedu_1b_100b_horizon1.yaml --checkpoint_dir "data/nextlat-finewebedu/JTP-1B-horizon1-seed1234" --last_checkpoint_only --num_samples_per_dataset 1024 --prompt_tokens 512 --max_new_tokens 512 --gamma 2 --top_k 1 --output_dir output/speculative_eval/jtp-finewebedu-h1 --wandb_project finewebspec --wandb_run_name jtp-finewebedu-spec-h1-gamma2 --wandb_tags speculative,decoding,finewebedu,jtp,horizon1,gamma2
