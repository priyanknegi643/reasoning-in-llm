#!/bin/bash

fabric run --main_port 29443 data/manhattan/generate_manhattan_trajectories.py \
    --num_samples 50 \
    --batch_size 128 \
    --checkpoint_path "weights/NextLat-seed1234-1760980370.5558422/ckpt_iter_255000.pt" \
    --data_dir "data/manhattan/random_walks" \
    --model_name "NextLat" \
    --config "weights/NextLat-seed1234-1760980370.5558422/materialized_config.yaml"

cd data/manhattan/
python make_graphs.py \
    --dataset "random_walks" \
    --model_name "NextLat" \
    --nsequences 6400 \
    --degree 8

cd ../..

python data/manhattan/compression_test.py \
    --checkpoint_path "weights/NextLat-seed1234-1760980370.5558422/ckpt_iter_255000.pt" \
    --data_dir "data/manhattan/random_walks" \
    --model_name "NextLat" \
    --config "weights/NextLat-seed1234-1760980370.5558422/materialized_config.yaml" \

python data/manhattan/detour_test.py \
    --checkpoint_path "weights/NextLat-seed1234-1760980370.5558422/ckpt_iter_255000.pt" \
    --data_dir "data/manhattan/random_walks" \
    --model_name "NextLat" \
    --config "weights/NextLat-seed1234-1760980370.5558422/materialized_config.yaml" \

python data/manhattan/latent_compression.py \
    --checkpoint_path "weights/NextLat-seed1234-1760980370.5558422/ckpt_iter_255000.pt" \
    --data_dir "data/manhattan/random_walks" \
    --model_name "NextLat" \
    --config "weights/NextLat-seed1234-1760980370.5558422/materialized_config.yaml" \