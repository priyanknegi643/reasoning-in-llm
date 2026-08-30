#!/bin/bash

python train.py \
--config config/tinystories/probe/nextlat_probe.yaml \
--checkpoint_path output/tinystories/NextLat/ckpt_iter_100001.pt \
--probe_depth 20 \
--seed 1234