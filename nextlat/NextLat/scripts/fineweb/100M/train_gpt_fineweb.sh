#!/bin/bash
 
CUDA_VISIBLE_DEVICES=0,1,2,3 fabric run --strategy ddp --devices 4 --precision bf16-mixed train.py --config config/fineweb/100M/gpt_finewebedu.yaml