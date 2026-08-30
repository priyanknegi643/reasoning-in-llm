#!/bin/bash

fabric run --strategy ddp --devices 2 --precision bf16-mixed train.py --config config/manhattan/mtp_manhattan.yaml