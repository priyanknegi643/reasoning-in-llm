#!/bin/bash

fabric run --strategy ddp --devices 8 --precision bf16-mixed train.py --config config/manhattan/bst_manhattan.yaml