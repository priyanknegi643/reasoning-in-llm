#!/bin/bash

fabric run --main_port 29421 --strategy ddp --precision bf16-mixed --devices 8 train.py --config config/stargraph/7_7/bst_stargraph_7_7.yaml