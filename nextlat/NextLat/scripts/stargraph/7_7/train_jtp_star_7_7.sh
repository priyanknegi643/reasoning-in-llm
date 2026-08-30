#!/bin/bash

fabric run --precision bf16-mixed --devices 2 --strategy ddp train.py --config config/stargraph/7_7/jtp_stargraph_7_7.yaml