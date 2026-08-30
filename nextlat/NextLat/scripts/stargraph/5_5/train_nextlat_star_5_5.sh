#!/bin/bash

fabric run --precision bf16-mixed --devices 2 --strategy ddp train.py --config config/stargraph/5_5/nextlat_stargraph_5_5.yaml