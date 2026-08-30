#!/bin/bash

fabric run --precision bf16-mixed --devices 2 --strategy ddp train.py --config config/stargraph/2_10/nextlat_stargraph_2_10.yaml