#!/bin/bash

fabric run  --main_port 29421 --strategy ddp --devices 8 train.py --config config/stargraph/2_10/bst_stargraph_2_10.yaml