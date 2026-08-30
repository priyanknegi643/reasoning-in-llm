#!/bin/bash

fabric run  --main_port 29421  --strategy ddp --devices 8 train.py --config config/stargraph/5_5/bst_stargraph_5_5.yaml