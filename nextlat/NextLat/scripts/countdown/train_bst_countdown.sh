#!/bin/bash

fabric run  --strategy ddp --devices 8  train.py  --config config/countdown/bst_countdown.yaml