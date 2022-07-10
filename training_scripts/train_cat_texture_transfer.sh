#!/bin/bash

CONFIG_PATH=configs/texture_transfer/cat_orig.yaml
EVAL_OUT_DIR=out/texture_transfer_source/orig_cat/test_eval

# Perform training
python train.py $CONFIG_PATH --allow_checkpoint_loading

# Evaluation on test split
python eval.py $EVAL_OUT_DIR $CONFIG_PATH data/cat_dataset_v2_tiny test
