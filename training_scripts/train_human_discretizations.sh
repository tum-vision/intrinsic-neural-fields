#!/bin/bash

# Select one of the following methods: "cloud40k", "cloud400k", "dense", "iso", "qes"
method="$1"

if [ "$method" = "cloud40k" ]; then
    echo "Selected discretization: cloud40k"
    CONFIG_PATH=configs/discretization_agnostic/human_cloud40k.yaml
    EVAL_OUT_DIR=out/discretization/human_cloud40k/test_eval
elif [ "$method" = "cloud400k" ]; then
    echo "Selected discretization: cloud400k"
    CONFIG_PATH=configs/discretization_agnostic/human_cloud400k.yaml
    EVAL_OUT_DIR=out/discretization/human_cloud400k/test_eval
elif [ "$method" = "dense" ]; then
    echo "Selected discretization: dense"
    CONFIG_PATH=configs/discretization_agnostic/human_dense.yaml
    EVAL_OUT_DIR=out/discretization/human_dense/test_eval
elif [ "$method" = "iso" ]; then
    echo "Selected discretization: iso"
    CONFIG_PATH=configs/discretization_agnostic/human_iso.yaml
    EVAL_OUT_DIR=out/discretization/human_iso/test_eval
elif [ "$method" = "qes" ]; then
    echo "Selected discretization: qes"
    CONFIG_PATH=configs/discretization_agnostic/human_qes.yaml
    EVAL_OUT_DIR=out/discretization/human_qes/test_eval
else
    echo "Unknown method: $method. Must be one of the following: cloud40k, cloud400k, dense, iso, qes"
    exit 1
fi

# Perform training
python train.py $CONFIG_PATH --allow_checkpoint_loading

# Evaluation on test split
python eval.py $EVAL_OUT_DIR $CONFIG_PATH data/human_dataset_v2_tiny test
