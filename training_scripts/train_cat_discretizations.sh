#!/bin/bash

# Select one of the following methods: "cloud10k", "cloud100k", "dense", "iso", "qes"
method="$1"

if [ "$method" = "cloud10k" ]; then
    echo "Selected discretization: cloud10k"
    CONFIG_PATH=configs/discretization_agnostic/cat_cloud10k.yaml
    EVAL_OUT_DIR=out/discretization/cat_cloud10k/test_eval
elif [ "$method" = "cloud100k" ]; then
    echo "Selected discretization: cloud100k"
    CONFIG_PATH=configs/discretization_agnostic/cat_cloud100k.yaml
    EVAL_OUT_DIR=out/discretization/cat_cloud100k/test_eval
elif [ "$method" = "dense" ]; then
    echo "Selected discretization: dense"
    CONFIG_PATH=configs/discretization_agnostic/cat_dense.yaml
    EVAL_OUT_DIR=out/discretization/cat_dense/test_eval
elif [ "$method" = "iso" ]; then
    echo "Selected discretization: iso"
    CONFIG_PATH=configs/discretization_agnostic/cat_iso.yaml
    EVAL_OUT_DIR=out/discretization/cat_iso/test_eval
elif [ "$method" = "qes" ]; then
    echo "Selected discretization: qes"
    CONFIG_PATH=configs/discretization_agnostic/cat_qes.yaml
    EVAL_OUT_DIR=out/discretization/cat_qes/test_eval
else
    echo "Unknown method: $method. Must be one of the following: cloud10k, cloud100k, dense, iso, qes"
    exit 1
fi

# Perform training
python train.py $CONFIG_PATH --allow_checkpoint_loading

# Evaluation on test split
python eval.py $EVAL_OUT_DIR $CONFIG_PATH data/cat_dataset_v2_tiny test
