#!/bin/bash

# Select one of the following methods: "tf+rff", "neutex", "intrinsic"
method="$1"

if [ "$method" = "intrinsic" ]; then
    echo "Selected method: Intrinsic"
    CONFIG_PATH=configs/texture_reconstruction/intrinsic_cat.yaml
    EVAL_OUT_DIR=out/texture_recon/intrinsic_cat/test_eval
elif [ "$method" = "tf+rff" ]; then
    echo "Selected method: TF + RFF"
    CONFIG_PATH=configs/texture_reconstruction/tf_rff_cat.yaml
    EVAL_OUT_DIR=out/texture_recon/tf_rff_cat/test_eval
elif [ "$method" = "neutex" ]; then
    echo "Selected method: NeuTex"
    CONFIG_PATH=configs/texture_reconstruction/neutex_cat.yaml
    EVAL_OUT_DIR=out/texture_recon/neutex_cat/test_eval
else
    echo "Unknown method: $method. Must be one of the following: tf+rff, neutex, intrinsic"
    exit 1
fi

# Perform training
python train.py $CONFIG_PATH --allow_checkpoint_loading

# Evaluation on test split and bake the texture
python eval.py $EVAL_OUT_DIR $CONFIG_PATH data/cat_dataset_v2_tiny test --uv_mesh_path data/cat_tri/12221_Cat_v1_l3.obj
