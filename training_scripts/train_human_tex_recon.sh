#!/bin/bash

# Select one of the following methods: "tf+rff", "neutex", "intrinsic"
method="$1"

if [ "$method" = "intrinsic" ]; then
    echo "Selected method: Intrinsic"
    CONFIG_PATH=configs/texture_reconstruction/intrinsic_human.yaml
    EVAL_OUT_DIR=out/texture_recon/intrinsic_human/test_eval
elif [ "$method" = "tf+rff" ]; then
    echo "Selected method: TF + RFF"
    CONFIG_PATH=configs/texture_reconstruction/tf_rff_human.yaml
    EVAL_OUT_DIR=out/texture_recon/tf_rff_human/test_eval
elif [ "$method" = "neutex" ]; then
    echo "Selected method: NeuTex"
    CONFIG_PATH=configs/texture_reconstruction/neutex_human.yaml
    EVAL_OUT_DIR=out/texture_recon/neutex_human/test_eval
else
    echo "Unknown method: $method. Must be one of the following: tf+rff, neutex, intrinsic"
    exit 1
fi

# Perform training
python train.py $CONFIG_PATH --allow_checkpoint_loading

# Evaluation on test split and bake the texture
python eval.py $EVAL_OUT_DIR $CONFIG_PATH data/human_dataset_v2_tiny test --uv_mesh_path data/human_tri/RUST_3d_Low1.obj
