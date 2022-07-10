#!/bin/bash

# Select one of the following shape descriptors: "efuncs", "hks", "shot"
shapeDescriptor="$1"

if [ "$method" = "efuncs" ]; then
    echo "Selected shape descriptor: Eigenfunctions"
    CONFIG_PATH=configs/shape_descriptors/human_efuncs.yaml
    EVAL_OUT_DIR=out/shape_descriptors/human_efuncs/test_eval
elif [ "$method" = "hks" ]; then
    echo "Selected shape descriptor: HKS"
    CONFIG_PATH=configs/shape_descriptors/human_hks.yaml
    EVAL_OUT_DIR=out/shape_descriptors/human_efuncs/test_eval
elif [ "$method" = "shot" ]; then
    echo "Selected shape descriptor: SHOT"
    CONFIG_PATH=configs/shape_descriptors/human_shot.yaml
    EVAL_OUT_DIR=out/shape_descriptors/human_efuncs/test_eval
else
    echo "Unknown shape descriptor: $method. Must be one of the following: efuncs, hks, shot"
    exit 1
fi

# Perform training
python train.py $CONFIG_PATH --allow_checkpoint_loading

# Evaluation on test split and bake the texture
python eval.py $EVAL_OUT_DIR $CONFIG_PATH data/human_dataset_v2_tiny test --uv_mesh_path data/human_tri/RUST_3d_Low1.obj
