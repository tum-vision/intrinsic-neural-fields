#!/bin/bash

DATA_DIR=data
OUT_EFUNCS_DIR=data/preprocessed/human_efuncs
OUT_DATASET_DIR=data/preprocessed/human_dataset_v2_tiny

NUM_EIGENFUNCTIONS=4096

# Preprocess eigenfunctions
python preprocess_eigenfunctions.py $OUT_EFUNCS_DIR $DATA_DIR/human/RUST_3d_Low1.obj $NUM_EIGENFUNCTIONS

# Preprocess views
python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/human/RUST_3d_Low1.obj $DATA_DIR/human_dataset_v2_tiny train
python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/human/RUST_3d_Low1.obj $DATA_DIR/human_dataset_v2_tiny val
python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/human/RUST_3d_Low1.obj $DATA_DIR/human_dataset_v2_tiny test
