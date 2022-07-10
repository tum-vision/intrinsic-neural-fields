#!/bin/bash

DATA_DIR=data
OUT_EFUNCS_DIR=data/preprocessed/human_qes
OUT_DATASET_DIR=data/preprocessed/human_qes

NUM_EIGENFUNCTIONS=4096

# Preprocess eigenfunctions
python preprocess_eigenfunctions.py $OUT_EFUNCS_DIR $DATA_DIR/discretizations/human/qes.ply $NUM_EIGENFUNCTIONS --laplacian_type robust

# Preprocess views
python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/discretizations/human/qes.ply $DATA_DIR/human_dataset_v2_tiny train
python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/discretizations/human/qes.ply $DATA_DIR/human_dataset_v2_tiny val
python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/discretizations/human/qes.ply $DATA_DIR/human_dataset_v2_tiny test
