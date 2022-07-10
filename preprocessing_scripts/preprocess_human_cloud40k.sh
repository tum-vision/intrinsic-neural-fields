#!/bin/bash

DATA_DIR=data
OUT_EFUNCS_DIR=data/preprocessed/human_cloud40000
OUT_DATASET_DIR=data/preprocessed/human_cloud40000

NUM_EIGENFUNCTIONS=4096

# Preprocess eigenfunctions
python preprocess_eigenfunctions.py $OUT_EFUNCS_DIR $DATA_DIR/discretizations/human/cloud_40000.ply $NUM_EIGENFUNCTIONS --laplacian_type pc_vert_robust

# Preprocess views
python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/discretizations/human/cloud_40000.ply $DATA_DIR/human_dataset_v2_tiny train
python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/discretizations/human/cloud_40000.ply $DATA_DIR/human_dataset_v2_tiny val
python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/discretizations/human/cloud_40000.ply $DATA_DIR/human_dataset_v2_tiny test
