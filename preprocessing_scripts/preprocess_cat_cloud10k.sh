#!/bin/bash

DATA_DIR=data
OUT_EFUNCS_DIR=data/preprocessed/cat_cloud10000
OUT_DATASET_DIR=data/preprocessed/cat_cloud10000

NUM_EIGENFUNCTIONS=4096

# Preprocess eigenfunctions
python preprocess_eigenfunctions.py $OUT_EFUNCS_DIR $DATA_DIR/discretizations/cat/cloud_10000.ply $NUM_EIGENFUNCTIONS --laplacian_type pc_vert_robust

# Preprocess views
python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/discretizations/cat/cloud_10000.ply $DATA_DIR/cat_dataset_v2_tiny train
python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/discretizations/cat/cloud_10000.ply $DATA_DIR/cat_dataset_v2_tiny val
python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/discretizations/cat/cloud_10000.ply $DATA_DIR/cat_dataset_v2_tiny test
