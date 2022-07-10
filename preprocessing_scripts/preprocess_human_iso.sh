#!/bin/bash

DATA_DIR=data
OUT_EFUNCS_DIR=data/preprocessed/human_isotropic
OUT_DATASET_DIR=data/preprocessed/human_isotropic

NUM_EIGENFUNCTIONS=4096

# Preprocess eigenfunctions
python preprocess_eigenfunctions.py $OUT_EFUNCS_DIR $DATA_DIR/discretizations/human/iso.ply $NUM_EIGENFUNCTIONS --laplacian_type robust

# Preprocess views
python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/discretizations/human/iso.ply $DATA_DIR/human_dataset_v2_tiny train
python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/discretizations/human/iso.ply $DATA_DIR/human_dataset_v2_tiny val
python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/discretizations/human/iso.ply $DATA_DIR/human_dataset_v2_tiny test
