#!/bin/bash

names=("3m_high_tack_spray_adhesive" "advil_liqui_gels" "band_aid_clear_strips" "band_aid_sheer_strips" "blue_clover_baby_toy" "bumblebee_albacore" "campbells_chicken_noodle_soup" "campbells_soup_at_hand_creamy_tomato" "canon_ack_e10_box" "cheez_it_white_cheddar")

for name in ${names[@]}; do
    DATA_DIR=data/bigbird/processed/${name}
    OUT_EFUNCS_DIR=data/preprocessed/bigbird/${name}
    OUT_DATASET_DIR=data/preprocessed/bigbird/${name}

    NUM_EIGENFUNCTIONS=4096

    # Preprocess eigenfunctions
    python preprocess_eigenfunctions.py $OUT_EFUNCS_DIR $DATA_DIR/mesh_world.ply $NUM_EIGENFUNCTIONS --laplacian_type robust

    # Preprocess views
    python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/mesh_world.ply $DATA_DIR train
    python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/mesh_world.ply $DATA_DIR val
    python preprocess_dataset.py $OUT_DATASET_DIR $DATA_DIR/mesh_world.ply $DATA_DIR test
done

