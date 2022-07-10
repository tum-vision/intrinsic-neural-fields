#!/bin/bash

mkdir data

dataPoints=("cat_rescaled_rotated" "cat_dataset_v2_tiny" "cat_tri" "discretizations" "human" "human_dataset_v2_tiny" "human_tri" "shape_descriptors" "texture_transfer")

for dataPoint in ${dataPoints[@]}; do
    wget "https://vision.in.tum.de/webshare/g/intrinsic-neural-fields/data/${dataPoint}.zip" -P data
    unzip "data/${dataPoint}.zip" -d data
    rm -rf "data/${dataPoint}.zip"
done
