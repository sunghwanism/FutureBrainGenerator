#!/bin/bash

input_dir="/NFS/FutureBrainGen/data/long/img"
output_dir="/NFS/FutureBrainGen/data/long/down_img_1.7mm"

mkdir -p "$output_dir"

for input_file in "$input_dir"/*.nii; do
    base_name=$(basename "$input_file" .nii)
    
    output_file="$output_dir/${base_name}.nii"
    
    flirt -in "$input_file" -ref "$input_file" -applyisoxfm 1.7 -out "$output_file"
    
done