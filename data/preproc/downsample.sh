#!/bin/bash

input_dir="/NFS/MRI/Image"
output_dir="/NFS/MRI/Image_downsampled"

mkdir -p "$output_dir"

for input_file in "$input_dir"/*.nii; do
    base_name=$(basename "$input_file" .nii)
    
    output_file="$output_dir/${base_name}_2mm.nii"
    
    # flirt -in "$input_file" -ref "$input_file" -applyisoxfm 2.0 -out "$output_file"
    
done