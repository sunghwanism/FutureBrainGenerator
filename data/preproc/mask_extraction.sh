#!/bin/bash

INPUT_DIR="/NFS/FutureBrainGen/data/long/down_img_2mm"

OUTPUT_DIR="/NFS/FutureBrainGen/data/long/down_img_2mm_mask"

for file in ${INPUT_DIR}/*.nii.gz; do
    filename=$(basename -- "$file")
    filename_noext="${filename%.nii.gz}"

    echo "Processing: $filename"

    fast -t 1 -n 3 -o "${OUTPUT_DIR}/${filename_noext}" "$file"

    mv "${OUTPUT_DIR}/${filename_noext}_pve_0.nii.gz" "${OUTPUT_DIR}/${filename_noext}_csf.nii.gz"
    mv "${OUTPUT_DIR}/${filename_noext}_pve_1.nii.gz" "${OUTPUT_DIR}/${filename_noext}_gm.nii.gz"
    mv "${OUTPUT_DIR}/${filename_noext}_pve_2.nii.gz" "${OUTPUT_DIR}/${filename_noext}_wm.nii.gz"
    mv "${OUTPUT_DIR}/${filename_noext}_seg.nii.gz" "${OUTPUT_DIR}/${filename_noext}_seg.nii.gz"

    fslmaths "${OUTPUT_DIR}/${filename_noext}_csf.nii.gz" \
        -add "${OUTPUT_DIR}/${filename_noext}_gm.nii.gz" \
        -add "${OUTPUT_DIR}/${filename_noext}_wm.nii.gz" \
        -thr 0.2 -bin "${OUTPUT_DIR}/${filename_noext}_mask.nii.gz"

    echo "Completed: $filename"
done

echo "All files processed successfully!"
