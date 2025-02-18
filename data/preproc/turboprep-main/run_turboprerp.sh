#!/bin/bash

# Input and output file paths passed as arguments to the script
input_file=$1
output_file=$2
template_image=$3

# Template image (fixed)
# template_image="/data/turboprep/MNI152_T1_1mm_brain.nii.gz"

# Check if input files exist
if [[ ! -f "$input_file" || ! -f "$output_file" ]]; then
  echo "Error: One or both of the input/output files do not exist."
  exit 1
fi

# Combine both files line by line (same order)
paste "$input_file" "$output_file" | while IFS=$'\t' read -r input_path output_path; do
    echo "Running turboprep on input: $input_path"
    echo "Output will be saved to: $output_path"

    if [[ ! -f "$input_path" ]]; then
        echo "Error: Input file \"$input_path\" does not exist."
        continue
    fi

    # Run the turboprep-docker command
    sudo -E ./turboprep-docker "$input_path" "$output_path" "$template_image"

    echo "Finished processing $input_path"
done

