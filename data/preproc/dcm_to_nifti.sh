#!/bin/bash

DICOM_ROOT="/NFS/MRI/ADNI/origin/t1w/ADNI_1"
OUTPUT_ROOT="/NFS/MRI/ADNI/origin/t1w/ADNI_1_nifti"

for subject_dir in "$DICOM_ROOT"/*; do
    if [ -d "$subject_dir" ]; then
        subject_name=$(basename "$subject_dir")

        ses_dirs="$subject_dir/MP-RAGE/"* # Accelerated_Sagittal_MPRAGE for ADNI_3

        for ses_dir in $ses_dirs; do
            if [ -d "$ses_dir" ]; then

                for preproc_dir in "$ses_dir"/*; do
                    file_name=$(ls -1 "$preproc_dir" | head -n 1)
                    file_name="${file_name%.dcm}"

                    output_dir="$OUTPUT_ROOT/$subject_name"

                    # 출력 폴더 생성
                    mkdir -p "$output_dir"

                    # dcm2niix 실행
                    dcm2niix -f "${file_name}" -o "$output_dir" "$preproc_dir"
                done

            fi
        done
    fi
done
