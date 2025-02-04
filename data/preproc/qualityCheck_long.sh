
#!/bin/bash

# MRI 데이터 폴더 경로
MRI_DIR="/NFS/FutureBrainGen/data/long/img"

# QC에 실패한 MRI 파일들을 기록할 텍스트 파일
BAD_QC_FILE="bad_qc_files_long.txt"

# 기존의 결과 파일이 있으면 삭제
if [ -f "$BAD_QC_FILE" ]; then
    rm "$BAD_QC_FILE"
fi

# 모든 MRI 파일들에 대해 QC 진행
for mri in "$MRI_DIR"/*; do
    if [ -f "$mri" ] ; then  # MRI 파일이 실제 파일인 경우만 처리
        
        # BET 파일 저장 경로 설정
        output_file="${mri%.*}_brain_extracted"

        # FSL BET (Brain Extraction Tool) 사용하여 뇌 추출
        bet "$mri" "$output_file" -R -f 0.5 -g 0

        # 뇌 추출 결과가 정상적인지 확인
        if [ ! -f "${output_file}.nii.gz" ]; then
            echo "$mri has QC issues (brain extraction failed)"
            echo "$mri" >> "$BAD_QC_FILE"
            continue
        fi

        # BET 뇌 추출 결과에 대한 QC 확인 (너무 적은 뇌 조직 남거나, 과도한 잡음이 있는지)
        brain_vol=$(fslstats "${output_file}.nii.gz" -V | awk '{print $1}')
        echo "Processing $mri ... "
        echo $brain_vol
        # 뇌 볼륨이 너무 작으면 오류로 간주
        if [ $brain_vol -lt 570000 ]; then
            echo "$mri has QC issues (small brain volume)"
            echo "$mri" >> "$BAD_QC_FILE"
            # rm -f "${mri}"
        fi
        rm -f "${output_file}.nii.gz"
        
    fi
done

echo "QC check complete. Issues logged in $BAD_QC_FILE."