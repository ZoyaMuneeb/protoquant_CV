#!/usr/bin/env bash
# Phase 0: download CUB-200-2011, crop to ground-truth boxes (authors' crop_cub.py),
# create the fixed stratified validation split (if missing) and run the data checks.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "${DATA_ROOT}/raw"
if [ ! -f "${CUB_RAW}/images.txt" ]; then
    curl -L -o "${DATA_ROOT}/raw/CUB_200_2011.tgz" https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
    echo "0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081  ${DATA_ROOT}/raw/CUB_200_2011.tgz" | shasum -a 256 -c -
    tar xzf "${DATA_ROOT}/raw/CUB_200_2011.tgz" -C "${DATA_ROOT}/raw"
fi
if [ ! -f "${CUB_PATH}/CUB_200_2011/images.txt" ]; then
    python "${CODE_DIR}/crop_cub.py" --input_dir "${CUB_RAW}" --output_dir "${CUB_PATH}/CUB_200_2011"
fi
if [ ! -f "${PQ_CUB_SPLIT_FILE}" ]; then
    python "${REPO_ROOT}/scripts/make_cub_val_split.py" --cub_root "${CUB_RAW}" --out "${PQ_CUB_SPLIT_FILE}"
fi
python "${REPO_ROOT}/scripts/check_phase0_data.py" --raw_root "${CUB_RAW}" --cropped_parent "${CUB_PATH}"
