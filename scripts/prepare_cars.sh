#!/usr/bin/env bash
# Stanford Cars for the replication (docs/phase0_protocol.md, section 14): convert the download to the CUB
# layout, make the fixed held-out validation split (10 % of the official training images, stratified, seed 0),
# and check that the strict loader reads it. Test images stay test.
#
#   CARS_SRC=/path/to/downloaded/stanford_cars bash scripts/prepare_cars.sh
#
# CARS_SRC holds either the devkit files (cars_train_annos.mat, cars_test_annos_withlabels.mat, cars_meta.mat,
# cars_train/, cars_test/) or cars_annos.mat + car_ims/. Output: ${DATA_ROOT}/cars196/stanford_cars and
# splits/cars196_val10_seed0.txt (commit the split file so every run uses the same one).
export PQ_DATASET=cars196
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${CARS_SRC:?Set CARS_SRC to the folder with the downloaded Stanford Cars files}"
if [ ! -f "${CUB_PATH}/stanford_cars/images.txt" ]; then
    python "${REPO_ROOT}/scripts/prepare_cars.py" --src "${CARS_SRC}" --out "${CUB_PATH}"
fi
if [ ! -f "${PQ_CARS_SPLIT_FILE}" ]; then
    python "${REPO_ROOT}/scripts/make_cub_val_split.py" --cub_root "${CUB_PATH}/stanford_cars" \
        --out "${PQ_CARS_SPLIT_FILE}" --name "Stanford Cars"
fi
PQ_CUB_PROTOCOL=strict python - "${CUB_PATH}" "${CODE_DIR}" <<'PY'
import sys
sys.path.insert(0, sys.argv[2])
from src.datasets.cub import Cars196
n = {s: len(Cars196(root=sys.argv[1], split=s)) for s in ("train", "val", "test")}
ok = n["train"] + n["val"] == 8144 and n["test"] == 8041
print(f"[CARS] strict loader: {n} -> {'PASS' if ok else 'CHECK (counts differ from the official 8,144 / 8,041)'}")
PY
