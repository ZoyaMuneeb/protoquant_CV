#!/usr/bin/env bash
# Live demo on the trained models: inference, and starting each recovery method's training.
#
#   bash scripts/demo.sh predict [predict.py options]   # inference on test images (default: test image 0)
#   bash scripts/demo.sh train-cfr                      # B1 + cached H and B1-CFR on cached features, < 1 min
#   bash scripts/demo.sh train-ir                       # B1-IR: retraining on augmented images (~41 min; Ctrl-C to stop)
#   bash scripts/demo.sh prune                          # rebuild the one-shot B1 / P1 / random masks, minutes
#
# Examples:
#   bash scripts/demo.sh predict --test_index 0 12 345 --out ../demo
#   bash scripts/demo.sh predict --random 3 --models Dense B1-IR B1-CFR
#   bash scripts/demo.sh predict --image ~/some_bird.jpg
#   PQ_DATASET=cars196 bash scripts/demo.sh predict --random 2
#   EPOCHS=3 bash scripts/demo.sh train-ir             # a short image-retraining run
#
# Uses seed SEED (default 40) of the finished runs: CUB in ../runs_bb2 (improved backbone), Cars in ../runs_cars.
# Everything the training commands write goes to a new folder in ../demo/, so the report's runs are never touched.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export PQ_CUB_PROTOCOL=strict
CMD="${1:-predict}"; shift || true
SEED="${SEED:-40}"
if [ "${PQ_DATASET}" = "cars196" ]; then
    RUNS="${DEMO_RUNS_ROOT:-${PROJECT_ROOT}/runs_cars}"; SEL="${RUNS}/backbone/SELECTED"
else
    RUNS="${DEMO_RUNS_ROOT:-${PROJECT_ROOT}/runs_bb2}"; SEL="${PROJECT_ROOT}/runs/backbone_sweep/SELECTED"
fi
if [ -z "${BACKBONE_CKPT:-}" ]; then
    [ -f "${SEL}" ] || { echo "No ${SEL}: set BACKBONE_CKPT to the fine-tuned backbone .pth"; exit 1; }
    BACKBONE_CKPT="$(cat "${SEL}")"
fi
PQ=$(ls -t "${RUNS}/phase1/strict_s${SEED}/stage2/"pipnet_*.pth 2>/dev/null | head -1)
: "${PQ:?no dense Stage 2 checkpoint for seed ${SEED} in ${RUNS}/phase1}"
MASKING="${RUNS}/phase2/strict_s${SEED}/masking"
OUT="${PROJECT_ROOT}/demo/$(date +%Y%m%d_%H%M%S)_${CMD}"
PY="python ${REPO_ROOT}/scripts"
echo "[DEMO] dataset=${PQ_DATASET} seed=${SEED} backbone=${BACKBONE_CKPT}"
echo "[DEMO] dense model=${PQ}"

case "${CMD}" in
    predict)
        $PY/predict.py --backbone_ckpt "${BACKBONE_CKPT}" --runs_root "${RUNS}" --cub_path "${CUB_PATH}" \
            --seed "${SEED}" "$@" ;;
    train-cfr)   # same code and settings as the report's B1-CFR (and B1 + cached H), on a copy of the B1 mask
        mkdir -p "${OUT}"; cp "${MASKING}/masks.pt" "${OUT}/"
        echo "[DEMO] B1 + cached H and B1-CFR, 145 epochs each on cached features -> ${OUT}"
        $PY/phase2_repair.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ}" \
            --cache_dir "${RUNS}/phase1/feature_cache" --masking_dir "${OUT}" --seed "${SEED}" \
            --keys B1_K10 --cached_retrain --tag demo "$@" 2>&1 | tee "${OUT}/train.log" ;;
    train-ir)    # same code and settings as the report's B1-IR
        mkdir -p "${OUT}"
        WARM=5; [ "${EPOCHS:-145}" -lt 10 ] && WARM=1   # short demo runs only; the report used 5 warm-up epochs
        echo "[DEMO] B1-IR: B1 mask, codes + head on augmented images, ${EPOCHS:-145} epochs -> ${OUT}"
        $PY/phase2_retrain.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ}" --masks "${MASKING}/masks.pt" \
            --mask_key B1_K10 --seed "${SEED}" --cub_path "${CUB_PATH}" --out_dir "${OUT}" \
            --epochs "${EPOCHS:-145}" --warmup_epochs "${WARM}" --batch_size 1024 "$@" 2>&1 | tee "${OUT}/train.log" ;;
    prune)       # one-shot masks from the dense model (B1 magnitude, P1 activation-aware, random, ...)
        mkdir -p "${OUT}"
        $PY/phase2_masking.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ}" \
            --cache_dir "${RUNS}/phase1/feature_cache" --out_dir "${OUT}" --seed "${SEED}" "$@" \
            2>&1 | tee "${OUT}/masking.log" ;;
    *) echo "usage: bash scripts/demo.sh {predict|train-cfr|train-ir|prune} [options]"; exit 1 ;;
esac
