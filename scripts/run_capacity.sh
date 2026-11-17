#!/usr/bin/env bash
# Capacity metrics (protocol section 5) for the dense, one-shot and retrained models of Phases 1-2.
#
#   BACKBONE_CKPT=... bash scripts/run_capacity.sh
#
# Needs the GPU to itself (latency). Writes results/capacity.{md,json}. Re-running overwrites them.
export PQ_CUB_PROTOCOL=strict
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${BACKBONE_CKPT:?Set BACKBONE_CKPT to the fine-tuned backbone .pth (full path)}"
SEEDS="${SEEDS:-40 41 42}"
gpu_check
# shellcheck disable=SC2086
python "${REPO_ROOT}/scripts/capacity.py" --backbone_ckpt "${BACKBONE_CKPT}" \
    --phase1_root "${RUNS_ROOT}/phase1" --phase2_root "${RUNS_ROOT}/phase2" --seeds ${SEEDS} \
    --batch_size "${BATCH_SIZE:-128}" --warmup "${WARMUP:-10}" --iters "${ITERS:-50}" \
    --out_json "${REPO_ROOT}/results/capacity.json" --out_md "${REPO_ROOT}/results/capacity.md" \
    2>&1 | tee "${RUNS_ROOT}/capacity.log"
