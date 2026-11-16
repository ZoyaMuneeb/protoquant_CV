#!/usr/bin/env bash
# Phase 1: run the official pipeline and build the baseline.
#
#   Strict baseline (3 seeds):        BACKBONE_CKPT=... bash scripts/run_phase1.sh
#   Paper-protocol sanity (1 seed):   BACKBONE_CKPT=... PROTOCOL=paper bash scripts/run_phase1.sh
#
# Per seed: Stage 1 (4,096 codes) -> B0 (prune to 3,000 on the protocol's validation loader) -> Stage 2
# -> evaluation on validation and, once, on test. Strict runs also cache backbone features (once) and run
# implementation Tests 1-4. The backbone itself is evaluated once on test as the black-box baseline.
#
# Resumable: each finished step writes a DONE file; rerunning skips finished steps and restarts an
# unfinished one from scratch. Everything goes to ${RUNS_ROOT}/phase1/<protocol>_s<seed>/.
# Summary: results/phase1_baseline.md (python scripts/phase1_summary.py).
export PQ_CUB_PROTOCOL="${PROTOCOL:-${PQ_CUB_PROTOCOL:-strict}}"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${BACKBONE_CKPT:?Set BACKBONE_CKPT to the fine-tuned backbone .pth (full path)}"
BACKBONE_CKPT="$(cd "$(dirname "${BACKBONE_CKPT}")" && pwd)/$(basename "${BACKBONE_CKPT}")"
export BACKBONE_CKPT
if [ "${PQ_CUB_PROTOCOL}" = "strict" ]; then SEEDS="${SEEDS:-40 41 42}"; else SEEDS="${SEEDS:-40}"; fi
BATCH="${BATCH_SIZE:-1024}"          # Stage 1/2 training batch
B0_BATCH="${B0_BATCH:-256}"          # B0 is evaluation only
CODEBOOK_ENTRIES="${CODEBOOK_ENTRIES:-4096}"
TARGET_ENTRIES="${TARGET_ENTRIES:-3000}"
P1_ROOT="${RUNS_ROOT}/phase1"
CACHE_DIR="${P1_ROOT}/feature_cache"
mkdir -p "${P1_ROOT}"
PY="python ${REPO_ROOT}/scripts"
echo "[PHASE1] protocol=${PQ_CUB_PROTOCOL} seeds=${SEEDS} batch=${BATCH} backbone=${BACKBONE_CKPT}"

step() {  # step <dir> <command...>: run once; DONE marks success; a half-finished dir is cleared first
    local d=$1; shift
    if [ -f "$d/DONE" ]; then echo "[PHASE1] skip (done): $d"; return 0; fi
    rm -rf "$d"; mkdir -p "$d"
    echo "[PHASE1] start: $d"
    "$@"
    date > "$d/DONE"
    echo "[PHASE1] done:  $d"
}
newest() { ls -t $1 2>/dev/null | head -1; }

# Black-box baseline: the backbone's own classifier on test (read once per protocol).
bb_eval() {
    $PY/eval_protoquant.py --backbone_ckpt "${BACKBONE_CKPT}" --split test --cub_path "${CUB_PATH}" \
        --out "${P1_ROOT}/backbone_${PQ_CUB_PROTOCOL}/eval_test" 2>&1 | tee "${P1_ROOT}/backbone_${PQ_CUB_PROTOCOL}/eval.log"
}
step "${P1_ROOT}/backbone_${PQ_CUB_PROTOCOL}" bb_eval

if [ "${PQ_CUB_PROTOCOL}" = "strict" ]; then
    cache() {
        $PY/cache_features.py --backbone_ckpt "${BACKBONE_CKPT}" --cub_path "${CUB_PATH}" \
            --out_dir "${CACHE_DIR}" 2>&1 | tee "${CACHE_DIR}.log"
    }
    if [ ! -f "${CACHE_DIR}/DONE" ]; then mkdir -p "${CACHE_DIR}"; cache; date > "${CACHE_DIR}/DONE"; fi
fi

for SEED in ${SEEDS}; do
    export SEED
    BASE="${P1_ROOT}/${PQ_CUB_PROTOCOL}_s${SEED}"
    mkdir -p "${BASE}"

    step "${BASE}/stage1" env RUN_DIR="${BASE}/stage1" BATCH_SIZE="${BATCH}" CODEBOOK_ENTRIES="${CODEBOOK_ENTRIES}" \
        bash "${REPO_ROOT}/scripts/run_stage1_codebook.sh"
    S1_CB=$(newest "${BASE}/stage1/codebook_*.pth")
    : "${S1_CB:?no Stage 1 codebook in ${BASE}/stage1}"

    step "${BASE}/b0" env RUN_DIR="${BASE}/b0" BATCH_SIZE="${B0_BATCH}" CODEBOOK_CKPT="${S1_CB}" CODEBOOK_ENTRIES="${CODEBOOK_ENTRIES}" \
        TARGET_ENTRIES="${TARGET_ENTRIES}" bash "${REPO_ROOT}/scripts/run_b0_prune.sh"
    B0_CB=$(newest "${BASE}/b0/pruned_*.pth")
    : "${B0_CB:?no B0 codebook in ${BASE}/b0}"

    step "${BASE}/stage2" env RUN_DIR="${BASE}/stage2" BATCH_SIZE="${BATCH}" CODEBOOK_CKPT="${B0_CB}" \
        bash "${REPO_ROOT}/scripts/run_stage2_head.sh"
    PQ=$(newest "${BASE}/stage2/pipnet_*.pth")
    : "${PQ:?no Stage 2 checkpoint in ${BASE}/stage2 (did validation run? see train.log)}"

    evaluate() {
        $PY/eval_protoquant.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ}" --split val \
            --cub_path "${CUB_PATH}" --out "${BASE}/eval/eval_val"
        # Final model is fixed (best validation epoch); test is read once here.
        $PY/eval_protoquant.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ}" --split test \
            --cub_path "${CUB_PATH}" --out "${BASE}/eval/eval_test"
    }
    step "${BASE}/eval" evaluate

    if [ "${PQ_CUB_PROTOCOL}" = "strict" ]; then
        tests() {
            $PY/phase1_tests.py run --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ}" \
                --cache_dir "${CACHE_DIR}" --cub_path "${CUB_PATH}" --b0_log "${BASE}/b0/train.log" \
                --out "${BASE}/tests/tests.json" 2>&1 | tee "${BASE}/tests/tests.log"
        }
        step "${BASE}/tests" tests
    fi
done

$PY/phase1_summary.py --phase1_root "${P1_ROOT}" --out_md "${REPO_ROOT}/results/phase1_baseline.md" \
    --out_json "${REPO_ROOT}/results/phase1_baseline.json"
