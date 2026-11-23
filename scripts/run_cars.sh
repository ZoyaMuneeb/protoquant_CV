#!/usr/bin/env bash
# Stanford Cars replication of the main CUB findings (docs/phase0_protocol.md, section 14).
#
#   bash scripts/prepare_cars.sh       # once: data in the CUB layout + the fixed validation split
#   bash scripts/run_cars.sh           # everything below, resumable
#
# Same strict protocol as CUB (90 % train / 10 % held-out validation, test read once per final model), same
# recipes, K = 10, the improved-backbone recipe without a sweep:
#  1. backbone: ConvNeXt-Tiny from ImageNet, lr 2e-4 for 30 epochs, then continued at lr 5e-5 for 30 epochs;
#     the run with the better validation top-1 is used (as the CUB sweep chose);
#  2. dense ProtoQuant (Stage 1 4,096 codes -> B0 3,000 -> Stage 2), SEEDS (default 40), + feature cache;
#  3. one-shot masks (magnitude, activation-aware, contrast, global budget, random; K = 5/10/20) and code-level
#     masks (M = 1,200 ... 400);
#  4. the code budget: the code-level activation-aware mask whose active codes are closest to 68 % of the
#     magnitude mask's (890 / 1,304 on CUB), chosen from code counts only (no accuracy);
#  5. head refit (CE, step count chosen on validation) and the retraining recipe on cached features (head only,
#     codes + head) for magnitude, activation-aware, random and the chosen code-level mask;
#  6. full retraining on images, 145 epochs: magnitude ("Pruned & Tuned") and the chosen code-level mask,
#     RETRAIN_SEEDS (default 40).
# Runs go to ../runs_cars, summaries to results/cars. About 7 GPU-hours with one seed.
export PQ_DATASET=cars196
export PQ_CUB_PROTOCOL=strict
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export RUNS_ROOT="${CARS_RUNS_ROOT:-${PROJECT_ROOT}/runs_cars}"
export RESULTS_DIR="${CARS_RESULTS_DIR:-${REPO_ROOT}/results/cars}"
export SEEDS="${SEEDS:-40}"
RETRAIN_SEEDS="${RETRAIN_SEEDS:-40}"
mkdir -p "${RUNS_ROOT}" "${RESULTS_DIR}"
[ -f "${CUB_PATH}/stanford_cars/images.txt" ] || { echo "No Cars data in ${CUB_PATH}: run scripts/prepare_cars.sh"; exit 1; }
[ -f "${PQ_CARS_SPLIT_FILE}" ] || { echo "No split ${PQ_CARS_SPLIT_FILE}: run scripts/prepare_cars.sh"; exit 1; }
S="${REPO_ROOT}/scripts"
PY="python ${S}"
P1_ROOT="${RUNS_ROOT}/phase1"
P2_ROOT="${RUNS_ROOT}/phase2"
newest() { ls -t $1 2>/dev/null | head -1; }
echo "[CARS] data=${CUB_PATH} split=${PQ_CARS_SPLIT_FILE} runs=${RUNS_ROOT} results=${RESULTS_DIR} seeds=${SEEDS}"

# 1. Backbone (two runs, the better one on validation is kept).
BB="${RUNS_ROOT}/backbone"
best_val() { grep -o "Validation | Top-1 Accuracy: [0-9.]*" "$1/train.log" | grep -o "[0-9.]*$" | sort -g | tail -1; }
ckpt_of() { ls -t "$1"/ckpt/*.pth 2>/dev/null | head -1; }
bb_run() {  # bb_run <name> <lr> [load_checkpoint]
    local d="${BB}/$1"
    if [ -f "$d/DONE" ]; then echo "[CARS] skip backbone (done): $1 (val $(best_val "$d"))"; return 0; fi
    rm -rf "$d"; mkdir -p "$d/ckpt"
    RUN_DIR="$d" BACKBONE_DIR="$d/ckpt" LR="$2" EPOCHS=30 BATCH_SIZE="${BB_BATCH:-128}" SEED=42 \
        LOAD_CHECKPOINT="${3:-}" bash "${S}/run_backbone_finetune.sh"
    [ -n "$(ckpt_of "$d")" ] || { echo "[CARS] no backbone checkpoint in $d"; exit 1; }
    date > "$d/DONE"
}
bb_run lr_2e-4 2e-4
bb_run continue_lr_5e-5 5e-5 "$(ckpt_of "${BB}/lr_2e-4")"
A=$(best_val "${BB}/lr_2e-4"); C=$(best_val "${BB}/continue_lr_5e-5")
if python -c "import sys; sys.exit(0 if float('$C') >= float('$A') else 1)"; then SEL=continue_lr_5e-5; else SEL=lr_2e-4; fi
export BACKBONE_CKPT="$(ckpt_of "${BB}/${SEL}")"
echo "${BACKBONE_CKPT}" > "${BB}/SELECTED"
printf "# Stanford Cars backbone (strict validation split)\n\n| Run | Best val top-1 |\n|---|---|\n| lr_2e-4 | %s |\n| continue_lr_5e-5 | %s |\n\nSelected: **%s** -> \`%s\`\n" \
    "$A" "$C" "${SEL}" "${BACKBONE_CKPT}" > "${RESULTS_DIR}/backbone.md"
echo "[CARS] backbone: ${SEL} (val $A -> $C) ${BACKBONE_CKPT}"

# 2. Dense ProtoQuant + feature cache.
bash "${S}/run_phase1.sh"

# 3. One-shot edge masks, then code-level masks.
STAGES=masking bash "${S}/run_phase2.sh"
for SEED in ${SEEDS}; do
    PQ=$(newest "${P1_ROOT}/strict_s${SEED}/stage2/pipnet_*.pth")
    M="${P2_ROOT}/strict_s${SEED}/masking"
    [ -f "${M}/codeprune.json" ] && continue
    $PY/phase2_codeprune.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ}" \
        --cache_dir "${P1_ROOT}/feature_cache" --masking_dir "${M}" --seed "${SEED}" 2>&1 | tee "${M}/codeprune.log"
done

# 4. Code budget from code counts only (first seed): CP1 active codes closest to 68 % of B1_K10's.
FIRST=$(echo ${SEEDS} | cut -d' ' -f1)
CODE_KEY=$(python - "${P2_ROOT}/strict_s${FIRST}/masking" <<'PY'
import json, sys
from pathlib import Path
m = Path(sys.argv[1])
b1 = next(r for r in json.loads((m / "results.json").read_text())["models"] if r["method"] == "B1" and r["K"] == 10)
target = 0.68 * b1["active_codes"]
cp = [r for r in json.loads((m / "codeprune.json").read_text())["models"] if r["method"].startswith("CP1-M") and r["K"] == 10]
best = min(cp, key=lambda r: (abs(r["active_codes"] - target), -r["active_codes"]))
print(f"{best['method']}_K10")
print(f"[CARS] code budget: B1_K10 {b1['active_codes']} codes, target {target:.0f} -> {best['method']} "
      f"({best['active_codes']} codes)", file=sys.stderr)
PY
)
echo "${CODE_KEY}" > "${RESULTS_DIR}/code_budget.txt"

# 5. Head refit and the retraining recipe on cached features (minutes).
export BB2_RUNS_ROOT="${RUNS_ROOT}" BB2_RESULTS_DIR="${RESULTS_DIR}"
export REPAIR_KEYS="B1_K10 P1_K10 random_K10 ${CODE_KEY}"
CE_SWEEP="300 1000 3000 10000" bash "${S}/run_repair.sh"
CACHED_RETRAIN=1 bash "${S}/run_repair.sh"
unset REPAIR_KEYS

# 6. Full retraining on images, 145 epochs.
SEEDS="${RETRAIN_SEEDS}" MASK_KEY=B1_K10 EPOCHS=145 bash "${S}/run_budget_control.sh"
SEEDS="${RETRAIN_SEEDS}" MASKS_NAME=codeprune_masks.pt MASK_KEY="${CODE_KEY}" EPOCHS=145 bash "${S}/run_budget_control.sh"
echo "[CARS] done. Summaries: ${RESULTS_DIR}/phase1_baseline.md, ${RESULTS_DIR}/phase2_results.md"
