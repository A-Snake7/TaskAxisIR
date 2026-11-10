#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/airnet/bin/python}"
WEIGHTS="${1:-checkpoints/promamba_legacy_epoch_0130.pth}"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="evaluations/${RUN_ID}"
REPORT_FILE="${RUN_DIR}/summary.txt"

mkdir -p "${RUN_DIR}"

{
  echo "ProMamba full-image evaluation"
  echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Weights: ${WEIGHTS}"
  echo
} > "${REPORT_FILE}"

run_task() {
  local task="$1"
  local input_dir="$2"
  local target_dir="$3"
  local task_dir="${RUN_DIR}/${task}"

  echo "Evaluating ${task}..."
  "${PYTHON_BIN}" test.py \
    --weights "${WEIGHTS}" \
    --input_dir "${input_dir}" \
    --target_dir "${target_dir}" \
    --task "${task}" \
    --save_dir "${task_dir}" \
    --amp | tee "${task_dir}.log"

  grep '^FINAL_METRICS' "${task_dir}.log" >> "${REPORT_FILE}"
}

run_task derain data/test/derain/input data/test/derain/target
run_task dehaze data/test/dehaze/input data/test/dehaze/target
run_task denoise data/test/denoise/input data/test/denoise/target

echo >> "${REPORT_FILE}"
echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')" >> "${REPORT_FILE}"
cat "${REPORT_FILE}"
