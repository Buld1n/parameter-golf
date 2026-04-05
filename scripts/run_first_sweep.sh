#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   NPROC=1 ./scripts/run_first_sweep.sh
#
# Optional overrides:
#   DATA_PATH=...
#   TOKENIZER_PATH=...
#   TRAIN_SHARDS experiments are controlled by your dataset download, not this script.

NPROC="${NPROC:-1}"
TORCHRUN_CMD="${TORCHRUN_CMD:-torchrun --standalone --nproc_per_node=${NPROC}}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_gpt.py}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"

python3 - <<'PY'
import importlib.util
missing = [m for m in ("torch", "sentencepiece", "numpy") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {', '.join(missing)}")
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. run_first_sweep.sh targets train_gpt.py (CUDA).")
PY

if [ ! -f "${TOKENIZER_PATH}" ]; then
  echo "Tokenizer not found: ${TOKENIZER_PATH}" >&2
  exit 1
fi

if ! ls "${DATA_PATH}"/fineweb_train_*.bin >/dev/null 2>&1; then
  echo "No train shards found at ${DATA_PATH}/fineweb_train_*.bin" >&2
  exit 1
fi

if ! ls "${DATA_PATH}"/fineweb_val_*.bin >/dev/null 2>&1; then
  echo "No val shards found at ${DATA_PATH}/fineweb_val_*.bin" >&2
  exit 1
fi

run_one() {
  local run_id="$1"
  shift
  echo "=== Running ${run_id} ==="
  RUN_ID="${run_id}" \
  "$@" \
  ${TORCHRUN_CMD} "${TRAIN_SCRIPT}"
}

run_one ab_base \
  env \
  DATA_PATH="${DATA_PATH}" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" \
  MLP_ACT=relu2 \
  MLP_MULT=2.0 \
  ROPE_DIMS=0 \
  XSA_LAST_N=0 \
  LN_SCALE=0 \
  EVAL_SLIDING=0 \
  EMA_ENABLED=0 \
  SWA_ENABLED=0

run_one ab_lrelu2_xsa \
  env \
  DATA_PATH="${DATA_PATH}" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" \
  MLP_ACT=lrelu2 \
  MLP_LEAKY_RELU_SLOPE=0.5 \
  MLP_MULT=2.0 \
  ROPE_DIMS=16 \
  XSA_LAST_N=4 \
  LN_SCALE=1 \
  EVAL_SLIDING=0 \
  EMA_ENABLED=0 \
  SWA_ENABLED=0

run_one ab_swiglu \
  env \
  DATA_PATH="${DATA_PATH}" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" \
  MLP_ACT=swiglu \
  MLP_MULT=2.0 \
  SWIGLU_PARAM_MATCH=1 \
  ROPE_DIMS=16 \
  XSA_LAST_N=4 \
  LN_SCALE=1 \
  EVAL_SLIDING=0 \
  EMA_ENABLED=0 \
  SWA_ENABLED=0

run_one ab_lrelu2_xsa_ema_sliding \
  env \
  DATA_PATH="${DATA_PATH}" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" \
  MLP_ACT=lrelu2 \
  MLP_LEAKY_RELU_SLOPE=0.5 \
  MLP_MULT=2.0 \
  ROPE_DIMS=16 \
  XSA_LAST_N=4 \
  LN_SCALE=1 \
  EVAL_SLIDING=1 \
  EVAL_STRIDE=64 \
  EMA_ENABLED=1 \
  EMA_DECAY=0.997 \
  EMA_START_STEP=0 \
  SWA_ENABLED=0

echo
echo "Sweep complete. Summary:"
python3 scripts/summarize_runs.py --logs-dir logs --sort val_bpb
