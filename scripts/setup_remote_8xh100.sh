#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATA_VARIANT="${DATA_VARIANT:-sp1024}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
TRAIN_SHARDS="${TRAIN_SHARDS:-}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    raise SystemExit(
        "torch is missing. Use a CUDA/PyTorch image first; this script only installs lightweight extras."
    )
print(f"python={sys.version.split()[0]}")
PY

"${PYTHON}" -m pip install --upgrade pip setuptools wheel

"${PYTHON}" - <<'PY'
import importlib.util
import subprocess
import sys

packages = {
    "numpy": "numpy",
    "sentencepiece": "sentencepiece",
    "datasets": "datasets",
    "huggingface_hub": "huggingface-hub",
    "tqdm": "tqdm",
    "tiktoken": "tiktoken",
    "zstandard": "zstandard",
}
missing = [pkg for mod, pkg in packages.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY

if ! "${PYTHON}" -c "import flash_attn_interface" >/dev/null 2>&1 && ! "${PYTHON}" -c "import flash_attn" >/dev/null 2>&1; then
  "${PYTHON}" -m pip install --no-cache-dir "${FLASH_ATTN_WHEEL}"
fi

if [ ! -f "${TOKENIZER_PATH}" ] || ! ls "${DATA_PATH}"/fineweb_train_*.bin >/dev/null 2>&1 || ! ls "${DATA_PATH}"/fineweb_val_*.bin >/dev/null 2>&1; then
  if [ -n "${TRAIN_SHARDS}" ]; then
    "${PYTHON}" data/cached_challenge_fineweb.py --variant "${DATA_VARIANT}" --train-shards "${TRAIN_SHARDS}"
  else
    "${PYTHON}" data/cached_challenge_fineweb.py --variant "${DATA_VARIANT}"
  fi
fi

EXPECTED_GPUS="${EXPECTED_GPUS:-8}" DATA_PATH="${DATA_PATH}" TOKENIZER_PATH="${TOKENIZER_PATH}" \
  bash ./scripts/preflight_8xh100.sh
