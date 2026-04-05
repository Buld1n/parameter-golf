#!/usr/bin/env bash
set -euo pipefail

EXPECTED_GPUS="${EXPECTED_GPUS:-8}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export EXPECTED_GPUS DATA_PATH TOKENIZER_PATH

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 1
fi

python3 - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

import torch

expected_gpus = int(os.environ["EXPECTED_GPUS"])
data_path = Path(os.environ["DATA_PATH"])
tokenizer_path = Path(os.environ["TOKENIZER_PATH"])

required = {
    "numpy": "numpy",
    "sentencepiece": "sentencepiece",
    "datasets": "datasets",
    "huggingface_hub": "huggingface-hub",
    "zstandard": "zstandard",
}
missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {', '.join(missing)}")

if importlib.util.find_spec("flash_attn_interface") is None and importlib.util.find_spec("flash_attn") is None:
    raise SystemExit("FlashAttention 3 is missing (need flash_attn_interface or flash_attn).")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available.")
count = torch.cuda.device_count()
if count != expected_gpus:
    raise SystemExit(f"Expected {expected_gpus} CUDA devices, found {count}.")

names = []
for i in range(count):
    props = torch.cuda.get_device_properties(i)
    names.append(props.name)
    print(f"gpu[{i}] name={props.name} mem_gb={props.total_memory / 1024**3:.1f}")

if any("H100" not in name for name in names):
    raise SystemExit(f"Non-H100 device detected: {names}")

if not tokenizer_path.exists():
    raise SystemExit(f"Tokenizer not found: {tokenizer_path}")

train_files = sorted(data_path.glob("fineweb_train_*.bin"))
val_files = sorted(data_path.glob("fineweb_val_*.bin"))
if not train_files:
    raise SystemExit(f"No train shards found in {data_path}")
if not val_files:
    raise SystemExit(f"No val shards found in {data_path}")

print(f"train_shards={len(train_files)}")
print(f"val_shards={len(val_files)}")
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
PY

echo
echo "nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo
echo "nvidia-smi topo -m"
nvidia-smi topo -m || true
