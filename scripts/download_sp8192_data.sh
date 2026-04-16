#!/usr/bin/env bash
set -euo pipefail

# The current SP8192 record line uses the matched FineWeb export from Kevin Clark.
# Override MATCHED_FINEWEB_REPO_ID if you want to test a different compatible export.
export MATCHED_FINEWEB_REPO_ID="${MATCHED_FINEWEB_REPO_ID:-kevclark/parameter-golf}"
TRAIN_SHARDS="${TRAIN_SHARDS:-128}"

python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards "${TRAIN_SHARDS}"
