#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-smoke_8xh100}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"

mkdir -p logs

export RUN_ID
export DATA_PATH
export TOKENIZER_PATH
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export NUM_LAYERS="${NUM_LAYERS:-11}"
export MODEL_DIM="${MODEL_DIM:-512}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export ITERATIONS="${ITERATIONS:-9000}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-45}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export MLP_ACT="${MLP_ACT:-relu2}"
export MLP_MULT="${MLP_MULT:-3.0}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}"
export XSA_LAST_N="${XSA_LAST_N:-4}"
export ROPE_DIMS="${ROPE_DIMS:-16}"
export LN_SCALE="${LN_SCALE:-1}"
export VE_ENABLED="${VE_ENABLED:-1}"
export VE_DIM="${VE_DIM:-128}"
export VE_LAYERS="${VE_LAYERS:-9,10}"
export EMA_ENABLED="${EMA_ENABLED:-0}"
export SWA_ENABLED="${SWA_ENABLED:-0}"
export EVAL_SLIDING="${EVAL_SLIDING:-0}"
export EVAL_STRIDE="${EVAL_STRIDE:-64}"
export LATE_QAT="${LATE_QAT:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

torchrun --standalone --nproc_per_node=8 train_gpt.py
python3 scripts/check_run_compliance.py --log "logs/${RUN_ID}.txt"
