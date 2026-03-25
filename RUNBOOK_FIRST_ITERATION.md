# Parameter Golf: First Iteration Runbook

## Objective

Optimize `val_bpb` while satisfying:
- `bytes_total < 16_000_000`
- reproducible training under the 10-minute track budget on `8xH100`

## Experiment Architecture

Single training entrypoint:
- `train_gpt.py` for train + evaluation + artifact size checks

A/B iteration loop:
1. run baseline
2. change one variable
3. compare `val_bpb`, size, wallclock
4. keep only statistically promising configs

## Endpoints (Done Criteria)

For each run, collect:
- `final_int8_zlib_roundtrip_exact val_bpb`
- `Total submission size int8+zlib`
- `train_time` and step throughput

Promotion gates:
- must stay under `16_000_000` bytes
- must not regress reproducibility
- for record track: must beat current SOTA with required significance

## New Model Knobs (implemented in `train_gpt.py`)

- `MLP_ACT`:
  - `relu2` (default, baseline-compatible)
  - `lrelu2`
  - `gelu`
  - `swiglu`
- `MLP_MULT` now accepts float (finer width control)
- `MLP_LEAKY_RELU_SLOPE` (default `0.5`, used for `lrelu2`)
- `SWIGLU_PARAM_MATCH` (`1` default): keeps SwiGLU hidden width near relu2 parameter budget
- `ROPE_DIMS` (`0` default): partial RoPE when set to an even value in `(0, head_dim]`
- `XSA_LAST_N` (`0` default): enables XSA on the last `N` transformer layers
- `LN_SCALE` (`0` default): depth-aware residual scale init (`1/sqrt(layer_idx+1)`)
- `EVAL_SLIDING`, `EVAL_STRIDE`: sliding-window validation mode for better BPB scoring
- `EMA_ENABLED`, `EMA_DECAY`, `EMA_START_STEP`, `EMA_EVAL_AFTER_APPLY`

## Starter Commands

Baseline:

```bash
RUN_ID=ab_base \
MLP_ACT=relu2 \
MLP_MULT=2.0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

LeakyReLU^2:

```bash
RUN_ID=ab_lrelu2 \
MLP_ACT=lrelu2 \
MLP_LEAKY_RELU_SLOPE=0.5 \
MLP_MULT=2.0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

SwiGLU:

```bash
RUN_ID=ab_swiglu \
MLP_ACT=swiglu \
MLP_MULT=2.0 \
SWIGLU_PARAM_MATCH=1 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Suggested Next Experiments

1. Sweep `MLP_ACT in {relu2, lrelu2, swiglu}` with fixed training budget.
2. Sweep `XSA_LAST_N in {0, 2, 4}` and `ROPE_DIMS in {0, 16}` for the best activation.
3. Sweep `MLP_MULT` around `1.6..2.4` by `0.1` for the best stack.
4. Re-check artifact size on every winner before longer runs.

## Helper Scripts

- `scripts/setup_env.sh`: creates `.venv` and installs `requirements.txt`.
- `scripts/run_first_sweep.sh`: runs a first A/B sweep (baseline, activation variants, EMA).
- `scripts/summarize_runs.py`: parses `logs/*.txt` and prints sortable metrics table.
- `scripts/init_submission.py`: scaffolds a `records/...` submission folder from a run log.

Example:

```bash
./scripts/setup_env.sh
source .venv/bin/activate
NPROC=1 ./scripts/run_first_sweep.sh
python3 scripts/summarize_runs.py --logs-dir logs --sort val_bpb
python3 scripts/init_submission.py \
  --track track_non_record_16mb \
  --name "My Run" \
  --author "Your Name" \
  --github-id your_github \
  --log logs/my_run.txt
```
