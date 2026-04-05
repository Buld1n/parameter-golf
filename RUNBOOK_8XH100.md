# 8xH100 Track Runbook

## Goal

Execute one technically compliant `track_10min_16mb` run on `8xH100` with:

- `MAX_WALLCLOCK_SECONDS <= 599`
- `Total submission size < 16_000_000` bytes
- no train-time access to validation labels beyond the allowed evaluation path already encoded in `train_gpt.py`

This runbook is optimized for a short remote session where setup mistakes are more dangerous than missing a tiny amount of model quality.

## Important Constraints

- The challenge track is defined around `8xH100` and a 10-minute training cap.
- Record submission quality is a separate bar: beating SOTA requires enough seeds and significance. With a one-hour budget, the realistic target is one clean compliant run first.
- Prefer `MAX_WALLCLOCK_SECONDS=599` over `600` to leave a small safety margin.

## Files

- `scripts/setup_remote_8xh100.sh`: install lightweight missing deps, install FlashAttention 3 wheel if needed, and fetch the dataset if missing.
- `scripts/preflight_8xh100.sh`: verify 8 H100 GPUs, Python deps, tokenizer, dataset, and print topology.
- `scripts/run_8xh100_smoke.sh`: short compile/integration smoke run.
- `scripts/run_8xh100_track.sh`: full 8xH100 track run with explicit env.
- `scripts/check_run_compliance.py`: verify training time, eval time, final metric presence, and artifact size from the produced log.

## Recommended Remote Sequence

1. Clone the fork onto local NVMe on the remote box.
2. Run `bash scripts/setup_remote_8xh100.sh`.
3. Run `bash scripts/preflight_8xh100.sh`.
4. Run `bash scripts/run_8xh100_smoke.sh`.
5. If smoke is clean, run `bash scripts/run_8xh100_track.sh`.
6. Re-run `python3 scripts/check_run_compliance.py --log logs/<run_id>.txt` on the final log and archive the log immediately.

## Smoke Run

Use the smoke run to catch:

- missing `flash_attn_interface`
- NCCL or topology issues
- OOM at full batch
- code paths that only break after quantization / final eval

The smoke run intentionally disables EMA, SWA, and sliding eval to minimize wasted time while still exercising the full export path.

## Full Track Run

The full run is pinned to the current tuned stack and makes all important knobs explicit instead of trusting code defaults:

- 11 layers, `512d`, `8` heads, `4` KV heads
- `BIGRAM_VOCAB_SIZE=2048`
- `XSA_LAST_N=4`
- `ROPE_DIMS=16`
- `LN_SCALE=1`
- `VE_ENABLED=1`
- `EMA_ENABLED=1`, `SWA_ENABLED=1`
- `LATE_QAT_THRESHOLD=0.15`
- `EVAL_SLIDING=1`, `EVAL_STRIDE=64`
- `MAX_WALLCLOCK_SECONDS=599`

`MLP_ACT` defaults to `lrelu2` in the launch script, but it is deliberately left overrideable from the shell:

```bash
MLP_ACT=relu2 bash scripts/run_8xh100_track.sh
```

Use that fallback if the `lrelu2` branch looks suspicious on the target machine.

## Budget Strategy

For a tight one-hour window, the safest order is:

1. one preflight
2. one short smoke
3. one full compliant run

Avoid multi-seed reruns unless the first full run is clearly strong and there is confirmed budget/time left.

## After the Run

The final log should contain:

- `final_int8_zlib_roundtrip_exact`
- `Total submission size ...`
- `train_time:...ms`

If those are present, run:

```bash
python3 scripts/check_run_compliance.py --log logs/<run_id>.txt
```

If it passes, the run is technically within the main track limits for time and artifact size. That still does not make it a statistically valid record submission by itself.
