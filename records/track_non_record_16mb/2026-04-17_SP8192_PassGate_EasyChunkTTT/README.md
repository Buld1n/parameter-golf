# SP8192 Pass-Gated Recurrence + Easy-Chunk TTT

Локальный experimental package для следующего server-side sweep. Это не submission и не record claim; папка нужна как аккуратная точка переноса на новый pod.

## Что нового

- `TTT_EASY_CHUNK_RATIO`, `TTT_EASY_CHUNK_EPOCHS`
  Снижает число TTT-эпох на "лёгких" чанках вместо того, чтобы одинаково адаптировать весь eval stream.
- `TTT_OUTLIER_DROP_FRACTION`, `TTT_SCORE_WEIGHT_POWER`
  Делает selective TTT более аккуратным: можно выкинуть экстремально тяжёлые noisy windows и перевзвесить loss по score внутри уже выбранного hard band.
- `TTT_PARAM_MODE=recur_control`
  Разрешает eval-time updates только для control-параметров в recurrent band и skip path.
- `TTT_BLEND_BACK`, `TTT_RESET_MOMENTUM`
  Мягко тянет TTT-параметры обратно к базовым после каждого чанка и опционально сбрасывает momentum, чтобы резать drift.
- `ENABLE_LOOPING_AT_STEP`, `ENABLE_PARALLEL_RESIDUAL_AT_STEP`
  Переводит onset recurrence / parallel residual с wallclock-фракции на точный step trigger.
- `RECUR_ATTN_GATE`, `RECUR_ATTN_GATE_SCALE`
  Добавляет маленький per-pass attention output gate только в looped layers.

## Откуда взялась идея

- Из текущих open PR:
  - step-based onset как более чистый curriculum fix, без machine-specific wallclock confound
  - tiny attention gating как дешёвый способ вмешаться именно в recurrent attention path
- Из TTT-литературы:
  - high-loss / high-perplexity samples обычно полезнее для adaptation, чем uniform updates по всему потоку

## Первый прогон

Базовый безопасный вариант:

```bash
RUN_ID=sp8192_easychunk_s42 \
SEED=42 \
QK_GAIN_INIT=5.0 \
QK_GAIN_DEPTH_RAMP=0.5 \
PARALLEL_RESIDUAL_START=6 \
ENABLE_PARALLEL_RESIDUAL_AT_STEP=0 \
ENABLE_LOOPING_AT_STEP=2000 \
TTT_ENABLED=1 \
TTT_PARAM_MODE=full \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
TTT_EASY_CHUNK_RATIO=0.998 \
TTT_EASY_CHUNK_EPOCHS=1 \
TTT_OUTLIER_DROP_FRACTION=0.03 \
TTT_SCORE_WEIGHT_POWER=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Более агрессивный уникальный вариант:

```bash
RUN_ID=sp8192_passgate_easychunk_s42 \
SEED=42 \
QK_GAIN_INIT=5.0 \
QK_GAIN_DEPTH_RAMP=0.5 \
PARALLEL_RESIDUAL_START=6 \
ENABLE_PARALLEL_RESIDUAL_AT_STEP=0 \
ENABLE_LOOPING_AT_STEP=2000 \
RECUR_ATTN_GATE=1 \
RECUR_ATTN_GATE_SCALE=0.5 \
TTT_ENABLED=1 \
TTT_PARAM_MODE=full \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
TTT_EASY_CHUNK_RATIO=0.998 \
TTT_EASY_CHUNK_EPOCHS=1 \
TTT_OUTLIER_DROP_FRACTION=0.03 \
TTT_SCORE_WEIGHT_POWER=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Если full-TTT начнёт drift'ить:

```bash
TTT_PARAM_MODE=recur_control \
TTT_BLEND_BACK=0.05 \
TTT_RESET_MOMENTUM=1
```
