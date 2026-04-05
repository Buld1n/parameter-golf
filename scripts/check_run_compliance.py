#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FINAL_EXACT_RE = re.compile(
    r"(?P<label>final_[^\s]*_exact|legal_ttt_exact)\s+val_loss:(?P<val_loss>-?\d+(?:\.\d+)?)\s+val_bpb:(?P<val_bpb>-?\d+(?:\.\d+)?)"
)
SIZE_RE = re.compile(r"Total submission size [^:]*:\s*(?P<size>\d+)\s*bytes")
TRAIN_TIME_RE = re.compile(r"train_time:(?P<train_time>\d+)ms")
EVAL_TIME_RE = re.compile(r"eval_time:(?P<eval_time>\d+)ms")
MODEL_PARAMS_RE = re.compile(r"model_params:(?P<params>\d+)")


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    final_matches = list(FINAL_EXACT_RE.finditer(text))
    size_matches = list(SIZE_RE.finditer(text))
    train_time_matches = list(TRAIN_TIME_RE.finditer(text))
    eval_time_matches = list(EVAL_TIME_RE.finditer(text))
    model_params_match = MODEL_PARAMS_RE.search(text)

    preferred = None
    for match in final_matches:
        if match.group("label") == "final_int8_zlib_roundtrip_exact":
            preferred = match
    if preferred is None and final_matches:
        preferred = final_matches[-1]

    return {
        "final_label": preferred.group("label") if preferred is not None else None,
        "val_loss": float(preferred.group("val_loss")) if preferred is not None else None,
        "val_bpb": float(preferred.group("val_bpb")) if preferred is not None else None,
        "bytes_total": int(size_matches[-1].group("size")) if size_matches else None,
        "max_train_time_ms": max((int(m.group("train_time")) for m in train_time_matches), default=None),
        "eval_time_ms_total": sum(int(m.group("eval_time")) for m in eval_time_matches),
        "eval_measurements": len(eval_time_matches),
        "model_params": int(model_params_match.group("params")) if model_params_match else None,
    }


def fmt_num(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def fmt_float(value: float | None, precision: int = 6) -> str:
    return "-" if value is None else f"{value:.{precision}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a Parameter Golf run log fits the track constraints.")
    parser.add_argument("--log", required=True, help="Path to the run log.")
    parser.add_argument("--max-train-ms", type=int, default=600_000, help="Maximum allowed training time in ms.")
    parser.add_argument("--max-eval-ms", type=int, default=600_000, help="Maximum allowed summed eval time in ms.")
    parser.add_argument("--max-bytes", type=int, default=16_000_000, help="Maximum total submission size in bytes.")
    args = parser.parse_args()

    log_path = Path(args.log).resolve()
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")

    metrics = parse_log(log_path)

    print(f"log: {log_path}")
    print(f"model_params: {fmt_num(metrics['model_params'])}")
    print(f"final_metric: {metrics['final_label'] or '-'}")
    print(f"val_loss: {fmt_float(metrics['val_loss'], 8)}")
    print(f"val_bpb: {fmt_float(metrics['val_bpb'], 8)}")
    print(f"bytes_total: {fmt_num(metrics['bytes_total'])}")
    print(f"max_train_time_ms: {fmt_num(metrics['max_train_time_ms'])}")
    print(f"eval_time_ms_total: {fmt_num(metrics['eval_time_ms_total'])} (measurements={metrics['eval_measurements']})")

    failures: list[str] = []
    if metrics["bytes_total"] is None:
        failures.append("missing total submission size in log")
    elif metrics["bytes_total"] >= args.max_bytes:
        failures.append(f"submission size {metrics['bytes_total']} >= {args.max_bytes}")

    if metrics["max_train_time_ms"] is None:
        failures.append("missing train_time in log")
    elif metrics["max_train_time_ms"] > args.max_train_ms:
        failures.append(f"train_time {metrics['max_train_time_ms']}ms > {args.max_train_ms}ms")

    if metrics["eval_measurements"] == 0:
        failures.append("missing eval_time measurements in log")
    elif metrics["eval_time_ms_total"] > args.max_eval_ms:
        failures.append(f"eval_time total {metrics['eval_time_ms_total']}ms > {args.max_eval_ms}ms")

    if metrics["val_bpb"] is None:
        failures.append("missing final exact val_bpb in log")

    if failures:
        print("compliance: FAIL")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)

    print("compliance: PASS")


if __name__ == "__main__":
    main()
