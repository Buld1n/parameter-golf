#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

FINAL_EXACT_RE = re.compile(
    r"final_[^\s]*_exact\s+val_loss:(?P<val_loss>-?\d+(?:\.\d+)?)\s+val_bpb:(?P<val_bpb>-?\d+(?:\.\d+)?)"
)
SIZE_RE = re.compile(r"Total submission size [^:]*:\s*(?P<size>\d+)\s*bytes")
TRAIN_TIME_RE = re.compile(r"train_time:(?P<train_time>\d+)ms")
PARAMS_RE = re.compile(r"model_params:(?P<params>\d+)")


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    final_matches = list(FINAL_EXACT_RE.finditer(text))
    size_matches = list(SIZE_RE.finditer(text))
    time_matches = list(TRAIN_TIME_RE.finditer(text))
    params_match = PARAMS_RE.search(text)

    val_loss = float(final_matches[-1].group("val_loss")) if final_matches else None
    val_bpb = float(final_matches[-1].group("val_bpb")) if final_matches else None
    bytes_total = int(size_matches[-1].group("size")) if size_matches else None
    train_time_ms = int(time_matches[-1].group("train_time")) if time_matches else None
    model_params = int(params_match.group("params")) if params_match else None

    return {
        "run": path.stem,
        "path": str(path),
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "bytes_total": bytes_total,
        "train_time_ms": train_time_ms,
        "model_params": model_params,
    }


def fmt_float(value: float | None, precision: int = 6) -> str:
    return "-" if value is None else f"{value:.{precision}f}"


def fmt_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Parameter Golf run logs.")
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory containing run logs (default: logs).",
    )
    parser.add_argument(
        "--glob",
        default="*.txt",
        help="Glob pattern inside --logs-dir (default: *.txt).",
    )
    parser.add_argument(
        "--sort",
        choices=["val_bpb", "bytes_total", "train_time_ms", "run"],
        default="val_bpb",
        help="Sort key (default: val_bpb).",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        raise SystemExit(f"Logs directory not found: {logs_dir}")

    rows = [parse_log(p) for p in sorted(logs_dir.glob(args.glob))]
    if not rows:
        raise SystemExit(f"No logs matched {logs_dir / args.glob}")

    missing_high = float("inf")
    if args.sort == "run":
        rows.sort(key=lambda r: str(r["run"]))
    else:
        rows.sort(key=lambda r: r[args.sort] if r[args.sort] is not None else missing_high)

    header = (
        f"{'run':<30} {'val_bpb':>10} {'val_loss':>10} "
        f"{'bytes_total':>12} {'train_ms':>10} {'params':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{str(r['run'])[:30]:<30} "
            f"{fmt_float(r['val_bpb'], 6):>10} "
            f"{fmt_float(r['val_loss'], 6):>10} "
            f"{fmt_int(r['bytes_total']):>12} "
            f"{fmt_int(r['train_time_ms']):>10} "
            f"{fmt_int(r['model_params']):>12}"
        )


if __name__ == "__main__":
    main()
