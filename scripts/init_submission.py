#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

FINAL_EXACT_RE = re.compile(
    r"final_[^\s]*_exact\s+val_loss:(?P<val_loss>-?\d+(?:\.\d+)?)\s+val_bpb:(?P<val_bpb>-?\d+(?:\.\d+)?)"
)
SIZE_RE = re.compile(r"Total submission size [^:]*:\s*(?P<size>\d+)\s*bytes")
CODE_SIZE_RE = re.compile(r"Code size:\s*(?P<size>\d+)\s*bytes")


def slugify(s: str) -> str:
    out = []
    for ch in s.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "run"


def parse_metrics(log_path: Path) -> dict[str, float | int | None]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    final_matches = list(FINAL_EXACT_RE.finditer(text))
    size_matches = list(SIZE_RE.finditer(text))
    code_size_matches = list(CODE_SIZE_RE.finditer(text))

    return {
        "val_loss": float(final_matches[-1].group("val_loss")) if final_matches else None,
        "val_bpb": float(final_matches[-1].group("val_bpb")) if final_matches else None,
        "bytes_total": int(size_matches[-1].group("size")) if size_matches else None,
        "bytes_code": int(code_size_matches[-1].group("size")) if code_size_matches else None,
    }


def write_readme(path: Path, run_name: str, track: str, metrics: dict[str, float | int | None], log_file: str) -> None:
    lines = [
        f"# {run_name}",
        "",
        f"Track: `{track}`",
        "",
        "## Summary",
        "- Fill in model/optimizer/tokenizer changes.",
        "- Add exact command used to run training.",
        "",
        "## Metrics",
        f"- `val_loss`: `{metrics['val_loss']}`",
        f"- `val_bpb`: `{metrics['val_bpb']}`",
        f"- `bytes_total`: `{metrics['bytes_total']}`",
        f"- `bytes_code`: `{metrics['bytes_code']}`",
        "",
        "## Included Files",
        "- `train_gpt.py`",
        f"- `{log_file}`",
        "- `submission.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a records submission folder from a run log.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to repo root (default: current directory).",
    )
    parser.add_argument(
        "--track",
        choices=["track_10min_16mb", "track_non_record_16mb"],
        required=True,
        help="Target records track directory.",
    )
    parser.add_argument("--name", required=True, help="Human-readable run name.")
    parser.add_argument("--author", required=True, help="Author name for submission.json.")
    parser.add_argument("--github-id", required=True, help="GitHub username for submission.json.")
    parser.add_argument("--log", required=True, help="Path to training log file.")
    parser.add_argument(
        "--script",
        default="train_gpt.py",
        help="Training script to snapshot into the submission folder (default: train_gpt.py).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print output path without creating files.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    log_path = Path(args.log).resolve()
    script_path = (repo_root / args.script).resolve()

    if not repo_root.exists():
        raise SystemExit(f"Repo root not found: {repo_root}")
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")
    if not script_path.exists():
        raise SystemExit(f"Script file not found: {script_path}")

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder_name = f"{run_date}_{slugify(args.name)}"
    out_dir = repo_root / "records" / args.track / folder_name
    if args.dry_run:
        metrics = parse_metrics(log_path)
        print(f"output_dir={out_dir}")
        print(f"val_bpb={metrics['val_bpb']} bytes_total={metrics['bytes_total']} bytes_code={metrics['bytes_code']}")
        return
    out_dir.mkdir(parents=True, exist_ok=False)

    log_target = out_dir / log_path.name
    script_target = out_dir / "train_gpt.py"
    shutil.copy2(log_path, log_target)
    shutil.copy2(script_path, script_target)

    metrics = parse_metrics(log_path)
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    submission = {
        "author": args.author,
        "github_id": args.github_id,
        "name": args.name,
        "blurb": "TODO: add concise summary of approach and setup.",
        "date": now_iso,
        "val_loss": metrics["val_loss"],
        "val_bpb": metrics["val_bpb"],
        "bytes_total": metrics["bytes_total"],
        "bytes_code": metrics["bytes_code"],
    }
    (out_dir / "submission.json").write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
    write_readme(out_dir / "README.md", args.name, args.track, metrics, log_target.name)

    print(out_dir)


if __name__ == "__main__":
    main()
