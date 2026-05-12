#!/usr/bin/env python3
"""Audit or repair a NeuroGolf submission zip with official-style ONNX processing checks."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import onnx

import run_remote


def load_zip_models(input_zip: Path, tmp: Path) -> Path:
    out = tmp / "models"
    out.mkdir()
    with zipfile.ZipFile(input_zip) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if name.startswith("task") and name.endswith(".onnx"):
                (out / name).write_bytes(zf.read(info.filename))
    return out


def write_zip(model_dir: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for tid in range(1, run_remote.TASK_COUNT + 1):
            p = model_dir / f"task{tid:03d}.onnx"
            if not p.exists():
                raise SystemExit(f"Missing {p.name}")
            zf.write(p, p.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_zip", type=Path)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--output-zip", type=Path, default=None)
    ap.add_argument("--include-arcgen", action="store_true")
    ap.add_argument("--repair", action="store_true", help="Replace processing-failing models with safe identity models")
    args = ap.parse_args()

    tasks = run_remote.load_tasks(args.data_dir, None)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        model_dir = load_zip_models(args.input_zip, tmp)
        fallback_dir = tmp / "fallback"
        fallback_dir.mkdir()
        failures = []

        for tid in range(1, run_remote.TASK_COUNT + 1):
            model_path = model_dir / f"task{tid:03d}.onnx"
            if not model_path.exists():
                failures.append({"task": tid, "reason": "missing model"})
                continue
            ok, reason = run_remote.official_process_ok(model_path, tasks[tid], args.include_arcgen)
            if ok:
                continue
            failures.append({"task": tid, "reason": reason})
            if args.repair:
                fallback = fallback_dir / f"task{tid:03d}.onnx"
                run_remote.make_identity_model(fallback)
                ok2, reason2 = run_remote.official_process_ok(fallback, tasks[tid], args.include_arcgen)
                if not ok2:
                    raise SystemExit(f"Identity fallback failed for task{tid:03d}: {reason2}")
                model_path.write_bytes(fallback.read_bytes())

        print(json.dumps({"failure_count": len(failures), "failures": failures}, indent=2))
        if failures and not args.repair:
            raise SystemExit(2)
        if args.output_zip:
            write_zip(model_dir, args.output_zip)
            print(f"Wrote {args.output_zip}")


if __name__ == "__main__":
    main()
