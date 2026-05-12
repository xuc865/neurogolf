#!/usr/bin/env python3
"""
Remote NeuroGolf ensemble runner.

This script is intentionally self-contained: it can run your existing models,
public solver repos, validate candidates against available train/test/arc-gen
examples, pick the best valid model per task, package a final zip, and optionally
submit through the Kaggle CLI.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


COMPETITION = "neurogolf-2026"
TASK_COUNT = 400
CHANNELS = 10
HEIGHT = 30
WIDTH = 30
GRID_SHAPE = (1, CHANNELS, HEIGHT, WIDTH)
BANNED_OPS = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
FILESIZE_LIMIT = int(1.44 * 1024 * 1024)


@dataclass
class Candidate:
    source: str
    path: Path
    score: float
    cost: float
    memory: int
    params: int
    valid_examples: int


@dataclass(frozen=True)
class PreparedExample:
    inp: np.ndarray
    exp: np.ndarray


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_tasks(data_dir: Path | None, data_file: Path | None) -> dict[int, dict[str, Any]]:
    if data_file:
        raw = json.loads(data_file.read_text())
        if all(str(k).isdigit() for k in raw.keys()):
            return {int(k): v for k, v in raw.items()}
        return {i: v for i, (_, v) in enumerate(sorted(raw.items()), 1)}

    if not data_dir:
        raise SystemExit("Provide --data-dir or --data-file.")

    tasks: dict[int, dict[str, Any]] = {}
    for tid in range(1, TASK_COUNT + 1):
        p = data_dir / f"task{tid:03d}.json"
        if not p.exists():
            raise SystemExit(f"Missing {p}")
        tasks[tid] = json.loads(p.read_text())
    return tasks


def write_all_tasks_json(tasks: dict[int, dict[str, Any]], path: Path) -> Path:
    payload = {f"task{tid:03d}": tasks[tid] for tid in range(1, TASK_COUNT + 1)}
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path


def to_onehot(grid: list[list[int]]) -> np.ndarray | None:
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h > HEIGHT or w > WIDTH:
        return None
    arr = np.zeros(GRID_SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if 0 <= int(color) < CHANNELS:
                arr[0, int(color), r, c] = 1.0
            else:
                return None
    return arr


def examples_for(task: dict[str, Any], include_arcgen: bool) -> list[dict[str, Any]]:
    examples = list(task.get("train", [])) + list(task.get("test", []))
    if include_arcgen:
        examples += list(task.get("arc-gen", []))
    return examples


def prepare_examples(task: dict[str, Any], include_arcgen: bool) -> list[PreparedExample]:
    prepared: list[PreparedExample] = []
    for ex in examples_for(task, include_arcgen):
        inp = to_onehot(ex["input"])
        exp = to_onehot(ex["output"])
        if inp is not None and exp is not None:
            prepared.append(PreparedExample(inp, exp))
    return prepared


def sanitize_for_official_processing(model: onnx.ModelProto) -> onnx.ModelProto | None:
    model = onnx.ModelProto().FromString(model.SerializeToString())
    for node in model.graph.node:
        if not node.output or not node.output[0]:
            return None
        node.name = node.output[0]
        if "kernel_time" in node.name:
            return None
    return model


def make_session_options(enable_profiling: bool = False) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if enable_profiling:
        options.enable_profiling = True
        prefix = Path(tempfile.gettempdir()) / f"ng_profile_{os.getpid()}_{time.time_ns()}"
        options.profile_file_prefix = str(prefix)
    return options


def validate_model(model_path: Path, task: dict[str, Any], include_arcgen: bool) -> int:
    model = onnx.load(str(model_path))
    if model_path.stat().st_size > FILESIZE_LIMIT:
        return 0
    if model.functions:
        return 0
    for node in model.graph.node:
        if node.op_type.upper() in {op.upper() for op in BANNED_OPS}:
            return 0
        if "kernel_time" in node.name:
            return 0
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                return 0

    options = make_session_options()
    sess = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    ok = 0
    for ex in examples_for(task, include_arcgen):
        inp = to_onehot(ex["input"])
        exp = to_onehot(ex["output"])
        if inp is None or exp is None:
            continue
        out = sess.run(["output"], {"input": inp})[0]
        pred = (out > 0.0).astype(np.float32)
        if not np.array_equal(pred, exp):
            return 0
        ok += 1
    return ok


def run_examples_with_profile(
    model: onnx.ModelProto,
    task: dict[str, Any],
    include_arcgen: bool,
    require_correct: bool,
) -> str | None:
    return run_prepared_examples_with_profile(
        model,
        prepare_examples(task, include_arcgen),
        require_correct=require_correct,
    )


def run_prepared_examples_with_profile(
    model: onnx.ModelProto,
    examples: list[PreparedExample],
    require_correct: bool,
) -> str | None:
    options = make_session_options(enable_profiling=True)
    sess = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    ran = False
    for ex in examples:
        out = sess.run(["output"], {"input": ex.inp})[0]
        ran = True
        if require_correct and not np.array_equal((out > 0.0).astype(np.float32), ex.exp):
            sess.end_profiling()
            return None
    return sess.end_profiling() if ran else None


def official_memory(model: onnx.ModelProto, trace_path: str) -> int | None:
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    if len(graph.input) > 1 or len(graph.output) > 1:
        return None
    init_names = {init.name for init in graph.initializer}
    init_names.update(init.name for init in graph.sparse_initializer)
    io_names = {t.name for t in itertools.chain(graph.input, graph.output)}
    if io_names.intersection(init_names):
        return None
    if model.functions:
        return None
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            return None

    node_outputs: dict[str, list[str]] = {}
    tensor_names: set[str] = set()
    for node in graph.node:
        if node.op_type.upper() in {op.upper() for op in BANNED_OPS}:
            return None
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                return None
        if "kernel_time" in node.name:
            return None
        node_outputs[node.name] = list(node.output)
        for output_name in node.output:
            if output_name:
                tensor_names.add(output_name)

    tensor_memory: dict[str, int] = {}
    tensor_dtypes: dict[str, np.dtype] = {}
    tensor_map = {t.name: t for t in itertools.chain(graph.input, graph.value_info, graph.output)}
    tensor_names.update(tensor_map.keys())
    for tensor_name in tensor_names:
        item = tensor_map.get(tensor_name)
        if not item:
            return None
        if item.type.HasField("sequence_type"):
            return None
        if not item.type.HasField("tensor_type"):
            continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"):
            return None
        num_elements = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
                return None
            num_elements *= dim.dim_value
        if tensor_name in ["input", "output"]:
            continue
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        tensor_memory[tensor_name] = num_elements * np.dtype(np_dtype).itemsize
        tensor_dtypes[tensor_name] = np.dtype(np_dtype)

    with open(trace_path, "r") as f:
        trace_data = json.load(f)
    for event in trace_data:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        if "output_type_shape" not in event["args"]:
            continue
        node_name = event.get("name", "").replace("_kernel_time", "")
        if node_name not in node_outputs:
            continue
        for i, shape_dict in enumerate(event["args"]["output_type_shape"]):
            if i >= len(node_outputs[node_name]):
                continue
            output_name = node_outputs[node_name][i]
            if output_name not in tensor_dtypes:
                continue
            itemsize = np.dtype(tensor_dtypes[output_name]).itemsize
            mem = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            tensor_memory[output_name] = max(tensor_memory[output_name], mem)
    return sum(tensor_memory.values())


def tensor_size_bytes(value_info: onnx.ValueInfoProto) -> int | None:
    tt = value_info.type.tensor_type
    if not tt.HasField("elem_type"):
        return None
    sizes = {
        onnx.TensorProto.FLOAT: 4,
        onnx.TensorProto.UINT8: 1,
        onnx.TensorProto.INT8: 1,
        onnx.TensorProto.UINT16: 2,
        onnx.TensorProto.INT16: 2,
        onnx.TensorProto.INT32: 4,
        onnx.TensorProto.INT64: 8,
        onnx.TensorProto.BOOL: 1,
        onnx.TensorProto.FLOAT16: 2,
        onnx.TensorProto.DOUBLE: 8,
        onnx.TensorProto.UINT32: 4,
        onnx.TensorProto.UINT64: 8,
    }
    item = sizes.get(tt.elem_type)
    if item is None:
        return None
    n = 1
    for dim in tt.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        n *= int(dim.dim_value)
    return n * item


def approximate_memory(model: onnx.ModelProto) -> int:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    total = 0
    for vi in values:
        size = tensor_size_bytes(vi)
        if size is None:
            return 10**12
        total += size
    return total


def count_params(model: onnx.ModelProto) -> int:
    params = 0
    for init in model.graph.initializer:
        dims = list(init.dims)
        if not dims or any(d <= 0 for d in dims):
            return 10**12
        params += math.prod(dims)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                dims = list(attr.t.dims)
                params += math.prod(dims) if dims else 1
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return int(params)


def model_score(model_path: Path) -> tuple[float, float, int, int]:
    model = onnx.load(str(model_path))
    memory = approximate_memory(model)
    params = count_params(model)
    cost = max(1.0, float(memory + params))
    return max(1.0, 25.0 - math.log(cost)), cost, memory, params


def official_score(model_path: Path, task: dict[str, Any], include_arcgen: bool) -> tuple[float, float, int, int] | None:
    model = sanitize_for_official_processing(onnx.load(str(model_path)))
    if model is None:
        return None
    trace = run_examples_with_profile(model, task, include_arcgen, require_correct=True)
    if not trace:
        return None
    try:
        memory = official_memory(model, trace)
    finally:
        try:
            Path(trace).unlink()
        except OSError:
            pass
    if memory is None or memory < 0:
        return None
    params = count_params(model)
    if params < 0 or params >= 10**12:
        return None
    cost = max(1.0, float(memory + params))
    return max(1.0, 25.0 - math.log(cost)), cost, memory, params


def score_sanitized_model(
    model: onnx.ModelProto,
    examples: list[PreparedExample],
    require_correct: bool,
) -> tuple[float, float, int, int, int] | None:
    trace = run_prepared_examples_with_profile(model, examples, require_correct=require_correct)
    if not trace:
        return None
    try:
        memory = official_memory(model, trace)
    finally:
        try:
            Path(trace).unlink()
        except OSError:
            pass
    if memory is None or memory < 0:
        return None
    params = count_params(model)
    if params < 0 or params >= 10**12:
        return None
    cost = max(1.0, float(memory + params))
    return max(1.0, 25.0 - math.log(cost)), cost, memory, params, len(examples)


def official_process_ok(model_path: Path, task: dict[str, Any], include_arcgen: bool) -> tuple[bool, str]:
    try:
        model = sanitize_for_official_processing(onnx.load(str(model_path)))
        if model is None:
            return False, "node without output or node name containing kernel_time"
        trace = run_examples_with_profile(model, task, include_arcgen, require_correct=False)
        if not trace:
            return False, "no runnable examples"
        try:
            memory = official_memory(model, trace)
        finally:
            try:
                Path(trace).unlink()
            except OSError:
                pass
        if memory is None or memory < 0:
            return False, "official memory calculation failed"
        params = count_params(model)
        if params < 0 or params >= 10**12:
            return False, "parameter counting failed"
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def official_process_ok_prepared(model_path: Path, examples: list[PreparedExample]) -> tuple[bool, str]:
    try:
        model = sanitize_for_official_processing(onnx.load(str(model_path)))
        if model is None:
            return False, "node without output or node name containing kernel_time"
        scored = score_sanitized_model(model, examples, require_correct=False)
        if scored is None:
            return False, "official processing failed"
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def evaluate_candidate(
    source_name: str,
    model_path: Path,
    examples: list[PreparedExample],
) -> Candidate | None:
    try:
        if model_path.stat().st_size > FILESIZE_LIMIT:
            return None
        model = sanitize_for_official_processing(onnx.load(str(model_path)))
        if model is None:
            return None
        scored = score_sanitized_model(model, examples, require_correct=True)
        if scored is None:
            return None
        score, cost, memory, params, valid_examples = scored
        return Candidate(source_name, model_path, score, cost, memory, params, valid_examples)
    except Exception:
        return None


def candidate_dirs_from_repo(repo: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for name in ["submission_opt2", "submission_minimal", "submission_opt", "submission"]:
        p = repo / name
        if len(list(p.glob("task*.onnx"))) >= TASK_COUNT:
            out.append((name, p))
    for z in sorted(repo.glob("*.zip")):
        if "submission" in z.name.lower():
            out.append((z.stem, z))
    return out


def materialize_source(source: tuple[str, Path], work: Path) -> tuple[str, Path]:
    name, path = source
    if path.is_dir():
        return name, path
    out = ensure_dir(work / f"unzipped_{path.stem}")
    with zipfile.ZipFile(path) as zf:
        zf.extractall(out)
    return name, out


def run_rogermt(work: Path, tasks_json: Path, conv_budget: int) -> Path | None:
    repo = work / "rogermt_neurogolf_solver"
    out = ensure_dir(work / "candidates" / f"rogermt_budget{conv_budget}")
    if not repo.exists():
        run(["git", "clone", "https://huggingface.co/rogermt/neurogolf-solver", str(repo)], check=False)
    if not repo.exists():
        return None
    run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=repo, check=False)
    # The repo supports Kaggle task folders better than all_tasks.json. Convert JSON to task files.
    data_dir = ensure_dir(work / "rogermt_tasks")
    raw = json.loads(tasks_json.read_text())
    for i, key in enumerate(sorted(raw.keys()), 1):
        (data_dir / f"task{i:03d}.json").write_text(json.dumps(raw[key], separators=(",", ":")))
    commands = [
        [sys.executable, "-m", "neurogolf_solver.main", "--kaggle", "--data_dir", str(data_dir), "--output_dir", str(out), "--conv_budget", str(conv_budget)],
        [sys.executable, "-m", "neurogolf_solver.main", "--data-dir", str(data_dir), "--output-dir", str(out), "--conv-budget", str(conv_budget)],
        [sys.executable, "own-solver/main.py", "--data_dir", str(data_dir), "--output_dir", str(out), "--conv_budget", str(conv_budget)],
        [sys.executable, "main.py", "--data_dir", str(data_dir), "--output_dir", str(out), "--conv_budget", str(conv_budget)],
    ]
    for cmd in commands:
        proc = run(cmd, cwd=repo, check=False)
        found = len(list(out.glob("task*.onnx")))
        log(f"  rogermt command exit={proc.returncode}, produced {found} models")
        if found >= 100:
            break
    return out if len(list(out.glob("task*.onnx"))) >= 100 else None


def run_ash(work: Path, tasks_json: Path, conv_budget: int) -> Path | None:
    out = ensure_dir(work / "candidates" / f"ash_budget{conv_budget}")
    script = work / "ash_neurogolf_solver_enhanced.py"
    if not script.exists():
        url = "https://huggingface.co/ashhhhhh26/neurogolf-2026/resolve/main/neurogolf_solver_enhanced.py"
        log(f"Downloading {url}")
        try:
            urllib.request.urlretrieve(url, script)
        except Exception as exc:
            log(f"download failed: {exc}")
    if not script.exists():
        return None
    commands = [
        [sys.executable, str(script), "--data_file", str(tasks_json), "--output_dir", str(out), "--conv_budget", str(conv_budget)],
        [sys.executable, str(script), "--data-file", str(tasks_json), "--output-dir", str(out), "--conv-budget", str(conv_budget)],
        [sys.executable, str(script), str(tasks_json), str(out)],
    ]
    for cmd in commands:
        proc = run(cmd, check=False)
        found = len(list(out.glob("task*.onnx")))
        log(f"  ash command exit={proc.returncode}, produced {found} models")
        if found >= 100:
            break
    return out if len(list(out.glob("task*.onnx"))) >= 100 else None


def collect_candidates(
    sources: list[tuple[str, Path]],
    tasks: dict[int, dict[str, Any]],
    prepared_tasks: dict[int, list[PreparedExample]],
    include_arcgen: bool,
    work: Path,
    workers: int,
) -> tuple[dict[int, list[Candidate]], list[dict[str, Any]]]:
    by_task: dict[int, list[Candidate]] = {tid: [] for tid in range(1, TASK_COUNT + 1)}
    source_stats: list[dict[str, Any]] = []
    for source in sources:
        source_name, source_dir = materialize_source(source, work)
        log(f"Validating source: {source_name} ({source_dir})")
        jobs = []
        for tid in range(1, TASK_COUNT + 1):
            model_path = source_dir / f"task{tid:03d}.onnx"
            if model_path.exists():
                jobs.append((tid, model_path))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            future_to_tid = {
                pool.submit(evaluate_candidate, source_name, path, prepared_tasks[tid]): tid
                for tid, path in jobs
            }
            accepted = 0
            for future in concurrent.futures.as_completed(future_to_tid):
                tid = future_to_tid[future]
                candidate = future.result()
                if candidate is not None:
                    by_task[tid].append(candidate)
                    accepted += 1
            log(f"  accepted {accepted}/{len(jobs)} candidates from {source_name}")
            source_stats.append({
                "source": source_name,
                "path": str(source_dir),
                "models_found": len(jobs),
                "accepted": accepted,
            })
    return by_task, source_stats


def make_identity_model(path: Path) -> None:
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.save(model, str(path))


def package_best(
    by_task: dict[int, list[Candidate]],
    tasks: dict[int, dict[str, Any]],
    prepared_tasks: dict[int, list[PreparedExample]],
    include_arcgen: bool,
    work: Path,
    output_zip: Path,
    preserve_dir: Path | None = None,
) -> dict[str, Any]:
    ensure_dir(output_zip.parent)
    final_dir = ensure_dir(work / "final_submission")
    fallback_dir = ensure_dir(work / "fallback")
    rows = []
    processing_replacements = []
    source_counts: dict[str, int] = {}
    source_scores: dict[str, float] = {}
    total = 0.0
    solved = 0
    for tid in range(1, TASK_COUNT + 1):
        preserved = preserve_dir / f"task{tid:03d}.onnx" if preserve_dir else None
        if preserved and preserved.exists():
            process_ok, process_reason = official_process_ok_prepared(preserved, prepared_tasks[tid])
            if process_ok:
                score, cost, memory, params = model_score(preserved)
                best = Candidate("preserved_baseline", preserved, score, cost, memory, params, len(prepared_tasks[tid]))
            else:
                processing_replacements.append({"task": tid, "source": "preserved_baseline", "reason": process_reason})
                best = None
        else:
            best = None

        candidates = by_task[tid]
        if best is None and not candidates:
            fallback = fallback_dir / f"task{tid:03d}.onnx"
            if not fallback.exists():
                make_identity_model(fallback)
            scored = evaluate_candidate("fallback_identity", fallback, prepared_tasks[tid])
            if scored is None:
                score, cost, memory, params = model_score(fallback)
                best = Candidate("fallback_identity", fallback, 1.0, cost, memory, params, 0)
            else:
                best = scored
        elif best is None:
            best = sorted(candidates, key=lambda c: (c.score, -c.cost), reverse=True)[0]
            solved += 1
        final_path = final_dir / f"task{tid:03d}.onnx"
        shutil.copy2(best.path, final_path)
        process_ok, process_reason = official_process_ok_prepared(final_path, prepared_tasks[tid])
        if not process_ok:
            fallback = fallback_dir / f"task{tid:03d}.onnx"
            if not fallback.exists():
                make_identity_model(fallback)
            shutil.copy2(fallback, final_path)
            fallback_ok, fallback_reason = official_process_ok_prepared(final_path, prepared_tasks[tid])
            if not fallback_ok:
                raise SystemExit(f"Safe fallback failed processing for task{tid:03d}: {fallback_reason}")
            processing_replacements.append({"task": tid, "source": best.source, "reason": process_reason})
            best = Candidate("processing_safe_identity", final_path, 1.0, best.cost, best.memory, best.params, 0)
        total += best.score if best.valid_examples else 1.0
        source_counts[best.source] = source_counts.get(best.source, 0) + 1
        source_scores[best.source] = source_scores.get(best.source, 0.0) + (best.score if best.valid_examples else 1.0)
        rows.append({
            "task": tid,
            "source": best.source,
            "score_est": round(best.score if best.valid_examples else 1.0, 6),
            "cost": best.cost,
            "memory": best.memory,
            "params": best.params,
            "valid_examples": best.valid_examples,
            "path": str(best.path),
        })

    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for tid in range(1, TASK_COUNT + 1):
            p = final_dir / f"task{tid:03d}.onnx"
            zf.write(p, p.name)

    report = {
        "estimated_score": total,
        "valid_task_count": solved,
        "processing_replacements": processing_replacements,
        "source_counts": source_counts,
        "source_scores": source_scores,
        "zip": str(output_zip),
        "zip_size": output_zip.stat().st_size,
        "rows": rows,
    }
    (work / "final_report.json").write_text(json.dumps(report, indent=2))
    return report


def submit(output_zip: Path, message: str) -> None:
    run(["kaggle", "competitions", "submit", "-c", COMPETITION, "-f", str(output_zip), "-m", message], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--data-file", type=Path, default=None)
    ap.add_argument("--work-dir", type=Path, default=Path("remote_runs"))
    ap.add_argument("--output-zip", type=Path, default=Path("final_submission.zip"))
    ap.add_argument("--conv-budget", type=int, default=60)
    ap.add_argument("--include-arcgen", action="store_true")
    ap.add_argument("--skip-public-solvers", action="store_true")
    ap.add_argument("--extra-source", type=Path, action="append", default=[],
                    help="Extra directory or zip containing task001.onnx ... task400.onnx")
    ap.add_argument("--preserve-source", type=Path, default=None,
                    help="Baseline zip or directory to preserve as final output except processing-failing tasks")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--min-submit-score", type=float, default=6000.0)
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    work = ensure_dir(args.work_dir.resolve())
    data_dir = args.data_dir or (repo / "data")
    tasks = load_tasks(data_dir if data_dir.exists() else None, args.data_file)
    prepared_tasks = {tid: prepare_examples(task, args.include_arcgen) for tid, task in tasks.items()}
    tasks_json = write_all_tasks_json(tasks, work / "all_tasks.json")

    sources = candidate_dirs_from_repo(repo)
    for extra in args.extra_source:
        if extra.exists():
            sources.append((extra.stem, extra.resolve()))
    log(f"Existing candidate sources: {[s[0] for s in sources]}")
    preserve_dir = None
    if args.preserve_source:
        _, preserve_dir = materialize_source(("preserved_baseline", args.preserve_source.resolve()), work)
        log(f"Preserving baseline source: {preserve_dir}")

    if not args.skip_public_solvers:
        try:
            ash = run_ash(work, tasks_json, args.conv_budget)
            if ash:
                sources.append((ash.name, ash))
            else:
                log("ash solver did not produce enough task*.onnx files")
        except Exception as exc:
            log(f"ash solver failed: {exc}")
        try:
            roger = run_rogermt(work, tasks_json, args.conv_budget)
            if roger:
                sources.append((roger.name, roger))
            else:
                log("rogermt solver did not produce enough task*.onnx files")
        except Exception as exc:
            log(f"rogermt solver failed: {exc}")

    by_task, source_stats = collect_candidates(sources, tasks, prepared_tasks, args.include_arcgen, work, args.workers)
    report = package_best(by_task, tasks, prepared_tasks, args.include_arcgen, work, args.output_zip.resolve(), preserve_dir)
    report["source_stats"] = source_stats
    (work / "final_report.json").write_text(json.dumps(report, indent=2))

    print("\n=== FINAL ===")
    print(f"zip: {report['zip']}")
    print(f"zip_size: {report['zip_size']:,} bytes")
    print(f"valid_task_count: {report['valid_task_count']}/{TASK_COUNT}")
    print(f"estimated_score: {report['estimated_score']:.3f}")
    print("source_counts:")
    for source, count in sorted(report["source_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {source}: {count}")
    print("source_stats:")
    for item in source_stats:
        print(f"  {item['source']}: accepted {item['accepted']}/{item['models_found']}")
    print(f"report: {work / 'final_report.json'}")

    if args.submit:
        if report["estimated_score"] < args.min_submit_score:
            raise SystemExit(
                f"Refusing to submit: estimated_score={report['estimated_score']:.3f} "
                f"< --min-submit-score={args.min_submit_score:.3f}"
            )
        submit(args.output_zip.resolve(), f"remote ensemble est={report['estimated_score']:.2f}")


if __name__ == "__main__":
    main()
