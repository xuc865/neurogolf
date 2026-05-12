#!/usr/bin/env python3
"""Replace task101.onnx in an existing submission zip with a safe identity model."""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def make_identity(path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])], "g", [x], [y])
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_zip", type=Path)
    ap.add_argument("output_zip", type=Path)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        identity = tmp / "task101.onnx"
        make_identity(identity)
        with zipfile.ZipFile(args.input_zip) as zin, zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zout:
            seen = set()
            for info in zin.infolist():
                name = Path(info.filename).name
                if not name or name in seen:
                    continue
                seen.add(name)
                if name == "task101.onnx":
                    zout.write(identity, "task101.onnx")
                else:
                    zout.writestr(name, zin.read(info.filename))
            if "task101.onnx" not in seen:
                zout.write(identity, "task101.onnx")
    print(f"Wrote {args.output_zip}")


if __name__ == "__main__":
    main()
