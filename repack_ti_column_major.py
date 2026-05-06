"""
Rewrite selected packed tensors in column-major order for faster TI-BASIC matvecs.

The tensor names, starts, lengths, and list names stay unchanged. Only the order
of values inside selected 2D tensors changes, and the manifest marks those
tensors with "layout": "col_major" so the runtime generator emits matching code.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def read_values(path: Path) -> list[float]:
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            values.extend(float(x) for x in row if x != "")
    return values


def write_values(path: Path, values: list[float], precision: int) -> None:
    path.write_text("\n".join(f"{v:.{precision}f}" for v in values), encoding="utf-8")


def list_filename(name: str) -> str:
    return f"{name}.csv"


def should_repack(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def repack_export(src: Path, dst: Path, prefixes: tuple[str, ...]) -> None:
    manifest = json.loads((src / "packed_manifest.json").read_text(encoding="utf-8"))
    precision = int(manifest.get("export_precision", 6))

    flat: list[float] = []
    for name in manifest["list_names"]:
        flat.extend(read_values(src / list_filename(name)))

    repacked = 0
    for name, tensor in manifest["tensors"].items():
        shape = tuple(int(v) for v in tensor["shape"])
        if len(shape) != 2 or not should_repack(name, prefixes):
            tensor.pop("layout", None)
            continue

        start = int(tensor["start"])
        length = int(tensor["length"])
        arr = np.array(flat[start:start + length], dtype=np.float64).reshape(shape)
        flat[start:start + length] = arr.reshape(-1, order="F").tolist()
        tensor["layout"] = "col_major"
        repacked += 1

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for item in src.iterdir():
        if item.name == "packed_manifest.json":
            continue
        if item.is_file() and not item.name.endswith(".csv"):
            shutil.copy2(item, dst / item.name)
        elif item.is_file() and item.name.endswith(".json"):
            shutil.copy2(item, dst / item.name)

    list_size = int(manifest["list_size"])
    for i, name in enumerate(manifest["list_names"]):
        start = i * list_size
        end = min(start + list_size, len(flat))
        write_values(dst / list_filename(name), flat[start:end], precision)

    for item in src.iterdir():
        if item.is_file() and item.name.endswith(".csv") and item.stem not in manifest["list_names"]:
            shutil.copy2(item, dst / item.name)

    (dst / "packed_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {dst}")
    print(f"Column-major tensors: {repacked}")
    print(f"Prefixes: {', '.join(prefixes)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="ti_packed_export")
    parser.add_argument("--dst", default="ti_packed_export_colmajor_h1")
    parser.add_argument(
        "--prefix",
        action="append",
        default=None,
        help="Tensor prefix to rewrite; can be repeated. Default: W_h_",
    )
    args = parser.parse_args()
    repack_export(Path(args.src), Path(args.dst), tuple(args.prefix or ["W_h_"]))


if __name__ == "__main__":
    main()
