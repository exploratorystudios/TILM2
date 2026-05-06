"""
Prune original H1 token-context weights from an H1-precompute TI export.

When H1 precompute is enabled, tensors named W_h_r*_c* are no longer needed on
calculator. This script removes those tensors and repacks the remaining base
parameter tensors into fewer L-lists while preserving support/precompute CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


BUILTIN_LIST_NAMES = [f"L{i}" for i in range(1, 7)]
TOKEN_CONTEXT_DIM = 850
DISCOURSE_STATE_DIM = 16
WORD_STATE_DIM = 8


def custom_list_name(index: int) -> str:
    return f"|LL{index}"


def ti_list_names(count: int) -> list[str]:
    return [BUILTIN_LIST_NAMES[i] if i < len(BUILTIN_LIST_NAMES) else custom_list_name(i + 1) for i in range(count)]


def list_filename(name: str) -> str:
    return f"{name}.csv"


def read_values(path: Path) -> list[float]:
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            values.extend(float(x) for x in row if x != "")
    return values


def write_values(path: Path, values: list[float], precision: int) -> None:
    path.write_text("\n".join(f"{v:.{precision}f}" for v in values), encoding="utf-8")


def tensor_values(flat: list[float], ref: dict[str, Any]) -> list[float]:
    start = int(ref["start"])
    length = int(ref["length"])
    return flat[start:start + length]


def tensor_array(flat: list[float], ref: dict[str, Any]) -> tuple[list[int], list[float]]:
    shape = [int(v) for v in ref["shape"]]
    return shape, tensor_values(flat, ref)


def fold_pruned_h1_state_columns(name: str, values: list[float], manifest: dict[str, Any], old_flat: list[float]) -> list[float]:
    if not (name.startswith("W_disc_h1_") or name.startswith("W_word_h1_")):
        return values

    row_chunk = int(name.rsplit("_", 1)[1])
    wh_name = f"W_h_r{row_chunk}_c8"
    wh_ref = manifest["tensors"].get(wh_name)
    if wh_ref is None:
        return values

    wh_shape, wh_values = tensor_array(old_flat, wh_ref)
    rows, cols = wh_shape
    state_offset = TOKEN_CONTEXT_DIM - 8 * 99
    if cols < state_offset + DISCOURSE_STATE_DIM + WORD_STATE_DIM:
        return values

    out = list(values)
    out_cols = DISCOURSE_STATE_DIM if name.startswith("W_disc_h1_") else WORD_STATE_DIM
    source_offset = state_offset if name.startswith("W_disc_h1_") else state_offset + DISCOURSE_STATE_DIM
    for r in range(rows):
        for c in range(out_cols):
            out[r * out_cols + c] += wh_values[r * cols + source_offset + c]
    return out


def segment_refs(start: int, length: int, names: list[str], list_size: int) -> list[dict[str, Any]]:
    if length <= 0:
        return []
    first = start // list_size
    last = (start + length - 1) // list_size
    return [
        {
            "list": names[i],
            "offset": 1 if i != first else start % list_size + 1,
            "length": min((i + 1) * list_size, start + length) - max(i * list_size, start),
        }
        for i in range(first, last + 1)
    ]


def should_prune(name: str) -> bool:
    return name.startswith("W_h_r")


def prune(src: Path, dst: Path) -> None:
    manifest = json.loads((src / "packed_manifest.json").read_text(encoding="utf-8"))
    precision = int(manifest.get("export_precision", 6))
    list_size = int(manifest["list_size"])

    old_flat: list[float] = []
    for name in manifest["list_names"]:
        old_flat.extend(read_values(src / list_filename(name)))

    new_flat: list[float] = []
    new_tensors: dict[str, Any] = {}
    pruned_values = 0
    pruned_tensors = 0
    for name, ref in manifest["tensors"].items():
        if should_prune(name):
            pruned_values += int(ref["length"])
            pruned_tensors += 1
            continue
        values = tensor_values(old_flat, ref)
        values = fold_pruned_h1_state_columns(name, values, manifest, old_flat)
        start = len(new_flat)
        new_flat.extend(values)
        new_tensors[name] = {
            **ref,
            "start": start,
            "length": len(values),
        }

    list_count = (len(new_flat) + list_size - 1) // list_size
    names = ti_list_names(list_count)
    for ref in new_tensors.values():
        ref["segments"] = segment_refs(int(ref["start"]), int(ref["length"]), names, list_size)

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    base_list_names = set(manifest["list_names"])
    for item in src.iterdir():
        if not item.is_file():
            continue
        if item.name == "packed_manifest.json":
            continue
        if item.suffix == ".csv" and item.stem in base_list_names:
            continue
        shutil.copy2(item, dst / item.name)

    for i, name in enumerate(names):
        start = i * list_size
        end = min(start + list_size, len(new_flat))
        write_values(dst / list_filename(name), new_flat[start:end], precision)

    manifest["list_names"] = names
    manifest["list_count"] = list_count
    manifest["total_values"] = len(new_flat)
    manifest["tensor_count"] = len(new_tensors)
    manifest["tensors"] = new_tensors
    manifest["h1_base_pruned"] = {
        "enabled": True,
        "removed_tensor_prefix": "W_h_r",
        "removed_tensors": pruned_tensors,
        "removed_values": pruned_values,
    }
    (dst / "packed_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {dst}")
    print(f"Removed H1 base tensors: {pruned_tensors}")
    print(f"Removed H1 base values:  {pruned_values}")
    print(f"Base list count:         {len(manifest['list_names'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="ti_packed_export_h1pre")
    parser.add_argument("--dst", default="ti_packed_export_h1pre_pruned")
    args = parser.parse_args()
    prune(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
