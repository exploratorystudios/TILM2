"""
Infer |LI.csv context IDs from an exported |LX.csv fixture and embedding tables.

This is useful when adding H1 precompute support to an older packed export that
already has |LX but was created before |LI was emitted by the fixture exporter.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


TOKEN_DIM = 85
PARTS = [
    ("E_onset", 0, 21),
    ("E_nucleus", 21, 21),
    ("E_coda", 42, 21),
    ("E_stress", 63, 21),
    ("E_wb", 84, 1),
]


def read_values(path: Path) -> list[float]:
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            values.extend(float(x) for x in row if x != "")
    return values


def write_values(path: Path, values: list[int]) -> None:
    path.write_text("\n".join(str(v) for v in values), encoding="utf-8")


class PackedReader:
    def __init__(self, exportdir: Path):
        self.exportdir = exportdir
        self.manifest = json.loads((exportdir / "packed_manifest.json").read_text(encoding="utf-8"))
        self.flat: list[float] = []
        for name in self.manifest["list_names"]:
            self.flat.extend(read_values(exportdir / f"{name}.csv"))

    def tensor(self, name: str) -> np.ndarray:
        ref = self.manifest["tensors"][name]
        start = int(ref["start"])
        length = int(ref["length"])
        return np.array(self.flat[start:start + length], dtype=np.float64).reshape(tuple(ref["shape"]))


def nearest_id(table: np.ndarray, value: np.ndarray) -> int:
    distances = np.linalg.norm(table - value.reshape(1, -1), axis=1)
    return int(np.argmin(distances)) + 1


def infer(exportdir: Path) -> None:
    reader = PackedReader(exportdir)
    lx = np.array(read_values(exportdir / "|LX.csv"), dtype=np.float64)
    if lx.size < 10 * TOKEN_DIM:
        raise ValueError(f"|LX.csv is too short: {lx.size}")

    ids: list[int] = []
    for slot in range(10):
        token = lx[slot * TOKEN_DIM:(slot + 1) * TOKEN_DIM]
        if float(np.linalg.norm(token)) < 1e-10:
            ids.extend([0, 0, 0, 0, 0])
            continue
        for tensor_name, offset, width in PARTS:
            table = reader.tensor(tensor_name)
            ids.append(nearest_id(table, token[offset:offset + width]))

    write_values(exportdir / "|LI.csv", ids)
    print(f"Wrote {exportdir / '|LI.csv'}")
    print("IDs:", ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exportdir", required=True)
    args = parser.parse_args()
    infer(Path(args.exportdir))


if __name__ == "__main__":
    main()
