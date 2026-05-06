"""
Add precomputed H1 token-context contribution lists to a packed TI export.

This preserves the trained model math, but moves most of H1's token embedding
matrix multiply off-calculator. The generated TI-BASIC runtime can then add
selected 198-value contribution vectors instead of multiplying all 850 token
context inputs by W_h on every step.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_DIM = 85
PARTS = [
    ("onset", "E_onset", 0, 21),
    ("nucleus", "E_nucleus", 21, 21),
    ("coda", "E_coda", 42, 21),
    ("stress", "E_stress", 63, 21),
    ("wb", "E_wb", 84, 1),
]


def list_filename(name: str) -> str:
    return f"{name}.csv"


def read_values(path: Path) -> list[float]:
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            values.extend(float(x) for x in row if x != "")
    return values


def write_values(path: Path, values: np.ndarray, precision: int) -> None:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    path.write_text("\n".join(f"{float(v):.{precision}f}" for v in flat), encoding="utf-8")


class PackedReader:
    def __init__(self, exportdir: Path):
        self.exportdir = exportdir
        self.manifest = json.loads((exportdir / "packed_manifest.json").read_text(encoding="utf-8"))
        self.flat: list[float] = []
        for name in self.manifest["list_names"]:
            self.flat.extend(read_values(exportdir / list_filename(name)))

    def tensor(self, name: str) -> np.ndarray:
        ref = self.manifest["tensors"][name]
        start = int(ref["start"])
        length = int(ref["length"])
        return np.array(self.flat[start:start + length], dtype=np.float64).reshape(tuple(ref["shape"]))


def next_precompute_name(index: int) -> str:
    return f"|LP{index}"


def copy_base_export(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for old in dst.glob("|LP*.csv"):
        old.unlink()


DISC_DIM = 16
WORD_DIM = 8
DISC_COL_START = 850   # x[850:866] = disc state within 874-dim input
WORD_COL_START = 866   # x[866:874] = word state


def build_precompute(src: Path, dst: Path, group_size: int = 10) -> None:
    reader = PackedReader(src)
    manifest = json.loads((src / "packed_manifest.json").read_text(encoding="utf-8"))
    precision = int(manifest.get("export_precision", 6))
    hidden_dim = int(reader.tensor("b_h").shape[0])
    rows_per_chunk = 99
    row_chunks = (hidden_dim + rows_per_chunk - 1) // rows_per_chunk
    context_len = 10

    copy_base_export(src, dst)

    tables: list[dict[str, Any]] = []
    list_index = 1
    for row_chunk in range(row_chunks):
        wh_parts = [reader.tensor(f"W_h_r{row_chunk}_c{col}") for col in range(9)]
        w_h = np.concatenate(wh_parts, axis=1)
        rows = int(w_h.shape[0])
        for slot in range(context_len):
            slot_base = slot * TOKEN_DIM
            for part_name, emb_name, part_offset, width in PARTS:
                emb = reader.tensor(emb_name)
                block = w_h[:, slot_base + part_offset:slot_base + part_offset + width]
                contrib = emb @ block.T
                groups: list[dict[str, Any]] = []
                for first_id0 in range(0, contrib.shape[0], group_size):
                    page = contrib[first_id0:first_id0 + group_size]
                    list_name = next_precompute_name(list_index)
                    write_values(dst / list_filename(list_name), page, precision)
                    groups.append({
                        "list": list_name,
                        "first_id": first_id0 + 1,
                        "count": int(page.shape[0]),
                    })
                    list_index += 1
                tables.append({
                    "row_chunk": row_chunk,
                    "slot": slot,
                    "part": part_name,
                    "context_id_index": slot * len(PARTS) + PARTS.index((part_name, emb_name, part_offset, width)) + 1,
                    "rows": rows,
                    "groups": groups,
                })

        # The precompute tables only cover x[0:850] (token context dims).  W_h
        # also has columns for x[850:866] (disc state) and x[866:874] (word state)
        # that are not in any precompute table.  Fold those columns into the stored
        # W_disc_h1 / W_word_h1 weights so the runtime adds the correct total.
        W_disc_h1_combined = (
            reader.tensor(f"W_disc_h1_{row_chunk}")
            + w_h[:, DISC_COL_START:DISC_COL_START + DISC_DIM]
        )
        W_word_h1_combined = (
            reader.tensor(f"W_word_h1_{row_chunk}")
            + w_h[:, WORD_COL_START:WORD_COL_START + WORD_DIM]
        )

        def write_combined(combined: np.ndarray, base_name: str) -> dict[str, Any]:
            flat = combined.reshape(-1)
            total = int(flat.size)
            segs: list[dict[str, Any]] = []
            offset = 0
            seg_idx = 0
            max_per_list = 999
            while offset < total:
                chunk = flat[offset:offset + max_per_list]
                seg_name = base_name if seg_idx == 0 else f"{base_name}_{seg_idx}"
                write_values(dst / list_filename(seg_name), chunk, precision)
                segs.append({"list": seg_name, "offset": 1, "length": int(chunk.size)})
                offset += int(chunk.size)
                seg_idx += 1
            return {
                "shape": list(combined.shape),
                "start": 0,
                "length": total,
                "segments": segs,
            }

        manifest["tensors"][f"W_disc_h1_{row_chunk}"] = write_combined(
            W_disc_h1_combined, f"|LPD{row_chunk}"
        )
        manifest["tensors"][f"W_word_h1_{row_chunk}"] = write_combined(
            W_word_h1_combined, f"|LPW{row_chunk}"
        )

    manifest["h1_precompute"] = {
        "enabled": True,
        "context_ids_list": "|LI",
        "context_len": context_len,
        "parts_per_token": len(PARTS),
        "rows_per_chunk": rows_per_chunk,
        "group_size": group_size,
        "list_count": list_index - 1,
        "tables": tables,
    }
    (dst / "packed_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {dst}")
    print(f"H1 precompute lists: {list_index - 1}")
    print(f"H1 precompute values: {(list_index - 1) * group_size * rows_per_chunk} max slots, {sum(t['rows'] * sum(g['count'] for g in t['groups']) for t in tables)} actual values")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="ti_packed_export")
    parser.add_argument("--dst", default="ti_packed_export_h1pre")
    parser.add_argument("--group-size", type=int, default=10)
    args = parser.parse_args()
    build_precompute(Path(args.src), Path(args.dst), group_size=args.group_size)


if __name__ == "__main__":
    main()
