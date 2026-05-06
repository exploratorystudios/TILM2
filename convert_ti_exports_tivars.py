"""
Convert generated TILM2 CSV exports to native TI variable files.

Requires tivars:
    python3 -m venv .venv_tivars
    .venv_tivars/bin/python -m pip install tivars

Outputs:
    .8xl for lists
    .8xm for matrices
    .8xp for generated TI-BASIC programs
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

from tivars.models import TI_84PCE
from tivars.types import TIMatrix, TIGroup, TIProgram, TIReal, TIRealList
from tivars.var import TIEntry


RUNTIME_LIST_SPECS = {
    "H": 198,
    "M": 198,
    "T": 198,
    "F": 28,
    "J": 99,
    "G": 85,
    "E": 42,
    "R": 5,
    "W": 2,
    "S": 3,
    "N": 14,
    "O": 28,
    "C": 24,
    "ID": 999,
    "I": 50,
    "HO": 15,
    "HU": 15,
    "HC": 15,
}

INIT_TRANSFER_LISTS = {"H", "M", "T", "F", "J", "G", "E", "I"}
SUPPORT_TRANSFER_LISTS = {
    "X", "HR", "MR", "RR", "WR", "SR", "NR", "OR", "CR", "NM",
    *{f"CM{i}" for i in range(1, 11)},
}
GEN_RUNTIME_LISTS = {"R", "W", "S", "N", "O", "C", "ID", "HO", "HU", "HC"}
PRECOMPUTE_LIST_PREFIXES = ("P",)
COMPARE_REF_LISTS = {"DRR", "DWR", "DSR", "DNR", "DOR", "DCR"}
STATE_PRECOMPUTE_PREFIXES = ("PD", "PW")


def read_numeric_csv(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            rows.append([float(cell) for cell in row if cell != ""])
    return rows


def real(value: float) -> TIReal:
    out = TIReal()
    out.load_float(float(value))
    return out


def list_var_name(stem: str) -> str:
    if stem.startswith("|L"):
        return stem[2:]
    if stem.startswith("|"):
        return stem[1:]
    return stem


def safe_filename(stem: str) -> str:
    if stem.startswith("[") and stem.endswith("]"):
        return stem[1:-1]
    if stem.startswith("|L"):
        return stem[2:]
    if stem.startswith("|"):
        return stem[1:]
    return stem


def force_ti_list_name(obj: TIRealList, name: str) -> None:
    if name in {f"L{i}" for i in range(1, 7)}:
        obj.raw.name = bytes([0x5D, int(name[1]) - 1]).ljust(8, b"\x00")
        return
    raw = bytes([0x5D]) + name.encode("ascii")
    obj.raw.name = raw[:8].ljust(8, b"\x00")


def convert_list(path: Path, outdir: Path) -> Path:
    values = [row[0] for row in read_numeric_csv(path)]
    name = list_var_name(path.stem)
    obj = TIRealList(name=name)
    obj.load_list([real(v) for v in values])
    obj.meta_length = TIEntry.flash_meta_length
    obj.archived = True
    force_ti_list_name(obj, name)
    outfile = outdir / f"{safe_filename(path.stem)}.8xl"
    obj.save(str(outfile), model=TI_84PCE)
    return outfile


def convert_runtime_list(name: str, length: int, outdir: Path) -> Path:
    obj = TIRealList(name=name)
    obj.load_list([real(0) for _ in range(length)])
    obj.meta_length = TIEntry.flash_meta_length
    obj.archived = True
    force_ti_list_name(obj, name)
    outfile = outdir / f"{safe_filename(name)}.8xl"
    obj.save(str(outfile), model=TI_84PCE)
    return outfile


def convert_matrix(path: Path, outdir: Path) -> Path:
    rows = read_numeric_csv(path)
    name = path.stem
    obj = TIMatrix(name=name)
    obj.load_matrix([[real(v) for v in row] for row in rows])
    outfile = outdir / f"{safe_filename(path.stem)}.8xm"
    obj.save(str(outfile), model=TI_84PCE)
    return outfile


def convert_program(path: Path, outdir: Path) -> Path:
    name = path.stem.upper()[:8]
    obj = TIProgram(name=name)
    obj.load_string(path.read_text(encoding="utf-8"), model=TI_84PCE)
    obj.archived = True
    outfile = outdir / f"{name}.8xp"
    obj.save(str(outfile), model=TI_84PCE)
    return outfile


def convert_group(list_paths: list[Path], outdir: Path, group_name: str) -> Path:
    entries = []
    for path in list_paths:
        values = [row[0] for row in read_numeric_csv(path)]
        name = list_var_name(path.stem)
        obj = TIRealList(name=name)
        obj.load_list([real(v) for v in values])
        obj.meta_length = TIEntry.flash_meta_length
        obj.archived = True
        force_ti_list_name(obj, name)
        entries.append(obj)

    group = TIGroup.group(entries, name=group_name)
    outfile = outdir / f"{group_name}.8xg"
    group.save(str(outfile), model=TI_84PCE)
    return outfile


def copy_files(paths: list[Path], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, target / path.name)


def _is_token_precompute(stem: str) -> bool:
    for prefix in STATE_PRECOMPUTE_PREFIXES:
        if stem.startswith(prefix):
            return False
    for prefix in PRECOMPUTE_LIST_PREFIXES:
        if stem.startswith(prefix):
            rest = stem[len(prefix):]
            parts = rest.split("_")
            return bool(parts) and all(p.isdigit() for p in parts)
    return False


def _is_state_precompute(stem: str) -> bool:
    for prefix in STATE_PRECOMPUTE_PREFIXES:
        if stem.startswith(prefix):
            rest = stem[len(prefix):]
            parts = rest.split("_")
            return bool(parts) and all(p.isdigit() for p in parts)
    return False


def organize_transfer_folders(outdir: Path, made: list[Path]) -> None:
    groups = {
        "programs": [p for p in made if p.suffix.lower() == ".8xp"],
        "support_lists": [
            p for p in made
            if p.suffix.lower() == ".8xl" and p.stem in SUPPORT_TRANSFER_LISTS
        ],
        "runtime_lists": [
            p for p in made
            if p.suffix.lower() == ".8xl" and p.stem in (INIT_TRANSFER_LISTS | GEN_RUNTIME_LISTS)
        ],
        "precompute_lists": [
            p for p in made
            if p.suffix.lower() == ".8xl" and _is_token_precompute(p.stem)
        ],
        "state_precompute": [
            p for p in made
            if p.suffix.lower() == ".8xl" and _is_state_precompute(p.stem)
        ],
        "compare_lists": [
            p for p in made
            if p.suffix.lower() == ".8xl" and p.stem in COMPARE_REF_LISTS
        ],
        "loose_weight_lists": [
            p for p in made
            if p.suffix.lower() == ".8xl" and p.stem.startswith("L") and p.stem[1:].isdigit()
        ],
    }

    for name, paths in groups.items():
        target = outdir / name
        if target.exists():
            shutil.rmtree(target)
        copy_files(sorted(paths), target)


def classify_csv(path: Path) -> str:
    stem = path.stem
    if stem.startswith("[") and stem.endswith("]"):
        return "matrix"
    return "list"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default="ti_packed_export")
    parser.add_argument("--program-dir", default="ti_basic_runtime")
    parser.add_argument("--outdir", default="ti_native")
    parser.add_argument("--skip-programs", action="store_true")
    parser.add_argument("--include-workspace-matrices", action="store_true",
                        help="Also export [A]-[J] workspace matrices. Usually unnecessary; T2INIT creates them on-calc.")
    parser.add_argument("--group-lists", action="store_true",
                        help="Also export all CSV lists as .8xg group batches")
    parser.add_argument("--group-batch-size", type=int, default=6,
                        help="Number of lists per .8xg group batch when --group-lists is set")
    parser.add_argument("--organize", action="store_true",
                        help="Also create transfer-focused subfolders under the output directory")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    program_dir = Path(args.program_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    made: list[Path] = []
    list_paths: list[Path] = []
    for path in sorted(csv_dir.glob("*.csv")):
        if classify_csv(path) == "matrix":
            if args.include_workspace_matrices:
                made.append(convert_matrix(path, outdir))
        else:
            list_paths.append(path)
            made.append(convert_list(path, outdir))

    existing_list_names = {list_var_name(path.stem) for path in list_paths}
    for name, length in RUNTIME_LIST_SPECS.items():
        if name not in existing_list_names:
            made.append(convert_runtime_list(name, length, outdir))

    if args.group_lists and list_paths:
        batch_size = max(1, args.group_batch_size)
        for i in range(0, len(list_paths), batch_size):
            batch = list_paths[i:i + batch_size]
            batch_name = f"T2G{i // batch_size + 1:02d}"
            made.append(convert_group(batch, outdir, batch_name))

    if not args.skip_programs:
        for path in sorted(program_dir.glob("*.txt")):
            if path.name == "README.md":
                continue
            made.append(convert_program(path, outdir))

    if args.organize:
        organize_transfer_folders(outdir, made)

    list_count = len([p for p in made if p.suffix.lower() == ".8xl"])
    matrix_count = len([p for p in made if p.suffix.lower() == ".8xm"])
    program_count = len([p for p in made if p.suffix.lower() == ".8xp"])
    print(f"Wrote {len(made)} native TI files to {outdir}")
    print(f"  lists:    {list_count}")
    print(f"  matrices: {matrix_count}")
    print(f"  programs: {program_count}")
    print("Bulk send example:")
    print(f"  tilp --calc=ti84+ --cable=DirectLink --port=1 --no-gui --silent {outdir}/*")
    if args.organize:
        print("Transfer folders:")
        for name in ["programs", "support_lists", "runtime_lists", "precompute_lists", "state_precompute", "compare_lists", "loose_weight_lists"]:
            count = len(list((outdir / name).iterdir())) if (outdir / name).exists() else 0
            print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
