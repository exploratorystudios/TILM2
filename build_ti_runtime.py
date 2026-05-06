"""
End-to-end TI export pipeline for a trained TILM2 model.

Builds:
  1. packed CSV weights/lists
  2. optional H1 precomputed contribution lists
  3. generated TI-BASIC source
  4. native .8xl/.8xp/.8xg transfer folders via tivars
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-prefix", default="ti_build")
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=21)
    parser.add_argument("--hidden-dim", type=int, default=198)
    parser.add_argument("--list-size", type=int, default=999)
    parser.add_argument("--h1-precompute", action="store_true")
    parser.add_argument("--keep-h1-base-weights", action="store_true")
    parser.add_argument("--h2-colmajor", action="store_true")
    parser.add_argument("--english-runtime", action="store_true")
    parser.add_argument("--include-debug", action="store_true")
    parser.add_argument("--include-compare-fixture", action="store_true")
    parser.add_argument("--compare-seed-text", default="the moon rose by the")
    parser.add_argument("--english-words", default=str(Path(__file__).resolve().parent / "vocab_words.txt"))
    parser.add_argument("--cmu-dict", default=str(Path(__file__).resolve().parent / "cmudict.dict"))
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--group-batch-size", type=int, default=6)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    prefix = root / args.out_prefix
    packed_base = prefix.with_name(prefix.name + "_packed")
    packed_h1pre = prefix.with_name(prefix.name + "_packed_h1pre")
    packed_pruned = prefix.with_name(prefix.name + "_packed_h1pre_pruned")
    packed_h2col = prefix.with_name(prefix.name + "_packed_h2col")
    packed_final = packed_base
    if args.h1_precompute:
        packed_final = packed_h1pre if args.keep_h1_base_weights else packed_pruned
    runtime_dir = prefix.with_name(prefix.name + "_basic")
    native_dir = prefix.with_name(prefix.name + "_native")

    for path in [packed_base, runtime_dir, native_dir]:
        if path.exists():
            shutil.rmtree(path)
    for path in [packed_h1pre, packed_pruned, packed_h2col]:
        if path.exists():
            shutil.rmtree(path)

    run([
        sys.executable, str(root / "ti_packed_runtime.py"),
        "--weights", args.weights,
        "--data", args.data,
        "--outdir", str(packed_base),
        "--precision", str(args.precision),
        "--list-size", str(args.list_size),
        "--context-len", str(args.context_len),
        "--embed-dim", str(args.embed_dim),
        "--hidden-dim", str(args.hidden_dim),
    ])
    run([
        sys.executable, str(root / "export_ti_forward_fixture.py"),
        "--weights", args.weights,
        "--data", args.data,
        "--outdir", str(packed_base),
        "--window-index", str(args.window_index),
        "--precision", str(args.precision),
        "--temperature", str(args.temperature),
        "--min-freq", str(args.min_freq),
        "--context-len", str(args.context_len),
        "--embed-dim", str(args.embed_dim),
        "--hidden-dim", str(args.hidden_dim),
    ])

    if args.h1_precompute:
        run([
            sys.executable, str(root / "add_ti_h1_precompute.py"),
            "--src", str(packed_base),
            "--dst", str(packed_h1pre),
        ])
        if not args.keep_h1_base_weights:
            run([
                sys.executable, str(root / "prune_ti_h1_base_weights.py"),
                "--src", str(packed_h1pre),
                "--dst", str(packed_pruned),
            ])

    if args.h2_colmajor:
        h2_src = packed_final
        run([
            sys.executable, str(root / "repack_ti_column_major.py"),
            "--src", str(h2_src),
            "--dst", str(packed_h2col),
            "--prefix", "W_h2_",
        ])
        packed_final = packed_h2col

    run([
        sys.executable, str(root / "generate_ti_basic_runtime.py"),
        "--exportdir", str(packed_final),
        "--outdir", str(runtime_dir),
        *([] if not args.include_debug else ["--include-debug"]),
    ])
    if args.english_runtime:
        run([
            sys.executable, str(root / "generate_ti_english_runtime.py"),
            "--runtime-dir", str(runtime_dir),
            "--vocab-json", str(packed_final / "vocab.json"),
            "--words", args.english_words,
            "--cmu-dict", args.cmu_dict,
        ])
    run([
        sys.executable, str(root / "convert_ti_exports_tivars.py"),
        "--csv-dir", str(packed_final),
        "--program-dir", str(runtime_dir),
        "--outdir", str(native_dir),
        "--group-lists",
        "--group-batch-size", str(args.group_batch_size),
        "--organize",
    ])

    if args.include_compare_fixture:
        compare_src = prefix.with_name(prefix.name + "_compare_fixture")
        compare_native = prefix.with_name(prefix.name + "_compare_native")
        if compare_src.exists():
            shutil.rmtree(compare_src)
        if compare_native.exists():
            shutil.rmtree(compare_native)
        run([
            sys.executable, str(root / "export_ti_step1_compare_fixture.py"),
            "--weights", args.weights,
            "--data", args.data,
            "--seed-text", args.compare_seed_text,
            "--outdir", str(compare_src),
            "--precision", str(args.precision),
            "--temperature", str(args.temperature),
            "--cmu-dict", args.cmu_dict,
            "--context-len", str(args.context_len),
            "--embed-dim", str(args.embed_dim),
            "--hidden-dim", str(args.hidden_dim),
        ])
        run([
            sys.executable, str(root / "convert_ti_exports_tivars.py"),
            "--csv-dir", str(compare_src),
            "--program-dir", str(compare_src / "__empty__"),
            "--outdir", str(compare_native),
            "--skip-programs",
            "--organize",
        ])
        compare_lists = compare_native / "compare_lists"
        target_compare_lists = native_dir / "compare_lists"
        if target_compare_lists.exists():
            shutil.rmtree(target_compare_lists)
        if compare_lists.exists():
            shutil.copytree(compare_lists, target_compare_lists)

    print("\nBuild complete")
    print(f"  packed:  {packed_final}")
    print(f"  basic:   {runtime_dir}")
    print(f"  native:  {native_dir}")
    if args.h1_precompute:
        print("  H1 precompute enabled; transfer programs plus weight/precompute groups.")
    if args.include_debug:
        print("  debug programs included in runtime build.")
    if args.include_compare_fixture:
        print("  compare fixture copied into native/compare_lists.")


if __name__ == "__main__":
    main()
