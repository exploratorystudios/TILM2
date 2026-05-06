"""
Export the current TILM2 model to TI-84-friendly CSV chunks plus metadata.

This exporter is aligned with the current chunked/stateful architecture in
`tilm2_model.py`. It writes every matrix or vector needed for inference as
separate CSV files, alongside a JSON manifest that describes how to reassemble
the pieces on the TI side.
"""

import argparse
import json
import os
from typing import Any

import numpy as np

from tilm2_model import TILM2, Vocab


def export_matrix(arr: np.ndarray, name: str, outdir: str, precision: int = 6) -> dict[str, Any]:
    path = os.path.join(outdir, f"{name}.csv")
    rows, cols = arr.shape
    with open(path, "w") as f:
        for row in arr:
            f.write(",".join(f"{float(v):.{precision}f}" for v in row) + "\n")
    return {"name": name, "type": "matrix", "shape": [rows, cols], "path": os.path.basename(path)}


def export_vector(arr: np.ndarray, name: str, outdir: str, precision: int = 6) -> dict[str, Any]:
    path = os.path.join(outdir, f"{name}.csv")
    flat = arr.reshape(-1)
    with open(path, "w") as f:
        f.write(",".join(f"{float(v):.{precision}f}" for v in flat) + "\n")
    return {"name": name, "type": "vector", "shape": [int(flat.shape[0])], "path": os.path.basename(path)}


def export_chunks(chunks: list[np.ndarray], base: str, outdir: str, precision: int = 6) -> list[dict[str, Any]]:
    entries = []
    for i, chunk in enumerate(chunks):
        entries.append(export_matrix(chunk, f"{base}_{i}", outdir, precision))
    return entries


def export_2d_chunks(chunks: list[list[np.ndarray]], base: str, outdir: str, precision: int = 6) -> list[dict[str, Any]]:
    entries = []
    for i, row_chunks in enumerate(chunks):
        for j, chunk in enumerate(row_chunks):
            entries.append(export_matrix(chunk, f"{base}_r{i}_c{j}", outdir, precision))
    return entries


def export_vocab_index(vocab: Vocab, outdir: str) -> None:
    path = os.path.join(outdir, "vocab_index.txt")
    with open(path, "w") as f:
        f.write("ONSET INDEX\n")
        for i, v in enumerate(vocab.onsets):
            f.write(f"  {i}: '{v}'\n")
        f.write("\nNUCLEUS INDEX\n")
        for i, v in enumerate(vocab.nuclei):
            f.write(f"  {i}: '{v}'\n")
        f.write("\nCODA INDEX\n")
        for i, v in enumerate(vocab.codas):
            f.write(f"  {i}: '{v}'\n")
        f.write("\nROLE INDEX\n")
        for i, v in enumerate(vocab.roles):
            f.write(f"  {i}: '{v}'\n")
        f.write("\nSTRESS: 0=unstressed, 1=primary, 2=secondary\n")


def collect_exports(model: TILM2, outdir: str, precision: int) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "architecture": "tilm2_stateful_chunked_v1",
        "hyperparams": {
            "context_len": model.context_len,
            "embed_dim": model.embed_dim,
            "hidden_dim": model.hidden_dim,
            "token_dim": model.token_dim,
            "discourse_state_dim": model.discourse_state_dim,
            "word_state_dim": model.word_state_dim,
            "chunk_size": model.chunk_size,
            "n_chunks": model.n_chunks,
            "n_input_col_chunks": model.n_input_col_chunks,
            "n_h_col_chunks": model.n_h_col_chunks,
        },
        "files": [],
    }

    files = manifest["files"]
    files.extend(export_2d_chunks(model.W_h_chunks, "W_h", outdir, precision))
    files.extend(export_2d_chunks(model.W_h2_chunks, "W_h2", outdir, precision))
    files.extend(export_chunks(model.W_disc_h1_chunks, "W_disc_h1", outdir, precision))
    files.extend(export_chunks(model.W_word_h1_chunks, "W_word_h1", outdir, precision))
    files.extend(export_chunks(model.W_disc_h2_chunks, "W_disc_h2", outdir, precision))
    files.extend(export_chunks(model.W_word_h2_chunks, "W_word_h2", outdir, precision))

    for name in ["W_onset_chunks", "W_nucleus_chunks", "W_coda_chunks", "W_stress_chunks", "W_wb_chunks", "W_role_chunks"]:
        chunks = getattr(model, name)
        files.extend(export_chunks(chunks, name.replace("_chunks", ""), outdir, precision))

    matrix_names = [
        "E_onset", "E_nucleus", "E_coda", "E_stress", "E_wb",
        "W_disc_in", "W_disc_h", "W_word_in", "W_word_h", "W_word_disc",
        "W_disc_onset", "W_disc_nucleus", "W_disc_coda", "W_disc_stress", "W_disc_wb", "W_disc_role",
        "W_word_onset", "W_word_nucleus", "W_word_coda", "W_word_stress", "W_word_wb", "W_word_role",
        "W_role_onset", "W_role_nucleus", "W_role_coda", "W_role_stress", "W_role_wb",
        "W_wb_onset", "W_wb_nucleus", "W_wb_coda",
        "W_nuc_gate", "W_stress_gate", "W_nuc_cond", "W_coda_cond",
    ]
    for name in matrix_names:
        files.append(export_matrix(getattr(model, name), name, outdir, precision))

    vector_names = [
        "b_h", "b_h2", "b_onset", "b_nucleus", "b_coda", "b_stress", "b_wb", "b_role",
        "b_disc", "b_word", "b_nuc_gate", "b_stress_gate",
    ]
    for name in vector_names:
        files.append(export_vector(getattr(model, name), name, outdir, precision))

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", default="ti_export")
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=21)
    parser.add_argument("--hidden-dim", type=int, default=198)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.data) as f:
        data = json.load(f)
    vocab_dict = dict(data["vocab"])
    if "role_vocab" in data:
        vocab_dict["role_vocab"] = data["role_vocab"]
    vocab = Vocab(vocab_dict)

    model = TILM2(vocab, context_len=args.context_len, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
    model.load(args.weights)

    manifest = collect_exports(model, args.outdir, args.precision)

    manifest["vocab_sizes"] = {
        "onsets": vocab.n_onsets,
        "nuclei": vocab.n_nuclei,
        "codas": vocab.n_codas,
        "roles": vocab.n_roles,
    }
    manifest["input_col_starts"] = model.input_col_starts
    manifest["h_col_starts"] = model.h_col_starts

    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(args.outdir, "vocab.json"), "w") as f:
        json.dump({
            "onsets": vocab.onsets,
            "nuclei": vocab.nuclei,
            "codas": vocab.codas,
            "roles": vocab.roles,
        }, f, indent=2)

    export_vocab_index(vocab, args.outdir)

    print(f"Exported {len(manifest['files'])} CSV files to {args.outdir}")
    print(f"Wrote manifest.json, vocab.json, and vocab_index.txt")
    print("This export is aligned with the current chunked/stateful desktop model.")


if __name__ == "__main__":
    main()
