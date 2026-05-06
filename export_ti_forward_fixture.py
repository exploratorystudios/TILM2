"""
Export a calculator-side fixture for TILM2 hidden-layer testing.

This writes:
- |LX.csv: encoded context input vector, length 874
- |LI.csv: 1-based context component IDs, length 50, for H1 precompute
- |LHR.csv: expected h1 vector, length 198
- |LMR.csv: expected h2 vector, length 198
- |LRR/|LWR/|LSR/|LNR/|LOR/|LCR: expected output probabilities
- |LNM and |LCM1..|LCM10: phonotactic masks for generation

These are not model parameters. They are temporary test lists for validating
T2H1/T2H2 on calculator before implementing full tokenization/output heads.
"""

import argparse
import json
import os

import numpy as np

from tilm2_model import TILM2, Vocab


def write_list(path: str, values: np.ndarray, precision: int) -> None:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{v:.{precision}f}" for v in flat))


def compute_hidden(model: TILM2, context: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = model._encode_context(context)
    disc_state = x[-model.state_dim:-model.word_state_dim]
    word_state = x[-model.word_state_dim:]

    h1_parts = []
    for i, row_chunks in enumerate(model.W_h_chunks):
        s = i * model.chunk_size
        e = s + row_chunks[0].shape[0]
        pre = model.b_h[s:e].copy()
        for j, chunk in enumerate(row_chunks):
            pre += chunk @ x[model.input_col_starts[j]:model.input_col_starts[j + 1]]
        pre += model.W_disc_h1_chunks[i] @ disc_state + model.W_word_h1_chunks[i] @ word_state
        h1_parts.append(model._relu(pre))
    h1 = np.concatenate(h1_parts)

    h2_parts = []
    for i, row_chunks in enumerate(model.W_h2_chunks):
        s = i * model.chunk_size
        e = s + row_chunks[0].shape[0]
        pre = model.b_h2[s:e].copy()
        for j, chunk in enumerate(row_chunks):
            pre += chunk @ h1[model.h_col_starts[j]:model.h_col_starts[j + 1]]
        pre += model.W_disc_h2_chunks[i] @ disc_state + model.W_word_h2_chunks[i] @ word_state
        h2_parts.append(model._relu(pre))
    h2 = np.concatenate(h2_parts)
    return x, h1, h2


def context_id_list(model: TILM2, context: list[dict]) -> np.ndarray:
    ctx = context[-model.context_len:]
    ids: list[int] = []
    for _ in range(model.context_len - len(ctx)):
        ids.extend([0, 0, 0, 0, 0])
    for tok in ctx:
        oi, ni, ci, si, wb = model.vocab.token_to_indices(tok)
        ids.extend([oi + 1, ni + 1, ci + 1, si + 1, wb + 1])
    return np.array(ids, dtype=np.float64)


def compute_forward_outputs(model: TILM2, context: list[dict], temperature: float) -> dict[str, np.ndarray]:
    out = model.forward(context, temperature=temperature)
    return {
        "|LRR.csv": out["role_probs"],
        "|LWR.csv": out["wb_probs"],
        "|LSR.csv": out["stress_probs"],
        "|LNR.csv": out["nucleus_probs"],
        "|LOR.csv": out["onset_probs"],
        "|LCR.csv": out["coda_probs"],
    }


def compute_masks(vocab: Vocab, windows: list[dict], min_freq: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((vocab.n_onsets, vocab.n_nuclei, vocab.n_codas), dtype=np.int32)
    for w in windows:
        for tok in w["context"] + [w["target"]]:
            oi, ni, ci, *_ = vocab.token_to_indices(tok)
            counts[oi, ni, ci] += 1
    cod_mask = (counts >= min_freq).astype(np.float64)
    nuc_mask = cod_mask.any(axis=2).astype(np.float64)
    return nuc_mask.reshape(-1), cod_mask.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", default="ti_packed_export")
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=21)
    parser.add_argument("--hidden-dim", type=int, default=198)
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    vocab_dict = dict(data["vocab"])
    if "role_vocab" in data:
        vocab_dict["role_vocab"] = data["role_vocab"]
    vocab = Vocab(vocab_dict)
    model = TILM2(vocab, context_len=args.context_len, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
    model.load(args.weights)

    context = data["windows"][args.window_index]["context"][-model.context_len:]
    x, h1, h2 = compute_hidden(model, context)
    context_ids = context_id_list(model, context)
    probs = compute_forward_outputs(model, context, temperature=args.temperature)
    nuc_mask, cod_mask = compute_masks(vocab, data["windows"], min_freq=args.min_freq)

    os.makedirs(args.outdir, exist_ok=True)
    write_list(os.path.join(args.outdir, "|LX.csv"), x, args.precision)
    write_list(os.path.join(args.outdir, "|LI.csv"), context_ids, 0)
    write_list(os.path.join(args.outdir, "|LHR.csv"), h1, args.precision)
    write_list(os.path.join(args.outdir, "|LMR.csv"), h2, args.precision)
    for filename, values in probs.items():
        write_list(os.path.join(args.outdir, filename), values, args.precision)
    write_list(os.path.join(args.outdir, "|LNM.csv"), nuc_mask, 0)
    for i in range(10):
        start = i * 999
        end = min(start + 999, len(cod_mask))
        write_list(os.path.join(args.outdir, f"|LCM{i + 1}.csv"), cod_mask[start:end], 0)

    print(f"Wrote hidden forward fixture to {args.outdir}")
    print(f"  |LX  {len(x)} values")
    print(f"  |LI  {len(context_ids)} values")
    print(f"  |LHR {len(h1)} values")
    print(f"  |LMR {len(h2)} values")
    for filename, values in probs.items():
        print(f"  {filename[:-4]} {len(values)} values")
    print(f"  |LNM {len(nuc_mask)} values")
    print("  |LCM1..|LCM10", len(cod_mask), "values")
    print("  first h2 values:", ", ".join(f"{v:.6f}" for v in h2[:5]))


if __name__ == "__main__":
    main()
