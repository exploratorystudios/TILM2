"""
Export a deterministic first-pass reference fixture for TILM2 calculator debugging.

This writes the PC-computed output-head probabilities for a seed context. The
calculator can then compare its own `OUT` program against these fixed reference
lists without depending on random sampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from syllabifier import load_cmu_dict, tokenize_text
from tilm2_model import TILM2, Vocab


def write_list(path: Path, values: np.ndarray, precision: int) -> None:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    path.write_text("\n".join(f"{v:.{precision}f}" for v in flat) + "\n", encoding="utf-8")


def deterministic_distributions(model: TILM2, seed_context: list[dict], temperature: float) -> dict[str, np.ndarray]:
    context = list(seed_context)[-model.context_len:]
    out = model.forward(context, temperature=temperature)
    return {
        "|LDRR.csv": out["role_probs"],
        "|LDWR.csv": out["wb_probs"],
        "|LDSR.csv": out["stress_probs"],
        "|LDNR.csv": out["nucleus_probs"],
        "|LDOR.csv": out["onset_probs"],
        "|LDCR.csv": out["coda_probs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed-text", required=True)
    parser.add_argument("--outdir", default="ti_step1_compare_fixture")
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--cmu-dict", default="cmudict.dict")
    parser.add_argument("--allow-missing-cmu", action="store_true")
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=21)
    parser.add_argument("--hidden-dim", type=int, default=198)
    args = parser.parse_args()

    load_cmu_dict(args.cmu_dict, required=not args.allow_missing_cmu)
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    vocab_dict = dict(data["vocab"])
    if "role_vocab" in data:
        vocab_dict["role_vocab"] = data["role_vocab"]
    vocab = Vocab(vocab_dict)
    model = TILM2(vocab, context_len=args.context_len, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
    model.load(args.weights)
    model.build_phonotactic_masks(data["windows"], min_freq=1, freq_weight=0.0)

    context = tokenize_text(args.seed_text) if args.seed_text else []
    context = context[-model.context_len:]
    probs = deterministic_distributions(model, context, args.temperature)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for filename, values in probs.items():
        write_list(outdir / filename, values, args.precision)

    meta = {
        "seed_text": args.seed_text,
        "temperature": args.temperature,
        "context_len": args.context_len,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
    }
    (outdir / "compare_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote step-1 compare fixture to {outdir}")
    for filename, values in probs.items():
        print(f"  {filename[:-4]} {len(values)} values")


if __name__ == "__main__":
    main()
