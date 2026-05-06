"""
Run inference from a TI-shaped packed export.

This script intentionally loads model parameters from ti_packed_export CSV files
instead of from the original .npz checkpoint. The optional --compare-weights path
loads the desktop model only as a parity reference.
"""

import argparse
import json
import os

import numpy as np

from syllabifier import (
    build_reverse_cmu_lexicon,
    build_reverse_cmu_lexicon_nostress,
    collect_known_words,
    collect_word_preferences,
    decode_tokens_to_words,
    load_cmu_dict,
    load_project_lexicon,
    tokenize_text,
)
from tilm2_model import TILM2, Vocab
from ti_packed_runtime import PackedTILM2, load_model, load_packed_store


def render(tokens: list[dict]) -> str:
    stress_mark = {0: ".", 1: "'", 2: ","}
    parts = []
    for t in tokens:
        syl = stress_mark.get(t["stress"], "") + t["onset"] + t["nucleus"] + t["coda"]
        if t.get("word_boundary") and parts:
            parts.append(" ")
        parts.append(syl)
    return "".join(parts)


def load_vocab_from_export(exportdir: str) -> Vocab:
    with open(os.path.join(exportdir, "vocab.json"), encoding="utf-8") as f:
        return Vocab(json.load(f))


def build_packed_model(exportdir: str, vocab: Vocab, args: argparse.Namespace) -> PackedTILM2:
    store = load_packed_store(exportdir)
    hparams = {
        "context_len": args.context_len,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "discourse_state_dim": args.discourse_state_dim,
        "word_state_dim": args.word_state_dim,
        "chunk_size": args.chunk_size,
        "n_chunks": (args.hidden_dim + args.chunk_size - 1) // args.chunk_size,
        "n_input_col_chunks": (args.input_dim + args.chunk_size - 1) // args.chunk_size,
        "n_h_col_chunks": (args.hidden_dim + args.chunk_size - 1) // args.chunk_size,
    }
    return PackedTILM2(vocab, store, hparams)


def load_decode_lexicons(args: argparse.Namespace):
    if args.decode_lexicon and os.path.exists(args.decode_lexicon):
        return load_project_lexicon(args.decode_lexicon)[:2]

    vocab_sources = []
    if args.decode_corpus_file:
        vocab_sources.append(args.decode_corpus_file)
    vocab_sources.extend(args.decode_word_source)
    preferred_words = collect_word_preferences(vocab_sources) if vocab_sources else None
    allowed_words = set(collect_known_words(vocab_sources)) if vocab_sources else None
    return (
        build_reverse_cmu_lexicon(preferred_words=preferred_words, allowed_words=allowed_words),
        build_reverse_cmu_lexicon_nostress(preferred_words=preferred_words, allowed_words=allowed_words),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exportdir", default="ti_packed_export")
    parser.add_argument("--data", required=True, help="training_data.json for phonotactic masks")
    parser.add_argument("--seed-text", default="")
    parser.add_argument("--n-syllables", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rep-penalty", type=float, default=0.5)
    parser.add_argument("--rep-window", type=int, default=15)
    parser.add_argument("--rng-seed", type=int, default=123)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--freq-weight", type=float, default=0.0)
    parser.add_argument("--cmu-dict", default="cmudict.dict")
    parser.add_argument("--allow-missing-cmu", action="store_true")
    parser.add_argument("--decode-corpus-file", default="corpus/thematic.txt")
    parser.add_argument("--decode-word-source", action="append", default=[])
    parser.add_argument("--decode-lexicon", default="project_lexicon.json")
    parser.add_argument("--compare-weights", default=None)
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=21)
    parser.add_argument("--hidden-dim", type=int, default=198)
    parser.add_argument("--chunk-size", type=int, default=99)
    parser.add_argument("--discourse-state-dim", type=int, default=16)
    parser.add_argument("--word-state-dim", type=int, default=8)
    args = parser.parse_args()

    args.input_dim = args.context_len * (4 * args.embed_dim + 1) + args.discourse_state_dim + args.word_state_dim

    load_cmu_dict(args.cmu_dict, required=not args.allow_missing_cmu)
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)

    vocab = load_vocab_from_export(args.exportdir)
    packed = build_packed_model(args.exportdir, vocab, args)
    packed.build_phonotactic_masks(data["windows"], min_freq=args.min_freq, freq_weight=args.freq_weight)

    decode_lexicon, decode_lexicon_nostress = load_decode_lexicons(args)
    ctx = tokenize_text(args.seed_text) if args.seed_text else []
    ctx = ctx[-args.context_len:]

    np.random.seed(args.rng_seed)
    generated = packed.generate(
        ctx,
        n_syllables=args.n_syllables,
        temperature=args.temperature,
        top_k=args.top_k,
        rep_penalty=args.rep_penalty,
        rep_window=args.rep_window,
    )

    print("Packed export inference")
    print(f"seed: {args.seed_text}")
    print(f"decoded: {decode_tokens_to_words(generated, decode_lexicon, decode_lexicon_nostress)}")
    print(f"raw: {render(generated)}")

    if args.compare_weights:
        model, _, _ = load_model(args.compare_weights, args.data, args.context_len, args.embed_dim, args.hidden_dim)
        model.build_phonotactic_masks(data["windows"], min_freq=args.min_freq, freq_weight=args.freq_weight)
        np.random.seed(args.rng_seed)
        ref = model.generate(
            ctx,
            n_syllables=args.n_syllables,
            temperature=args.temperature,
            top_k=args.top_k,
            rep_penalty=args.rep_penalty,
            rep_window=args.rep_window,
        )
        print("\nDesktop checkpoint comparison")
        print(f"decoded: {decode_tokens_to_words(ref, decode_lexicon, decode_lexicon_nostress)}")
        print(f"raw: {render(ref)}")
        print(f"exact token match: {generated == ref}")


if __name__ == "__main__":
    main()
