"""
TILM2 interactive inference — generate syllable sequences from a trained model.

Usage:
    python3 inference.py --weights tilm2_weights.npz --data training_data.json
"""

import json
import argparse
import os
import numpy as np
from tilm2_model import TILM2, Vocab
from syllabifier import (
    load_cmu_dict,
    tokenize_text,
    decode_tokens_to_words,
    build_reverse_cmu_lexicon,
    build_reverse_cmu_lexicon_nostress,
    collect_word_preferences,
    collect_known_words,
    load_project_lexicon,
)


def render_tokens(tokens: list[dict], show_stress: bool = True) -> str:
    stress_mark = {0: "·", 1: "ˈ", 2: "ˌ"}
    parts = []
    for t in tokens:
        syl = t["onset"] + t["nucleus"] + t["coda"]
        if show_stress:
            syl = stress_mark.get(t["stress"], "") + syl
        if t.get("word_boundary") and parts:
            parts.append(" ")
        parts.append(syl)
    return "".join(parts)


def interactive(model: TILM2, context_len: int, decode_lexicon: dict, decode_lexicon_nostress: dict, show_raw: bool, rep_penalty: float = 0.5, rep_window: int = 15) -> None:
    print("\nTILM2 Inference")
    print("  Commands: 'q' to quit, 't <val>' to set temperature, 'k <val>' for top-k")
    print("  Enter seed text (or press Enter for blank seed)\n")

    temperature = 1.0
    top_k = 5
    n_syllables = 30

    while True:
        cmd = input("seed> ").strip()
        if cmd.lower() == "q":
            break
        if cmd.startswith("t "):
            try:
                temperature = float(cmd.split()[1])
                print(f"Temperature set to {temperature}")
            except:
                print("Usage: t <float>")
            continue
        if cmd.startswith("k "):
            try:
                top_k = int(cmd.split()[1])
                print(f"Top-k set to {top_k}")
            except:
                print("Usage: k <int>")
            continue
        if cmd.startswith("n "):
            try:
                n_syllables = int(cmd.split()[1])
                print(f"Syllables to generate: {n_syllables}")
            except:
                print("Usage: n <int>")
            continue

        # Tokenize seed
        if cmd:
            seed_tokens = tokenize_text(cmd)
            seed_tokens = seed_tokens[-context_len:] if seed_tokens else []
        else:
            seed_tokens = []

        generated = model.generate(
            seed_context=seed_tokens,
            n_syllables=n_syllables,
            temperature=temperature,
            top_k=top_k,
            rep_penalty=rep_penalty,
            rep_window=rep_window,
        )

        print()
        print(f"Decoded words:  {decode_tokens_to_words(generated, decode_lexicon, decode_lexicon_nostress)}")
        if show_raw:
            print(f"Raw phonemes:   {render_tokens(generated, show_stress=True)}")
            print(f"Joined phoneme: {model.tokens_to_text(generated)}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data",    required=True, help="training_data.json (for vocab)")
    parser.add_argument("--cmu-dict", default="cmudict.dict")
    parser.add_argument("--allow-missing-cmu", action="store_true",
                        help="Allow rule-based fallback if CMU dict is unavailable")
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--embed-dim",   type=int, default=21)
    parser.add_argument("--min-freq",    type=int,   default=1,
                        help="Min training frequency for a syllable to be allowed (default 1)")
    parser.add_argument("--freq-weight", type=float, default=0.0,
                        help="Log-frequency bonus weight on coda logits; 0.0=binary mask only (default)")
    parser.add_argument("--rep-penalty", type=float, default=0.5,
                        help="Repetition penalty subtracted from logits for recent tokens (default 0.5)")
    parser.add_argument("--rep-window",  type=int,   default=15,
                        help="Number of recent tokens to penalize (default 15)")
    parser.add_argument("--decode-corpus-file", default="corpus/thematic.txt",
                        help="Exact corpus text file used to build the decode vocabulary")
    parser.add_argument("--decode-word-source", action="append", default=[],
                        help="Extra file or directory to include in the exact decode vocabulary")
    parser.add_argument("--decode-lexicon", default="project_lexicon.json",
                        help="Canonical project lexicon JSON. Falls back to rebuilding if missing.")
    parser.add_argument("--show-raw", action="store_true",
                        help="Also show raw phoneme output")
    args = parser.parse_args()

    print("Loading CMU dict...")
    load_cmu_dict(args.cmu_dict, required=not args.allow_missing_cmu)

    print("Loading vocab...")
    with open(args.data) as f:
        data = json.load(f)
    vocab_dict = dict(data["vocab"])
    if "role_vocab" in data:
        vocab_dict["role_vocab"] = data["role_vocab"]
    vocab = Vocab(vocab_dict)

    print("Loading model...")
    model = TILM2(vocab, context_len=args.context_len, embed_dim=args.embed_dim)
    model.load(args.weights)

    print("Building phonotactic masks...")
    model.build_phonotactic_masks(data["windows"],
                                  min_freq=args.min_freq,
                                  freq_weight=args.freq_weight)
    if args.decode_lexicon and os.path.exists(args.decode_lexicon):
        decode_lexicon, decode_lexicon_nostress, _ = load_project_lexicon(args.decode_lexicon)
    else:
        vocab_sources = []
        if args.decode_corpus_file:
            vocab_sources.append(args.decode_corpus_file)
        vocab_sources.extend(args.decode_word_source)
        preferred_words = collect_word_preferences(vocab_sources) if vocab_sources else None
        allowed_words = set(collect_known_words(vocab_sources)) if vocab_sources else None
        decode_lexicon = build_reverse_cmu_lexicon(preferred_words=preferred_words, allowed_words=allowed_words)
        decode_lexicon_nostress = build_reverse_cmu_lexicon_nostress(preferred_words=preferred_words, allowed_words=allowed_words)

    interactive(model, args.context_len, decode_lexicon, decode_lexicon_nostress, args.show_raw,
                rep_penalty=args.rep_penalty, rep_window=args.rep_window)


if __name__ == "__main__":
    main()
