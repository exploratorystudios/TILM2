"""
Batch inference for TILM2 — run a text file of seeds (one per line) and
write completed outputs to a results file.

Usage:
    python3 batch_inference.py --weights tilm2_weights.npz --data training_data.json \
        --seeds seeds.txt --output results.txt
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


def render(tokens: list[dict]) -> str:
    stress_mark = {0: "·", 1: "ˈ", 2: "ˌ"}
    parts = []
    for t in tokens:
        syl = stress_mark.get(t["stress"], "") + t["onset"] + t["nucleus"] + t["coda"]
        if t.get("word_boundary") and parts:
            parts.append(" ")
        parts.append(syl)
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",     required=True)
    parser.add_argument("--data",        required=True, help="training_data.json (for vocab)")
    parser.add_argument("--seeds",       required=True, help="text file with one seed per line")
    parser.add_argument("--output",      default="results.txt")
    parser.add_argument("--cmu-dict",    default="cmudict.dict")
    parser.add_argument("--allow-missing-cmu", action="store_true",
                        help="Allow rule-based fallback if CMU dict is unavailable")
    parser.add_argument("--context-len", type=int,   default=10)
    parser.add_argument("--embed-dim",   type=int,   default=21)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k",       type=int,   default=3)
    parser.add_argument("--n-syllables", type=int,   default=30)
    parser.add_argument("--runs",        type=int,   default=1,
                        help="Number of completions per seed (default 1)")
    parser.add_argument("--min-freq",    type=int,   default=1)
    parser.add_argument("--freq-weight", type=float, default=0.0)
    parser.add_argument("--rep-penalty", type=float, default=0.5,
                        help="Repetition penalty for recent tokens (default 0.5)")
    parser.add_argument("--rep-window",  type=int,   default=15,
                        help="Number of recent tokens to penalize (default 15)")
    parser.add_argument("--seed",        type=int,   default=None,
                        help="NumPy random seed for reproducibility")
    parser.add_argument("--decode-corpus-file", default="corpus/thematic.txt",
                        help="Exact corpus text file used to build the decode vocabulary")
    parser.add_argument("--decode-word-source", action="append", default=[],
                        help="Extra file or directory to include in the exact decode vocabulary")
    parser.add_argument("--decode-lexicon", default="project_lexicon.json",
                        help="Canonical project lexicon JSON. Falls back to rebuilding if missing.")
    parser.add_argument("--show-raw", action="store_true",
                        help="Include raw phoneme lines in the output file")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    load_cmu_dict(args.cmu_dict, required=not args.allow_missing_cmu)

    with open(args.data) as f:
        data = json.load(f)
    vocab_dict = dict(data["vocab"])
    if "role_vocab" in data:
        vocab_dict["role_vocab"] = data["role_vocab"]
    vocab = Vocab(vocab_dict)

    model = TILM2(vocab, context_len=args.context_len, embed_dim=args.embed_dim)
    model.load(args.weights)
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
        preferred_words = collect_word_preferences(vocab_sources)
        allowed_words = set(collect_known_words(vocab_sources))
        decode_lexicon = build_reverse_cmu_lexicon(preferred_words=preferred_words, allowed_words=allowed_words)
        decode_lexicon_nostress = build_reverse_cmu_lexicon_nostress(preferred_words=preferred_words, allowed_words=allowed_words)

    with open(args.seeds) as f:
        seeds = [line.rstrip("\n") for line in f if line.strip()]

    lines_out = []
    for seed in seeds:
        ctx = tokenize_text(seed) if seed else []
        ctx = ctx[-args.context_len:]

        if args.runs == 1:
            gen = model.generate(ctx, n_syllables=args.n_syllables,
                                 temperature=args.temperature, top_k=args.top_k,
                                 rep_penalty=args.rep_penalty, rep_window=args.rep_window)
            lines_out.append(f"[{seed}]")
            lines_out.append(f"  {decode_tokens_to_words(gen, decode_lexicon, decode_lexicon_nostress)}")
            if args.show_raw:
                lines_out.append(f"  raw: {render(gen)}")
        else:
            lines_out.append(f"[{seed}]")
            for i in range(args.runs):
                gen = model.generate(ctx, n_syllables=args.n_syllables,
                                     temperature=args.temperature, top_k=args.top_k)
                lines_out.append(f"  {i+1}: {decode_tokens_to_words(gen, decode_lexicon, decode_lexicon_nostress)}")
                if args.show_raw:
                    lines_out.append(f"    raw: {render(gen)}")
        lines_out.append("")

    with open(args.output, "w") as f:
        f.write("\n".join(lines_out))

    print(f"Wrote {len(seeds)} seed(s) to {args.output}")


if __name__ == "__main__":
    main()
