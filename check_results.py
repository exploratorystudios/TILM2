"""
Heuristic to check generated results against corpus.

Reports:
  - n-gram overlap rates (how much of the output verbatim matches the corpus)
  - OOV token rate (bracketed tokens like [liy], [mahn])
  - Longest verbatim run per sequence
  - Per-sequence and aggregate summary
"""

import argparse
import re
from collections import defaultdict

OOV_RE = re.compile(r'\[[^\]]+\]')


def load_corpus_ngrams(corpus_path: str, max_n: int = 6) -> dict[int, set]:
    ngrams: dict[int, set] = {n: set() for n in range(2, max_n + 1)}
    with open(corpus_path, encoding="utf-8") as f:
        text = f.read().lower()
    words = re.findall(r"[a-z']+", text)
    for n in range(2, max_n + 1):
        for i in range(len(words) - n + 1):
            ngrams[n].add(tuple(words[i:i+n]))
    return ngrams


def parse_results(results_path: str) -> list[dict]:
    """
    Returns list of {seed, sequences: list[str]}.
    Each sequence is a raw generated line (may include [oov] tokens).
    """
    entries = []
    current_seed = None
    current_seqs = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('[') and line.endswith(']'):
                if current_seed is not None:
                    entries.append({"seed": current_seed, "sequences": current_seqs})
                current_seed = line[1:-1]
                current_seqs = []
            elif re.match(r'\s+\d+:', line):
                seq = re.sub(r'^\s+\d+:\s*', '', line)
                current_seqs.append(seq)
    if current_seed is not None:
        entries.append({"seed": current_seed, "sequences": current_seqs})
    return entries


def sequence_to_words(seq: str) -> list[str]:
    """Strip OOV tokens and return clean word list."""
    clean = OOV_RE.sub('', seq)
    return re.findall(r"[a-z']+", clean.lower())


def oov_rate(seq: str) -> float:
    tokens = seq.strip().split()
    if not tokens:
        return 0.0
    oov_count = sum(1 for t in tokens if OOV_RE.match(t))
    return oov_count / len(tokens)


def ngram_overlap(words: list[str], corpus_ngrams: dict[int, set], n: int) -> float:
    if len(words) < n:
        return 0.0
    hits = sum(
        1 for i in range(len(words) - n + 1)
        if tuple(words[i:i+n]) in corpus_ngrams[n]
    )
    total = len(words) - n + 1
    return hits / total if total > 0 else 0.0


def longest_verbatim_run(words: list[str], corpus_ngrams: dict[int, set]) -> int:
    """Longest consecutive run of words that forms a corpus n-gram at every step."""
    if not words:
        return 0
    max_run = 0
    # Slide a window: find max n such that tuple(words[i:i+n]) is in corpus
    for i in range(len(words)):
        run = 1
        for n in range(2, len(words) - i + 1):
            if tuple(words[i:i+n]) in corpus_ngrams.get(n, set()):
                run = n
            else:
                break
        max_run = max(max_run, run)
    return max_run


def analyze(results_path: str, corpus_path: str, max_n: int = 6) -> None:
    print(f"Loading corpus n-grams (up to {max_n}-gram)...")
    corpus_ngrams = load_corpus_ngrams(corpus_path, max_n)
    for n, s in corpus_ngrams.items():
        print(f"  {n}-grams in corpus: {len(s)}")

    print(f"\nParsing results from {results_path}...")
    entries = parse_results(results_path)
    print(f"  {len(entries)} seeds, {sum(len(e['sequences']) for e in entries)} sequences\n")

    all_oov = []
    all_overlap = defaultdict(list)
    all_runs = []

    print(f"{'Seed':<40} {'Seq':>3}  {'OOV%':>6}  {'2g%':>6}  {'3g%':>6}  {'4g%':>6}  {'5g%':>6}  {'MaxRun':>7}")
    print("-" * 92)

    for entry in entries:
        seed = entry["seed"]
        for i, seq in enumerate(entry["sequences"], 1):
            ov = oov_rate(seq)
            words = sequence_to_words(seq)
            overlaps = {n: ngram_overlap(words, corpus_ngrams, n) for n in range(2, max_n + 1)}
            run = longest_verbatim_run(words, corpus_ngrams)

            all_oov.append(ov)
            for n, v in overlaps.items():
                all_overlap[n].append(v)
            all_runs.append(run)

            seed_display = (seed[:37] + "...") if len(seed) > 40 else seed
            print(
                f"{seed_display:<40} {i:>3}  "
                f"{ov*100:>5.1f}%  "
                f"{overlaps[2]*100:>5.1f}%  "
                f"{overlaps[3]*100:>5.1f}%  "
                f"{overlaps[4]*100:>5.1f}%  "
                f"{overlaps[5]*100:>5.1f}%  "
                f"{run:>7}"
            )

    print("-" * 92)
    n_seqs = len(all_oov)
    print(f"\n{'AVERAGES':<40} {'':>3}  "
          f"{sum(all_oov)/n_seqs*100:>5.1f}%  "
          f"{sum(all_overlap[2])/n_seqs*100:>5.1f}%  "
          f"{sum(all_overlap[3])/n_seqs*100:>5.1f}%  "
          f"{sum(all_overlap[4])/n_seqs*100:>5.1f}%  "
          f"{sum(all_overlap[5])/n_seqs*100:>5.1f}%  "
          f"{sum(all_runs)/n_seqs:>7.1f}")

    print(f"\nInterpretation guide:")
    print(f"  OOV%   — bracketed tokens; lower is better (0% ideal)")
    print(f"  2g%    — bigram overlap with corpus; ~80–100% is normal for fluent output")
    print(f"  3g%    — trigram overlap; >70% suggests heavy corpus reliance")
    print(f"  4g%    — 4-gram overlap; >40% is a regurgitation warning")
    print(f"  5g%    — 5-gram overlap; >20% is strong verbatim copying")
    print(f"  MaxRun — longest exact corpus match; >8 words is suspicious")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.txt")
    parser.add_argument("--corpus",  default="corpus/thematic.txt")
    parser.add_argument("--max-n",   type=int, default=6)
    args = parser.parse_args()
    analyze(args.results, args.corpus, args.max_n)


if __name__ == "__main__":
    main()
