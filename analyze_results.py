"""
Pattern analysis for TILM2 results.txt.
Identifies attractor tokens, loop bigrams, vocabulary coverage,
degenerate runs, and per-seed coherence scores.
"""

import json
from pathlib import Path
import re
import sys
from collections import Counter

RESULTS_FILE = "results.txt"
STRESS = re.compile(r"[ˈˌ·]")
ALPHA = re.compile(r"^[a-z]+$")
BRACKETED = re.compile(r"^\[[^\]]+\]$")

LEXICON_FILE = Path(__file__).with_name("project_lexicon.json")
with open(LEXICON_FILE, encoding="utf-8") as f:
    LEXICON_DATA = json.load(f)
KNOWN_WORDS = set(LEXICON_DATA["words"])

SHARD_HINTS = {
    "af", "wa", "bove", "pened", "tened", "vening", "ter", "lis", "ove", "be",
    "move", "wait", "side",
}

def strip_stress(tok):
    return STRESS.sub("", tok).strip()

def parse_results(path):
    """Return list of (seed, run_idx, [tokens]) tuples."""
    runs = []
    seed = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("[") and line.endswith("]"):
                seed = line[1:-1]
            elif re.match(r"\s+\d+:", line):
                m = re.match(r"\s+(\d+):\s*(.*)", line)
                if m:
                    idx    = int(m.group(1))
                    tokens = [strip_stress(t) for t in m.group(2).split() if strip_stress(t)]
                    runs.append((seed, idx, tokens))
    return runs

def longest_run(tokens):
    """Length of the longest consecutive repeated token."""
    if not tokens:
        return 0, ""
    best_len, best_tok = 1, tokens[0]
    cur_len,  cur_tok  = 1, tokens[0]
    for t in tokens[1:]:
        if t == cur_tok:
            cur_len += 1
            if cur_len > best_len:
                best_len, best_tok = cur_len, cur_tok
        else:
            cur_len, cur_tok = 1, t
    return best_len, best_tok

def repetition_ratio(tokens, window=5):
    """Fraction of positions where the token appeared in the previous `window` tokens."""
    if len(tokens) < 2:
        return 0.0
    hits = 0
    for i in range(1, len(tokens)):
        if tokens[i] in tokens[max(0, i - window):i]:
            hits += 1
    return hits / (len(tokens) - 1)


def suspicious_token(tok):
    if not tok:
        return False
    if BRACKETED.match(tok):
        return True
    if not ALPHA.match(tok):
        return True
    if len(tok) <= 2 and tok not in {"a", "he", "in", "on", "by", "to", "of"}:
        return True
    if tok in SHARD_HINTS and tok not in KNOWN_WORDS:
        return True
    if tok not in KNOWN_WORDS:
        return True
    return False

# ---------------------------------------------------------------------------

runs = parse_results(RESULTS_FILE)
if not runs:
    print("No runs found — check file path.")
    sys.exit(1)

all_tokens   = [t for _, _, toks in runs for t in toks]
all_bigrams  = []
for _, _, toks in runs:
    all_bigrams.extend(zip(toks, toks[1:]))

tok_freq    = Counter(all_tokens)
bigram_freq = Counter(all_bigrams)

total_tokens = len(all_tokens)
unique_toks  = len(tok_freq)

print("=" * 60)
print(f"TILM2 Results Analysis — {len(runs)} runs, {total_tokens} tokens")
print("=" * 60)

# --- Token frequency ---
print(f"\nVocabulary coverage: {unique_toks} unique tokens")
print(f"\nTop 25 tokens (% of all output):")
for tok, cnt in tok_freq.most_common(25):
    bar = "█" * int(40 * cnt / total_tokens)
    print(f"  {tok:<12} {cnt:5d}  {100*cnt/total_tokens:5.1f}%  {bar}")

# --- Bigram frequency ---
print(f"\nTop 20 bigrams (attractor pairs):")
for (a, b), cnt in bigram_freq.most_common(20):
    pct = 100 * cnt / (total_tokens - len(runs))
    print(f"  {a} → {b:<12}  {cnt:4d}  {pct:.1f}%")

# --- Self-loop bigrams (token → same token) ---
self_loops = {tok: cnt for (a, b), cnt in bigram_freq.items() if a == b for tok in [a]}
if self_loops:
    print(f"\nSelf-loop bigrams (X → X):")
    for tok, cnt in sorted(self_loops.items(), key=lambda x: -x[1])[:10]:
        print(f"  {tok} → {tok}  {cnt}x")

# --- Per-run degeneration ---
print(f"\nPer-run degeneration analysis:")
rep_ratios   = []
worst_runs   = []
for seed, idx, toks in runs:
    rr          = repetition_ratio(toks)
    rlen, rtok  = longest_run(toks)
    rep_ratios.append(rr)
    worst_runs.append((rr, rlen, rtok, seed, idx))

avg_rep = sum(rep_ratios) / len(rep_ratios)
print(f"  Average repetition ratio: {avg_rep:.3f}  (0=none, 1=all repeated)")
print(f"  Runs with ratio > 0.5:    {sum(1 for r in rep_ratios if r > 0.5)}/{len(rep_ratios)}")
print(f"  Runs with ratio > 0.7:    {sum(1 for r in rep_ratios if r > 0.7)}/{len(rep_ratios)}")

print(f"\n  5 worst runs (highest repetition):")
for rr, rlen, rtok, seed, idx in sorted(worst_runs, reverse=True)[:5]:
    print(f"    [{seed}] run {idx}  ratio={rr:.2f}  longest_run={rlen}x '{rtok}'")

# --- Corruption / shard analysis ---
print(f"\nMalformed/OOV analysis:")
suspicious = Counter(t for t in all_tokens if suspicious_token(t))
bad_total = sum(suspicious.values())
bad_pct = 100 * bad_total / total_tokens if total_tokens else 0.0
print(f"  Tokens outside project lexicon or bracket-decoded: {bad_total}/{total_tokens}  ({bad_pct:.1f}%)")
print(f"\n  Top malformed/OOV tokens:")
for tok, cnt in suspicious.most_common(20):
    print(f"    {tok:<12} {cnt:4d}x")

bad_runs = []
for seed, idx, toks in runs:
    bad = sum(1 for t in toks if suspicious_token(t))
    ratio = bad / len(toks) if toks else 0.0
    bad_runs.append((ratio, bad, len(toks), seed, idx))

print(f"\n  5 worst runs (highest malformed/OOV ratio):")
for ratio, bad, n, seed, idx in sorted(bad_runs, reverse=True)[:5]:
    print(f"    [{seed}] run {idx}  malformed_or_oov={bad}/{n}  ratio={ratio:.2f}")

# --- Seed-level summary ---
print(f"\nPer-seed average repetition ratio:")
from collections import defaultdict
by_seed = defaultdict(list)
for rr, rlen, rtok, seed, idx in worst_runs:
    by_seed[seed].append(rr)
for seed, ratios in sorted(by_seed.items(), key=lambda x: -sum(x[1])/len(x[1])):
    avg = sum(ratios) / len(ratios)
    bar = "█" * int(20 * avg)
    print(f"  {avg:.2f} {bar}  [{seed}]")

# --- Token class breakdown ---
in_vocab = sum(cnt for tok, cnt in tok_freq.items() if tok in KNOWN_WORDS)
oov = total_tokens - in_vocab
print(f"\nProject-vocab coverage:")
print(f"  In-vocab tokens:  {in_vocab:5d}  {100*in_vocab/total_tokens:.1f}%")
print(f"  OOV/malformed:    {oov:5d}  {100*oov/total_tokens:.1f}%")

print(f"\n  Top OOV/malformed tokens:")
for tok, cnt in suspicious.most_common(20):
    print(f"    {tok:<14} {cnt:4d}x")
