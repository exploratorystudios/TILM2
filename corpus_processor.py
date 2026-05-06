"""
Corpus processor for TILM2.

Reads raw text files, tokenizes them into syllable tokens, applies repetition
weighting for reinforcement, and saves training sequences as JSON.

Usage:
    python3 corpus_processor.py --corpus-dir ./corpus --output training_data.json
    python3 corpus_processor.py --text "some inline text" --output training_data.json
"""

import argparse
import json
import os
import re
import random
from collections import Counter
from syllabifier import load_cmu_dict, tokenize_text, build_vocab

ROLE_OTHER = "other"
ROLE_STATE = "state"
ROLE_MOTION = "motion"
ROLE_PERCEPTION = "perception"
ROLE_NEGATION = "negation"
ROLE_TRANSITION = "transition"

ROLE_ORDER = [
    ROLE_OTHER,
    ROLE_STATE,
    ROLE_MOTION,
    ROLE_PERCEPTION,
    ROLE_NEGATION,
    ROLE_TRANSITION,
]

SENTENCE_SPLIT_RE = re.compile(r"[.!?]+\s+|\n+")
WORD_RE = re.compile(r"[a-zA-Z']+")
NEGATION_PATTERNS = ("did not", "was not", "were not", "never")
PERCEPTION_WORDS = {"looked", "watched", "heard", "listened"}
MOTION_WORDS = {"came", "walked", "moved", "ran", "rose", "fell", "stood"}
STATE_WORDS = {"was", "were", "lay", "shone", "grew"}
TRANSITION_WORDS = {"then", "now", "again", "still"}

# ---------------------------------------------------------------------------
# Repetition / reinforcement weighting
# ---------------------------------------------------------------------------

def compute_repetitions(freq: int, base: int = 1, max_reps: int = 8) -> int:
    """
    Logarithmic repetition scaling: rare sequences repeat more, common cap out.
    freq: how many times this n-gram appeared in the corpus naturally.
    """
    if freq <= 0:
        return max_reps
    import math
    reps = max_reps - int(math.log2(freq + 1))
    return max(base, reps)


def build_ngram_counts(sequences: list[list[dict]], n: int = 3) -> Counter:
    counts: Counter = Counter()
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            key = tuple(
                (t["onset"], t["nucleus"], t["coda"], t["stress"])
                for t in seq[i:i+n]
            )
            counts[key] += 1
    return counts


# ---------------------------------------------------------------------------
# Sequence windowing
# ---------------------------------------------------------------------------

def make_windows(tokens: list[dict], context_len: int = 10) -> list[dict]:
    """
    Slide a window of `context_len` tokens over the sequence.
    Each window: {"context": [token, ...], "target": token}
    """
    windows = []
    for i in range(context_len, len(tokens)):
        windows.append({
            "context": tokens[i - context_len : i],
            "target":  tokens[i],
            "target_role": tokens[i].get("role", ROLE_OTHER),
        })
    return windows


def classify_sentence_role(sentence: str) -> str:
    lower = sentence.lower()
    words = WORD_RE.findall(lower)
    if not words:
        return ROLE_OTHER

    word_set = set(words)
    if any(pattern in lower for pattern in NEGATION_PATTERNS):
        return ROLE_NEGATION
    if word_set & PERCEPTION_WORDS:
        return ROLE_PERCEPTION
    if word_set & TRANSITION_WORDS and word_set & STATE_WORDS:
        return ROLE_TRANSITION
    if word_set & MOTION_WORDS:
        return ROLE_MOTION
    if word_set & STATE_WORDS:
        return ROLE_STATE
    return ROLE_OTHER


def tokenize_text_with_roles(text: str) -> list[dict]:
    tokens: list[dict] = []
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        sentences = [text]
    for sentence in sentences:
        role = classify_sentence_role(sentence)
        for token in tokenize_text(sentence):
            token["role"] = role
            tokens.append(token)
    return tokens


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

def load_texts_from_dir(corpus_dir: str) -> list[str]:
    texts = []
    for fname in sorted(os.listdir(corpus_dir)):
        if fname.endswith((".txt", ".md")):
            fpath = os.path.join(corpus_dir, fname)
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                texts.append(f.read())
    return texts


def load_texts_from_gutenberg(urls: list[str]) -> list[str]:
    """Download plain text from Project Gutenberg URLs."""
    import urllib.request
    texts = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                texts.append(resp.read().decode("utf-8", errors="replace"))
            print(f"  Downloaded: {url}")
        except Exception as e:
            print(f"  Failed {url}: {e}")
    return texts


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_corpus(
    texts: list[str],
    context_len: int = 10,
    repeat_rare: bool = True,
    max_tokens: int | None = None,
    seed: int = 42,
) -> tuple[list[dict], dict, list[str]]:
    """
    Full pipeline: text → syllable tokens → windows → (optionally repeated).
    Returns (windows, vocab, role_vocab).
    """
    random.seed(seed)

    print("Tokenizing texts...")
    all_token_seqs: list[list[dict]] = []
    role_counts: Counter = Counter()
    for idx, text in enumerate(texts):
        tokens = tokenize_text_with_roles(text)
        if tokens:
            all_token_seqs.append(tokens)
            for token in tokens:
                role_counts[token.get("role", ROLE_OTHER)] += 1
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(texts)} texts tokenized")

    print("Building vocabulary...")
    vocab = build_vocab(all_token_seqs)
    print(f"  Onsets:  {len(vocab['onsets'])}")
    print(f"  Nuclei:  {len(vocab['nuclei'])}")
    print(f"  Codas:   {len(vocab['codas'])}")

    # Trim to max_tokens if requested
    if max_tokens is not None:
        flat = [t for seq in all_token_seqs for t in seq]
        flat = flat[:max_tokens]
        all_token_seqs = [flat]

    print("Building n-gram counts for repetition weighting...")
    ngram_counts = build_ngram_counts(all_token_seqs, n=3)

    print("Windowing sequences...")
    all_windows: list[dict] = []
    for seq in all_token_seqs:
        all_windows.extend(make_windows(seq, context_len=context_len))

    if repeat_rare:
        print("Applying reinforcement repetition...")
        repeated: list[dict] = []
        for w in all_windows:
            key = tuple(
                (t["onset"], t["nucleus"], t["coda"], t["stress"])
                for t in w["context"][-3:]
            )
            freq = ngram_counts.get(key, 1)
            reps = compute_repetitions(freq)
            for _ in range(reps):
                repeated.append(w)
        random.shuffle(repeated)
        all_windows = repeated

    print(f"Total training windows: {len(all_windows)}")
    role_vocab = [role for role in ROLE_ORDER if role_counts.get(role, 0) > 0]
    if ROLE_OTHER not in role_vocab:
        role_vocab.insert(0, ROLE_OTHER)
    return all_windows, vocab, role_vocab


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

GUTENBERG_DEFAULTS = [
    # Pride and Prejudice
    "https://www.gutenberg.org/files/1342/1342-0.txt",
    # Moby Dick
    "https://www.gutenberg.org/files/2701/2701-0.txt",
    # A Tale of Two Cities
    "https://www.gutenberg.org/files/98/98-0.txt",
    # The Adventures of Sherlock Holmes
    "https://www.gutenberg.org/files/1661/1661-0.txt",
    # Alice's Adventures in Wonderland
    "https://www.gutenberg.org/files/11/11-0.txt",
]


def main():
    parser = argparse.ArgumentParser(description="TILM2 corpus processor")
    parser.add_argument("--corpus-dir", default=None, help="Directory of .txt files")
    parser.add_argument("--text", default=None, help="Inline text string")
    parser.add_argument("--gutenberg", action="store_true", help="Download default Gutenberg texts")
    parser.add_argument("--cmu-dict", default="cmudict.dict", help="Path to CMU dict file")
    parser.add_argument("--allow-missing-cmu", action="store_true",
                        help="Allow rule-based fallback if CMU dict is unavailable")
    parser.add_argument("--output", default="training_data.json", help="Output JSON file")
    parser.add_argument("--context-len", type=int, default=10, help="Syllable context window size")
    parser.add_argument("--max-tokens", type=int, default=None, help="Cap total tokens")
    parser.add_argument("--no-repeat", action="store_true", help="Disable reinforcement repetition")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading CMU dict...")
    load_cmu_dict(args.cmu_dict, required=not args.allow_missing_cmu)

    texts = []
    if args.text:
        texts.append(args.text)
    if args.corpus_dir:
        print(f"Loading texts from {args.corpus_dir}...")
        texts.extend(load_texts_from_dir(args.corpus_dir))
    if args.gutenberg:
        print("Downloading Gutenberg texts...")
        texts.extend(load_texts_from_gutenberg(GUTENBERG_DEFAULTS))
    if not texts:
        print("No input provided. Use --text, --corpus-dir, or --gutenberg.")
        return

    windows, vocab, role_vocab = process_corpus(
        texts,
        context_len=args.context_len,
        repeat_rare=not args.no_repeat,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    output = {
        "vocab": vocab,
        "role_vocab": role_vocab,
        "context_len": args.context_len,
        "num_windows": len(windows),
        "windows": windows,
    }
    with open(args.output, "w") as f:
        json.dump(output, f)
    print(f"Saved {len(windows)} training windows to {args.output}")
    print(f"Vocab sizes — onsets: {len(vocab['onsets'])}, nuclei: {len(vocab['nuclei'])}, codas: {len(vocab['codas'])}")


if __name__ == "__main__":
    main()
