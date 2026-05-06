"""
Syllabifier with stress marking for TILM2.

Converts raw text into (syllable, stress) token pairs using a rule-based
onset-nucleus-coda decomposition plus CMU dict stress patterns where available.

Stress levels:
    0 = unstressed
    1 = primary stress
    2 = secondary stress
"""

import re
import os
import json
from collections import Counter

# ---------------------------------------------------------------------------
# CMU Pronouncing Dictionary loader
# ---------------------------------------------------------------------------

CMU_DICT: dict[str, list[str]] = {}

def load_cmu_dict(path: str = "cmudict.dict", required: bool = False) -> bool:
    CMU_DICT.clear()
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"CMU dictionary not found at {path}. "
                "Download cmudict.dict and pass --cmu-dict /path/to/cmudict.dict."
            )
        return False
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            parts = line.split()
            word = parts[0].lower().rstrip("(0123456789)")
            phones = parts[1:]
            if word not in CMU_DICT:
                CMU_DICT[word] = phones
    if required and not CMU_DICT:
        raise ValueError(
            f"CMU dictionary at {path} loaded zero entries. "
            "Check that the file is a valid cmudict.dict."
        )
    return bool(CMU_DICT)

# ---------------------------------------------------------------------------
# Phoneme → onset / nucleus / coda mappings
# ---------------------------------------------------------------------------

VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY",
    "IH", "IY",
    "OW", "OY",
    "UH", "UW",
}

STRESS_MARKER = re.compile(r"(\d)$")

def phones_to_syllables(phones: list[str]) -> list[dict]:
    """
    Split a CMU phone list into syllables.
    Each syllable: {"onset": str, "nucleus": str, "coda": str, "stress": int}
    """
    syllables = []
    current_onset = []
    i = 0
    while i < len(phones):
        phone = phones[i]
        m = STRESS_MARKER.search(phone)
        if m:
            # This is a vowel — nucleus of a new syllable
            stress = int(m.group(1))
            nucleus = STRESS_MARKER.sub("", phone)
            coda = []
            j = i + 1
            # Collect coda consonants up to next vowel or end
            while j < len(phones) and STRESS_MARKER.sub("", phones[j]) not in VOWELS:
                # Check ahead: if next phone is a vowel, this consonant is onset of next syllable
                if j + 1 < len(phones) and STRESS_MARKER.search(phones[j + 1]):
                    break  # leave for next onset
                coda.append(STRESS_MARKER.sub("", phones[j]))
                j += 1
            syllables.append({
                "onset":   "".join(current_onset) if current_onset else "",
                "nucleus": nucleus,
                "coda":    "".join(coda),
                "stress":  stress,
            })
            current_onset = []
            i = j
        else:
            # Consonant — accumulate as onset for next vowel
            current_onset.append(phone)
            i += 1
    return syllables


# ---------------------------------------------------------------------------
# Rule-based fallback syllabifier (no CMU dict required)
# ---------------------------------------------------------------------------

VOWEL_CHARS = set("aeiouAEIOU")

def rule_based_syllabify(word: str) -> list[str]:
    """
    Very simple rule-based English syllabification.
    Returns list of syllable strings (no stress info).
    """
    word = word.lower()
    syllables = []
    current = ""
    i = 0
    while i < len(word):
        ch = word[i]
        current += ch
        if ch in VOWEL_CHARS:
            # Look ahead: if next chars are consonant(s) then vowel, split before last consonant
            j = i + 1
            cons = ""
            while j < len(word) and word[j] not in VOWEL_CHARS:
                cons += word[j]
                j += 1
            if j < len(word):
                # There's another vowel ahead
                if len(cons) >= 2:
                    current += cons[0]
                    syllables.append(current)
                    current = cons[1:]
                    i = j
                    continue
                else:
                    syllables.append(current)
                    current = ""
                    i = i + 1
                    continue
        i += 1
    if current:
        if syllables:
            syllables[-1] += current
        else:
            syllables.append(current)
    return syllables if syllables else [word]


def rule_stress(syllables: list[str]) -> list[int]:
    """
    Heuristic English stress: primary on penultimate for 2+ syllables,
    first for monosyllables.
    """
    n = len(syllables)
    if n == 1:
        return [1]
    stresses = [0] * n
    stresses[-2] = 1  # penultimate primary (common English pattern)
    return stresses


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tokenize_word(word: str) -> list[dict]:
    """
    Returns a list of syllable dicts for a single word.
    Uses CMU dict if available, otherwise rule-based.
    """
    key = re.sub(r"[^a-z]", "", word.lower())
    if not key:
        return []

    if key in CMU_DICT:
        phones = CMU_DICT[key]
        sylls = phones_to_syllables(phones)
        if sylls:
            return sylls

    # Fallback
    raw_sylls = rule_based_syllabify(key)
    stresses = rule_stress(raw_sylls)
    result = []
    for syll, stress in zip(raw_sylls, stresses):
        # Decompose syll string into onset/nucleus/coda
        nucleus_pos = next((i for i, c in enumerate(syll) if c in VOWEL_CHARS), None)
        if nucleus_pos is None:
            # All consonants — treat whole thing as coda of previous or skip
            result.append({"onset": syll, "nucleus": "", "coda": "", "stress": stress})
            continue
        onset = syll[:nucleus_pos]
        rest = syll[nucleus_pos:]
        # Split rest at last vowel run
        nuc = ""
        coda = ""
        in_nuc = True
        for c in rest:
            if in_nuc and c in VOWEL_CHARS:
                nuc += c
            else:
                in_nuc = False
                coda += c
        result.append({"onset": onset, "nucleus": nuc, "coda": coda, "stress": stress})
    return result


def tokenize_text(text: str) -> list[dict]:
    """
    Tokenize a full text string into a flat list of syllable token dicts.
    Each dict: {"onset", "nucleus", "coda", "stress", "word_boundary": bool}
    """
    tokens = []
    words = re.findall(r"[a-zA-Z']+", text)
    for word in words:
        sylls = tokenize_word(word)
        for idx, s in enumerate(sylls):
            s["word_boundary"] = (idx == 0)
            tokens.append(s)
    return tokens


def token_signature(token: dict) -> tuple[str, str, str, int]:
    return (
        token["onset"],
        token["nucleus"],
        token["coda"],
        int(token.get("stress", 0)),
    )


def token_signature_nostress(token: dict) -> tuple[str, str, str]:
    return (
        token["onset"],
        token["nucleus"],
        token["coda"],
    )


def split_words(tokens: list[dict]) -> list[list[dict]]:
    words: list[list[dict]] = []
    current: list[dict] = []
    for tok in tokens:
        if tok.get("word_boundary", False) and current:
            words.append(current)
            current = []
        current.append(tok)
    if current:
        words.append(current)
    return words


def render_phoneme_word(tokens: list[dict], show_stress: bool = False) -> str:
    stress_mark = {0: "", 1: "ˈ", 2: "ˌ"}
    parts = []
    for tok in tokens:
        syl = tok["onset"] + tok["nucleus"] + tok["coda"]
        if show_stress:
            syl = stress_mark.get(int(tok.get("stress", 0)), "") + syl
        parts.append(syl)
    return "".join(parts)


def render_unknown_word(tokens: list[dict]) -> str:
    return "[" + render_phoneme_word(tokens, show_stress=False).lower() + "]"


def collect_word_preferences(paths: list[str]) -> Counter:
    counts: Counter = Counter()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                if os.path.isfile(fpath) and fname.endswith((".txt", ".md")):
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        counts.update(re.findall(r"[a-zA-Z']+", f.read().lower()))
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            counts.update(re.findall(r"[a-zA-Z']+", f.read().lower()))
    return counts


def collect_known_words(paths: list[str]) -> list[str]:
    counts = collect_word_preferences(paths)
    return sorted(counts)


def signature_to_key(sig: tuple[tuple[str, str, str, int], ...]) -> str:
    return "|".join(f"{o},{n},{c},{s}" for o, n, c, s in sig)


def signature_nostress_to_key(sig: tuple[tuple[str, str, str], ...]) -> str:
    return "|".join(f"{o},{n},{c}" for o, n, c in sig)


def build_reverse_cmu_lexicon(
    max_syllables: int = 4,
    preferred_words: Counter | None = None,
    allowed_words: set[str] | None = None,
) -> dict[tuple[tuple[str, str, str, int], ...], str]:
    lexicon: dict[tuple[tuple[str, str, str, int], ...], str] = {}
    for word in sorted(CMU_DICT):
        if allowed_words is not None and word not in allowed_words:
            continue
        sylls = tokenize_word(word)
        if not sylls or len(sylls) > max_syllables:
            continue
        key = tuple(token_signature(tok) for tok in sylls)
        existing = lexicon.get(key)
        if existing is None:
            lexicon[key] = word
            continue
        word_pref = preferred_words.get(word, 0) if preferred_words else 0
        existing_pref = preferred_words.get(existing, 0) if preferred_words else 0
        if word_pref > existing_pref:
            lexicon[key] = word
        elif word_pref == existing_pref and len(word) < len(existing):
            lexicon[key] = word
    return lexicon


def build_reverse_cmu_lexicon_nostress(
    max_syllables: int = 4,
    preferred_words: Counter | None = None,
    allowed_words: set[str] | None = None,
) -> dict[tuple[tuple[str, str, str], ...], str]:
    lexicon: dict[tuple[tuple[str, str, str], ...], str] = {}
    for word in sorted(CMU_DICT):
        if allowed_words is not None and word not in allowed_words:
            continue
        sylls = tokenize_word(word)
        if not sylls or len(sylls) > max_syllables:
            continue
        key = tuple(token_signature_nostress(tok) for tok in sylls)
        existing = lexicon.get(key)
        if existing is None:
            lexicon[key] = word
            continue
        word_pref = preferred_words.get(word, 0) if preferred_words else 0
        existing_pref = preferred_words.get(existing, 0) if preferred_words else 0
        if word_pref > existing_pref:
            lexicon[key] = word
        elif word_pref == existing_pref and len(word) < len(existing):
            lexicon[key] = word
    return lexicon


def build_project_lexicon_data(
    vocab_sources: list[str],
    max_syllables: int = 4,
) -> dict:
    preferred_words = collect_word_preferences(vocab_sources)
    allowed_words = set(collect_known_words(vocab_sources))
    lexicon = build_reverse_cmu_lexicon(
        max_syllables=max_syllables,
        preferred_words=preferred_words,
        allowed_words=allowed_words,
    )
    lexicon_nostress = build_reverse_cmu_lexicon_nostress(
        max_syllables=max_syllables,
        preferred_words=preferred_words,
        allowed_words=allowed_words,
    )
    words = {}
    for word in sorted(allowed_words):
        sylls = tokenize_word(word)
        if sylls:
            words[word] = sylls
    return {
        "vocab_sources": vocab_sources,
        "max_syllables": max_syllables,
        "words": words,
        "decode_lexicon": {signature_to_key(sig): word for sig, word in lexicon.items()},
        "decode_lexicon_nostress": {signature_nostress_to_key(sig): word for sig, word in lexicon_nostress.items()},
    }


def save_project_lexicon(path: str, vocab_sources: list[str], max_syllables: int = 4) -> dict:
    data = build_project_lexicon_data(vocab_sources, max_syllables=max_syllables)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


def load_project_lexicon(path: str) -> tuple[dict[str, str], dict[str, str], dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["decode_lexicon"], data["decode_lexicon_nostress"], data


def decode_tokens_to_words(
    tokens: list[dict],
    lexicon: dict[tuple[tuple[str, str, str, int], ...], str] | None = None,
    lexicon_nostress: dict[tuple[tuple[str, str, str], ...], str] | None = None,
) -> str:
    if lexicon is None:
        lexicon = build_reverse_cmu_lexicon()
    if lexicon_nostress is None:
        lexicon_nostress = build_reverse_cmu_lexicon_nostress()
    words = []
    for word_tokens in split_words(tokens):
        key = tuple(token_signature(tok) for tok in word_tokens)
        decoded = lexicon.get(key)
        if decoded is None and key:
            decoded = lexicon.get(signature_to_key(key))
        if decoded is None:
            key_nostress = tuple(token_signature_nostress(tok) for tok in word_tokens)
            decoded = lexicon_nostress.get(key_nostress)
            if decoded is None and key_nostress:
                decoded = lexicon_nostress.get(signature_nostress_to_key(key_nostress))
        if decoded is None:
            decoded = render_unknown_word(word_tokens)
        words.append(decoded)
    return " ".join(words)


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

def build_vocab(token_lists: list[list[dict]], max_onsets: int = 90, max_codas: int = 90) -> dict:
    """
    Build onset/nucleus/coda vocabularies from a list of token sequences.
    Codas are capped at the top max_codas most frequent to stay within the
    TI-84 99×99 matrix limit. Rare codas map to "" (index 0) at runtime.
    Returns {"onsets": [...], "nuclei": [...], "codas": [...]}
    """
    from collections import Counter
    onset_counts: Counter = Counter()
    nuclei: set = set()
    coda_counts: Counter = Counter()
    for tokens in token_lists:
        for t in tokens:
            onset_counts[t["onset"]] += 1
            nuclei.add(t["nucleus"])
            coda_counts[t["coda"]] += 1

    top_onsets = [o for o, _ in onset_counts.most_common(max_onsets)]
    if "" not in top_onsets:
        top_onsets = [""] + top_onsets[:max_onsets - 1]

    top_codas = [c for c, _ in coda_counts.most_common(max_codas)]
    if "" not in top_codas:
        top_codas = [""] + top_codas[:max_codas - 1]

    return {
        "onsets": sorted(top_onsets),
        "nuclei": sorted(nuclei),
        "codas":  sorted(top_codas),
    }


if __name__ == "__main__":
    load_cmu_dict()
    sample = "The syllabifier converts language into phonemic tokens with stress markings"
    tokens = tokenize_text(sample)
    print(f"{'ONSET':<10} {'NUCLEUS':<10} {'CODA':<10} {'STRESS'}")
    print("-" * 45)
    for t in tokens:
        print(f"{t['onset']:<10} {t['nucleus']:<10} {t['coda']:<10} {t['stress']}")
