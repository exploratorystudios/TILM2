"""
Corpus cleaner for TILM2.

Usage:
    python3 clean_corpus.py --input corpus.txt --output corpus_clean.txt
    python3 clean_corpus.py --input corpus.txt --output corpus_clean.txt --report
"""

import re
import argparse


# ---------------------------------------------------------------------------
# Patterns to strip
# ---------------------------------------------------------------------------

# Chapter header lines: "Chapter I", "Chapter XII", etc.
RE_CHAPTER_HEADER = re.compile(r"^Chapter\s+[IVXLCDM]+\s*$", re.IGNORECASE)

# ALL-CAPS chapter title lines (e.g. "AN UNEXPECTED PARTY", "ROAST MUTTON")
# Must be all uppercase letters, spaces, colons, hyphens — no lowercase
RE_ALL_CAPS_TITLE = re.compile(r"^[A-Z][A-Z\s\-:,'\.]+$")

# Known illustration caption fragments (matched anywhere in line)
ILLUSTRATION_CAPTIONS = [
    "Thror's Map",
    "The Trolls",
    "The Mountain-path",
    "The Misty Mountains",
    "The Misty Mountains Looking West from the Eyrie towards Goblin",
    "The Misty Mountains looking West",
    "Beorn's Hall",
    "The Elvenking's Gate",
    "Lake Town",
    "The Front Gate",
    "The Hall at Bag-End",
    "The Hall at Bag-End Residence of B.Baggins Esquire",
    "Map of Wilderland",
]

# Footnote / special character markers
RE_FOOTNOTE_MARKERS = re.compile(r"[†‡§¶]|\*(?=\s|$)")

# Lines that are just a number (page numbers in some eBooks)
RE_STANDALONE_NUMBER = re.compile(r"^\d+\s*$")

# Lines that are clearly metadata / boilerplate
STRIP_LINE_EXACT = {
    "CONTENTS",
    "COVER PAGE",
    "TITLE PAGE",
    "LIST OF ILLUSTRATIONS",
    "NOTE ON THE TEXT",
    "AUTHOR'S NOTE",
    "WORKS BY J.R.R. TOLKIEN",
    "COPYRIGHT",
    "ABOUT THE PUBLISHER",
    "ILLUSTRATIONS",
    "Author's Note",
    "THE HOBBIT",
    "OR",
    "THERE AND BACK AGAIN",
    "BY",
    "J.R.R. TOLKIEN",
    "The Times",
}

# Front matter ends just before the first real prose line.
# We detect this by looking for the sentinel that precedes Chapter I prose.
PROSE_START_SENTINEL = "In a hole in the ground there lived a hobbit."


# ---------------------------------------------------------------------------
# Line-level decisions
# ---------------------------------------------------------------------------

def is_strip_line(line: str) -> tuple[bool, str]:
    """Return (should_strip, reason)."""
    stripped = line.strip()

    if not stripped:
        return False, ""  # blank lines handled separately

    if stripped in STRIP_LINE_EXACT:
        return True, "metadata"

    if RE_CHAPTER_HEADER.match(stripped):
        return True, "chapter_header"

    if RE_ALL_CAPS_TITLE.match(stripped) and len(stripped) > 3:
        return True, "all_caps_title"

    # Only strip if the line IS the caption exactly — not prose that mentions it.
    # Exact match, or exact match ignoring trailing punctuation.
    for cap in ILLUSTRATION_CAPTIONS:
        if stripped.lower() == cap.lower():
            return True, "illustration_caption"
        if stripped.rstrip(".,:;").lower() == cap.lower():
            return True, "illustration_caption"

    if RE_STANDALONE_NUMBER.match(stripped):
        return True, "standalone_number"

    return False, ""


def clean_line(line: str) -> str:
    """Clean inline noise from a kept line."""
    # Remove footnote markers
    line = RE_FOOTNOTE_MARKERS.sub("", line)
    # Normalize unicode dashes to ASCII hyphen where appropriate
    line = line.replace("–", "-").replace("—", "—")
    # Collapse multiple spaces
    line = re.sub(r"  +", " ", line)
    return line.rstrip()


# ---------------------------------------------------------------------------
# Multi-line join: soft hyphenated line breaks
# ---------------------------------------------------------------------------

def join_hyphenated(lines: list[str]) -> list[str]:
    """
    Join lines where a word is broken across lines with a trailing hyphen.
    e.g. "tube-" + "shaped hall" → "tube-shaped hall"
    Only joins when the hyphen is mid-word (not an em-dash or list bullet).
    """
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Soft hyphen: line ends with a letter then hyphen, next line starts with lowercase
        if (line.endswith("-")
                and len(line) >= 2
                and line[-2].isalpha()
                and i + 1 < len(lines)
                and lines[i + 1]
                and lines[i + 1][0].islower()):
            result.append(line + lines[i + 1].lstrip())
            i += 2
        else:
            result.append(line)
            i += 1
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def clean_corpus(text: str, verbose: bool = False) -> tuple[str, dict]:
    raw_lines = text.splitlines()
    stats = {
        "raw_lines": len(raw_lines),
        "stripped_front_matter": 0,
        "stripped_metadata": 0,
        "stripped_chapter_header": 0,
        "stripped_all_caps_title": 0,
        "stripped_illustration_caption": 0,
        "stripped_standalone_number": 0,
        "joined_hyphenated": 0,
        "collapsed_blank_runs": 0,
    }

    # --- Phase 1: Strip front matter up to prose sentinel ---
    in_front_matter = True
    phase1 = []
    for line in raw_lines:
        if in_front_matter:
            if PROSE_START_SENTINEL in line:
                in_front_matter = False
                phase1.append(line)
            else:
                stats["stripped_front_matter"] += 1
        else:
            phase1.append(line)

    # --- Phase 2: Strip bad lines, clean inline noise ---
    phase2 = []
    for line in phase1:
        stripped = line.strip()
        if not stripped:
            phase2.append("")
            continue
        should_strip, reason = is_strip_line(stripped)
        if should_strip:
            stats[f"stripped_{reason}"] = stats.get(f"stripped_{reason}", 0) + 1
            if verbose:
                print(f"  STRIP [{reason}]: {stripped[:80]}")
            phase2.append("")  # blank placeholder to preserve paragraph breaks
        else:
            phase2.append(clean_line(line))

    # --- Phase 3: Join soft-hyphenated line breaks ---
    before_join = len([l for l in phase2 if l])
    phase3 = join_hyphenated(phase2)
    after_join = len([l for l in phase3 if l])
    stats["joined_hyphenated"] = before_join - after_join

    # --- Phase 4: Collapse runs of blank lines to a single blank ---
    phase4 = []
    prev_blank = False
    for line in phase3:
        if line == "":
            if not prev_blank:
                phase4.append("")
            else:
                stats["collapsed_blank_runs"] += 1
            prev_blank = True
        else:
            phase4.append(line)
            prev_blank = False

    # Strip leading/trailing blank lines
    while phase4 and phase4[0] == "":
        phase4.pop(0)
    while phase4 and phase4[-1] == "":
        phase4.pop()

    stats["output_lines"] = len(phase4)
    stats["output_words"] = sum(len(l.split()) for l in phase4)

    return "\n".join(phase4), stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean a text corpus for TILM2 training")
    parser.add_argument("--input",  required=True, help="Raw input text file")
    parser.add_argument("--output", required=True, help="Cleaned output text file")
    parser.add_argument("--report", action="store_true", help="Print cleanup report")
    parser.add_argument("--verbose", action="store_true", help="Print every stripped line")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    cleaned, stats = clean_corpus(raw, verbose=args.verbose)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(cleaned)

    if args.report:
        print("\n=== Corpus Cleanup Report ===")
        print(f"  Raw lines:                {stats['raw_lines']}")
        print(f"  Front matter stripped:    {stats['stripped_front_matter']}")
        print(f"  Metadata lines stripped:  {stats['stripped_metadata']}")
        print(f"  Chapter headers stripped: {stats['stripped_chapter_header']}")
        print(f"  ALL-CAPS titles stripped: {stats['stripped_all_caps_title']}")
        print(f"  Illustration captions:    {stats['stripped_illustration_caption']}")
        print(f"  Standalone numbers:       {stats['stripped_standalone_number']}")
        print(f"  Hyphenated joins:         {stats['joined_hyphenated']}")
        print(f"  Blank line runs collapsed:{stats['collapsed_blank_runs']}")
        print(f"  ─────────────────────────")
        print(f"  Output lines:             {stats['output_lines']}")
        print(f"  Output words:             {stats['output_words']:,}")
        print(f"\nSaved to: {args.output}")
    else:
        print(f"Cleaned corpus saved to {args.output} ({stats['output_lines']} lines, {stats['output_words']:,} words)")


if __name__ == "__main__":
    main()
