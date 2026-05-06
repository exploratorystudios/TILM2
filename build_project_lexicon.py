"""
Build a canonical project lexicon for TILM2 decoding and future TI export.

Usage:
    python3 build_project_lexicon.py --cmu-dict cmudict.dict \
        --source corpus/thematic.txt --source seeds.txt --output project_lexicon.json
"""

import argparse

from syllabifier import load_cmu_dict, save_project_lexicon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmu-dict", default="cmudict.dict")
    parser.add_argument("--source", action="append", required=True,
                        help="Corpus text file or directory to include in the project vocabulary")
    parser.add_argument("--output", default="project_lexicon.json")
    parser.add_argument("--max-syllables", type=int, default=4)
    args = parser.parse_args()

    load_cmu_dict(args.cmu_dict, required=True)
    data = save_project_lexicon(args.output, args.source, max_syllables=args.max_syllables)
    print(f"Wrote {args.output}")
    print(f"  sources: {len(data['vocab_sources'])}")
    print(f"  words: {len(data['words'])}")
    print(f"  exact decode entries: {len(data['decode_lexicon'])}")
    print(f"  stressless decode entries: {len(data['decode_lexicon_nostress'])}")


if __name__ == "__main__":
    main()
