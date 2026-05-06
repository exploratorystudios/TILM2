"""
Generate TI-BASIC English seed/input and graph-screen decode programs.

This is intentionally vocabulary-bounded: it supports words listed in
vocab_words.txt that can be tokenized into the current model component vocab.
Unknown words are rejected on the calculator.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from syllabifier import load_cmu_dict, tokenize_word

OUTPUT_SYLLABLES = 8


def write_program(outdir: Path, name: str, lines: list[str]) -> None:
    (outdir / f"{name}.txt").write_text("\n".join(output_to_disp(line) for line in lines) + "\n", encoding="utf-8")


def output_to_disp(line: str) -> str:
    match = re.fullmatch(r"Output\(\d+,\d+,(.+)\)", line)
    if not match:
        return line
    return f"Disp {match.group(1)}"


def token_ids(tok: dict, vocab: dict, word_boundary: bool) -> tuple[int, int, int, int, int] | None:
    try:
        onset = vocab["onsets"].index(tok["onset"]) + 1
        nucleus = vocab["nuclei"].index(tok["nucleus"]) + 1
        coda = vocab["codas"].index(tok["coda"]) + 1
    except ValueError:
        return None
    stress = min(int(tok.get("stress", 0)), 2) + 1
    wb = 2 if word_boundary else 1
    return onset, nucleus, coda, stress, wb


def load_word_tokens(words_path: Path, vocab: dict) -> dict[str, list[tuple[int, int, int, int, int]]]:
    out: dict[str, list[tuple[int, int, int, int, int]]] = {}
    for raw in words_path.read_text(encoding="utf-8").splitlines():
        word = raw.strip().lower()
        if not word:
            continue
        toks = tokenize_word(word)
        ids: list[tuple[int, int, int, int, int]] = []
        ok = True
        for i, tok in enumerate(toks):
            encoded = token_ids(tok, vocab, word_boundary=(i == 0))
            if encoded is None:
                ok = False
                break
            ids.append(encoded)
        if ok and ids:
            out[word] = ids
    return out


def load_project_decode_tokens(project_path: Path, vocab: dict) -> dict[str, list[tuple[int, int, int, int, int]]]:
    data = json.loads(project_path.read_text(encoding="utf-8"))
    out: dict[str, list[tuple[int, int, int, int, int]]] = {}
    for word, toks in data.get("words", {}).items():
        ids: list[tuple[int, int, int, int, int]] = []
        ok = True
        for i, tok in enumerate(toks):
            encoded = token_ids(tok, vocab, word_boundary=(i == 0))
            if encoded is None:
                ok = False
                break
            ids.append(encoded)
        if ok and ids:
            out[word.lower()] = ids
    return out


def make_seed_program() -> list[str]:
    return [
        "ClrHome",
        "Disp \"SEED SETUP\"",
        "Disp \"ENTER COUNT\"",
        "Disp \"ONE WORD\"",
        "Disp \"AT A TIME\"",
        "UnArchive |LX",
        "UnArchive |LI",
        "Fill(0,|LX)",
        "Fill(0,|LI)",
        "\" \"->Str9",
        "Input N",
        "N->E",
        "1->L",
        "While L<=E",
        "prgmT2WORD",
        "L+1->L",
        "End",
        "Output(6,1,\"SEED READY\")",
        "Pause ",
        "Return",
    ]


def make_word_program(word_tokens: dict[str, list[tuple[int, int, int, int, int]]]) -> list[str]:
    lines = [
        "ClrHome",
        "Disp \"SEED WORD\"",
        "Disp \"WORD #\"",
        "Disp L",
        "Disp \"UPPERCASE\"",
        "Input Str1",
        "0->Z",
    ]
    for word, toks in word_tokens.items():
        lines += [
            f"If Str1=\"{word.upper()}\"",
            "Then",
            "1->Z",
            f"Str9+\"{word} \"->Str9",
        ]
        for onset, nucleus, coda, stress, wb in toks:
            lines += [
                f"{onset}->O",
                f"{nucleus}->U",
                f"{coda}->C",
                f"{stress}->S",
                f"{wb}->W",
                "prgmT2ENC",
                "prgmT2CTX",
            ]
        lines += ["End"]
    lines += [
        "If Z=0",
        "Then",
        "Output(8,1,\"UNKNOWN WORD\")",
        "Pause ",
        "End",
        "Return",
    ]
    return lines


def make_decode_program(word_tokens: dict[str, list[tuple[int, int, int, int, int]]], vocab: dict) -> list[str]:
    singles: dict[tuple[int, int, int], str] = {}
    doubles: dict[tuple[int, int, int, int, int, int], str] = {}
    for word, toks in word_tokens.items():
        if len(toks) == 1:
            onset, nucleus, coda, _stress, _wb = toks[0]
            singles.setdefault((onset, nucleus, coda), word)
        elif len(toks) == 2:
            o1, u1, c1, _s1, _w1 = toks[0]
            o2, u2, c2, _s2, _w2 = toks[1]
            doubles.setdefault((o1, u1, c1, o2, u2, c2), word)

    lines = [
        "\"[?]\"->Str1",
        "1->G",
        "0->Z",
    ]
    for (o1, u1, c1, o2, u2, c2), word in sorted(doubles.items(), key=lambda item: item[1]):
        lines += [
            f"If O={o1} and U={u1} and C={c1} and P={o2} and Q={u2} and R={c2}",
            "Then",
            f"\"{word}\"->Str1",
            "2->G",
            "1->Z",
            "End",
        ]
    for (onset, nucleus, coda), word in sorted(singles.items(), key=lambda item: item[1]):
        lines += [
            f"If O={onset} and U={nucleus} and C={coda}",
            "Then",
            f"\"{word}\"->Str1",
            "1->G",
            "1->Z",
            "End",
        ]
    lines += [
        "If Z=0",
        "Then",
        "\"_\"->Str2",
        "\"_\"->Str3",
        "\"_\"->Str4",
    ]
    for index, phoneme in enumerate(vocab["onsets"], start=1):
        display = phoneme or "_"
        lines += [
            f"If O={index}",
            f"\"{display}\"->Str2",
        ]
    for index, phoneme in enumerate(vocab["nuclei"], start=1):
        display = phoneme or "_"
        lines += [
            f"If U={index}",
            f"\"{display}\"->Str3",
        ]
    for index, phoneme in enumerate(vocab["codas"], start=1):
        display = phoneme or "_"
        lines += [
            f"If C={index}",
            f"\"{display}\"->Str4",
        ]
    lines += [
        "Str2+Str3+Str4->Str1",
        "If length(Str1)=0",
        "\"[?]\"->Str1",
        "End",
    ]
    lines += ["Return"]
    return lines


def make_draw_program() -> list[str]:
    return [
        "ClrDraw",
        "AxesOff",
        "0->A",
        "0->B",
        "Text(A,B,\"SEED:\")",
        "11->A",
        "0->B",
        "If length(Str9)>1",
        "Then",
        "sub(Str9,2,length(Str9)-1)->Str8",
        "Else",
        "\"SEED SET\"->Str8",
        "End",
        "Text(A,B,Str8)",
        "32->A",
        "0->B",
        "Text(A,B,\"OUT:\")",
        "45->A",
        "0->B",
        "1->X",
        f"While X<={OUTPUT_SYLLABLES}",
        "5*X-4->Y",
        "iPart(|LID(Y)+.5)->O",
        "iPart(|LID(Y+1)+.5)->U",
        "iPart(|LID(Y+2)+.5)->C",
        "iPart(|LID(Y+3)+.5)->S",
        "iPart(|LID(Y+4)+.5)->W",
        "0->P",
        "0->Q",
        "0->R",
        f"If X<{OUTPUT_SYLLABLES} and iPart(|LID(Y+9)+.5)=1",
        "Then",
        "iPart(|LID(Y+5)+.5)->P",
        "iPart(|LID(Y+6)+.5)->Q",
        "iPart(|LID(Y+7)+.5)->R",
        "End",
        "prgmT2DEC",
        "If B+7*(length(Str1)+2)>250",
        "Then",
        "0->B",
        "A+14->A",
        "End",
        "If A>158",
        "Then",
        "Pause ",
        "ClrDraw",
        "AxesOff",
        "0->A",
        "0->B",
        "Text(A,B,\"OUT CONT:\")",
        "13->A",
        "0->B",
        "End",
        "Text(A,B,Str1)",
        "B+7*(length(Str1)+2)->B",
        "X+G->X",
        "End",
        "Pause ",
        "Return",
    ]


def make_dump_program() -> list[str]:
    return [
        "UnArchive |LID",
        f"For(X,1,{OUTPUT_SYLLABLES})",
        "ClrHome",
        "Disp \"SYL\"",
        "Disp X",
        "5*X-4->Y",
        "Disp \"O U C S W\"",
        "Disp |LID(Y)",
        "Disp |LID(Y+1)",
        "Disp |LID(Y+2)",
        "Disp |LID(Y+3)",
        "Disp |LID(Y+4)",
        "Pause ",
        "End",
        "Return",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--vocab-json", required=True)
    parser.add_argument("--words", default="vocab_words.txt")
    parser.add_argument("--cmu-dict", default="cmudict.dict")
    parser.add_argument("--project-lexicon",
                        help="Optional project_lexicon.json to use for generated-output decoding")
    args = parser.parse_args()

    runtime_dir = Path(args.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    vocab = json.loads(Path(args.vocab_json).read_text(encoding="utf-8"))
    words_path = Path(args.words)
    cmu_path = Path(args.cmu_dict)
    load_cmu_dict(str(cmu_path), required=True)
    word_tokens = load_word_tokens(words_path, vocab)
    decode_tokens = word_tokens
    if args.project_lexicon:
        decode_tokens = load_project_decode_tokens(Path(args.project_lexicon), vocab)

    write_program(runtime_dir, "T2SEED", make_seed_program())
    write_program(runtime_dir, "T2WORD", make_word_program(word_tokens))
    write_program(runtime_dir, "T2DEC", make_decode_program(decode_tokens, vocab))
    write_program(runtime_dir, "T2DRAW", make_draw_program())
    write_program(runtime_dir, "T2DUMP", make_dump_program())

    print(f"Wrote English TI programs to {runtime_dir}")
    print(f"Supported seed words: {len(word_tokens)}")
    print(f"Supported decode words: {len(decode_tokens)}")


if __name__ == "__main__":
    main()
