"""
Generate TI-BASIC runtime scaffolding from the packed TILM2 export.

The calculator cannot consume JSON, so this script compiles manifest/vocab
metadata into plain TI-BASIC .txt programs. This first layer validates object
names, dimensions, and list paging. The full forward-pass generator should build
on these helpers rather than reading JSON on-device.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


INPUT_DIM = 874
TOKEN_CONTEXT_DIM = 850
DISCOURSE_OFFSET = 850
WORD_OFFSET = 866
HIDDEN_DIM = 198
CHUNK_SIZE = 99
ACC_LIST = "|LT"
LOGIT_LIST = "|LF"
SCRATCH_LIST = "|LJ"
ENCODE_LIST = "|LG"
COND_LIST = "|LE"
CONTEXT_ID_LIST = "|LI"
RUNTIME_LISTS = {
    "|LH", "|LM", "|LT", "|LF", "|LJ", "|LG", "|LE",
    "|LR", "|LW", "|LS", "|LN", "|LO", "|LC",
    "|LID", "|LI", "|LHO", "|LHU", "|LHC", "|LX",
    "|LDR", "|LDW", "|LDS", "|LDN", "|LDO", "|LDC",
}

DEBUG_LIST_SPECS = [
    ("|LDR", 40),
    ("|LDW", 16),
    ("|LDS", 24),
    ("|LDN", 112),
    ("|LDO", 224),
    ("|LDC", 192),
]

OUTPUT_SYLLABLES = 8
HEAD_SPECS = {
    "role": ("|LR", 5, "b_role", ["W_role_0", "W_role_1"], "W_disc_role", "W_word_role"),
    "wb": ("|LW", 2, "b_wb", ["W_wb_0", "W_wb_1"], "W_disc_wb", "W_word_wb"),
    "stress": ("|LS", 3, "b_stress", ["W_stress_0", "W_stress_1"], "W_disc_stress", "W_word_stress"),
    "nucleus": ("|LN", 14, "b_nucleus", ["W_nucleus_0", "W_nucleus_1"], "W_disc_nucleus", "W_word_nucleus"),
    "onset": ("|LO", 28, "b_onset", ["W_onset_0", "W_onset_1"], "W_disc_onset", "W_word_onset"),
    "coda": ("|LC", 24, "b_coda", ["W_coda_0", "W_coda_1"], "W_disc_coda", "W_word_coda"),
}


def write_program(outdir: Path, name: str, lines: list[str]) -> None:
    if len(name) > 8:
        raise ValueError(f"TI program name too long: {name}")
    path = outdir / f"{name}.txt"
    path.write_text("\n".join(output_to_disp(line) for line in lines) + "\n", encoding="utf-8")


def output_to_disp(line: str) -> str:
    match = re.fullmatch(r"Output\(\d+,\d+,(.+)\)", line)
    if not match:
        return line
    return f"Disp {match.group(1)}"


def ti_dim_matrix(name: str, rows: int, cols: int) -> str:
    return f"{{{rows},{cols}}}->dim({name})"


def list_expr(name: str, idx: str) -> str:
    return f"{name}({idx})"


def list_expected_dim(manifest: dict[str, Any], i: int) -> int:
    if i < manifest["list_count"] - 1:
        return manifest["list_size"]
    return manifest["total_values"] - manifest["list_size"] * (manifest["list_count"] - 1)


def page_open_lines(list_name: str) -> list[str]:
    return [f"UnArchive {list_name}"]


def page_close_lines(list_name: str) -> list[str]:
    if list_name in RUNTIME_LISTS:
        return []
    return [f"Archive {list_name}"]


def runtime_open_lines(*list_names: str) -> list[str]:
    seen: set[str] = set()
    lines: list[str] = []
    for name in list_names:
        if name not in seen:
            lines += page_open_lines(name)
            seen.add(name)
    return lines


def generate_init(manifest: dict[str, Any]) -> list[str]:
    runtime_lists = ["|LH", "|LM", "|LT", "|LF", "|LJ", "|LG", "|LE", CONTEXT_ID_LIST]
    lines = [
        "ClrHome",
        "Output(1,1,\"TILM2 INIT\")",
        ".5->T",
        "3->K",
        ".5->G",
        "1->P",
        f"{manifest['list_count']}->L",
        f"{manifest['tensor_count']}->R",
    ]
    for name in runtime_lists:
        lines += page_open_lines(name)
    for name, length in DEBUG_LIST_SPECS:
        lines += [
            f"{length}->dim({name})",
            f"Fill(0,{name})",
        ]
    lines += [
        "Output(3,1,\"SCRATCH OK\")",
        "Output(4,1,\"LISTS EXPECTED\")",
        f"Output(5,1,\"{manifest['list_count']} LISTS\")",
        "Pause ",
    ]
    return lines


def generate_info(manifest: dict[str, Any], vocab: dict[str, list[str]]) -> list[str]:
    return [
        "ClrHome",
        "Output(1,1,\"TILM2 INFO\")",
        f"Output(2,1,\"LISTS {manifest['list_count']}\")",
        f"Output(3,1,\"VALS {manifest['total_values']}\")",
        f"Output(4,1,\"TENS {manifest['tensor_count']}\")",
        f"Output(5,1,\"ON {len(vocab['onsets'])} NU {len(vocab['nuclei'])}\")",
        f"Output(6,1,\"CO {len(vocab['codas'])} RO {len(vocab['roles'])}\")",
        "Output(8,1,\"NO JSON ON TI\")",
        "Pause ",
    ]


def generate_smoke(manifest: dict[str, Any]) -> list[str]:
    names = manifest["list_names"]
    checks = [
        (0, names[0], list_expected_dim(manifest, 0)),
        (5, names[5], list_expected_dim(manifest, 5)),
        (6, names[6], list_expected_dim(manifest, 6)),
        (manifest["list_count"] - 1, names[-1], list_expected_dim(manifest, manifest["list_count"] - 1)),
    ]
    lines = [
        "ClrHome",
        "Output(1,1,\"TILM2 SMOKE\")",
        "0->E",
    ]
    row = 2
    for _, name, expected in checks:
        lines += [
            *page_open_lines(name),
            f"If dim({name})!={expected}",
            "Then",
            "1->E",
            f"Output({row},1,\"BAD {name}\")",
            "Else",
            f"Output({row},1,\"OK {name}\")",
            "End",
            *page_close_lines(name),
        ]
        row += 1

    first_list = names[0]
    last_list = names[-1]
    last_len = list_expected_dim(manifest, manifest["list_count"] - 1)
    lines += [
        *page_open_lines(first_list),
        f"{first_list}(1)->A",
        *page_close_lines(first_list),
        *page_open_lines(last_list),
        f"{last_list}({last_len})->B",
        *page_close_lines(last_list),
        "Output(7,1,\"FIRST/LAST\")",
        "Disp A",
        "Disp B",
        "If E=0",
        "Then",
        "Disp \"SMOKE PASS\"",
        "Else",
        "Disp \"SMOKE FAIL\"",
        "End",
        "Pause ",
    ]
    return lines


def generate_getter(manifest: dict[str, Any]) -> list[str]:
    """Generic global flat-index getter.

    Input:  Z = 1-based flat parameter index.
    Output: V = parameter value.

    This is correctness-oriented, not fast enough for full inference. Generated
    direct tensor loaders should replace it in hot paths.
    """
    lines = [
        "0->V",
        "If Z<1",
        "Then",
        "Return",
        "End",
        f"If Z>{manifest['total_values']}",
        "Then",
        "Return",
        "End",
        f"int((Z-1)/{manifest['list_size']})+1->I",
        f"Z-{manifest['list_size']}*(I-1)->J",
    ]
    for i, name in enumerate(manifest["list_names"], start=1):
        lines += [
            f"If I={i}",
            "Then",
            *page_open_lines(name),
            f"{list_expr(name, 'J')}->V",
            *page_close_lines(name),
            "Return",
            "End",
        ]
    lines.append("Return")
    return lines


def generate_load_matrix_a(name: str, tensor: dict[str, Any]) -> list[str]:
    first_seg = tensor["segments"][0]
    list_name = first_seg["list"]
    first = int(first_seg["offset"])
    take = min(99, int(first_seg["length"]))
    last = first + take - 1
    return [
        "ClrHome",
        f"Output(1,1,\"READ {name[:10]}\")",
        "0->A",
        *runtime_open_lines(ACC_LIST),
        *page_open_lines(list_name),
        f"For(N,{first},{last})",
        "A+1->A",
        f"{list_name}(N)->{ACC_LIST}(A)",
        "End",
        *page_close_lines(list_name),
        "Output(3,1,\"LIST READY\")",
        "Return",
    ]


def matrix_segment_copy_lines(target: str, rows: int, cols: int, segments: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for seg in segments:
        list_name = seg["list"]
        first = int(seg["offset"])
        last = first + int(seg["length"]) - 1
        lines += [
            f"For(N,{first},{last})",
            "A+1->A",
            f"{list_name}(N)->{target}(int((A-1)/{cols})+1,A-{cols}*int((A-1)/{cols}))",
            "End",
        ]
    return lines


def flat_range_segments(manifest: dict[str, Any], start: int, length: int) -> list[dict[str, Any]]:
    """Return list segments for a 0-based flat range."""
    names = manifest["list_names"]
    list_size = manifest["list_size"]
    end = start + length
    out: list[dict[str, Any]] = []
    pos = start
    while pos < end:
        list_idx = pos // list_size
        offset0 = pos % list_size
        take = min(end - pos, list_size - offset0)
        out.append({
            "list": names[list_idx],
            "offset": offset0 + 1,
            "length": take,
        })
        pos += take
    return out


def load_vector_to_c_lines(manifest: dict[str, Any], tensor: dict[str, Any], start_in_tensor: int, length: int) -> list[str]:
    start = int(tensor["start"]) + start_in_tensor
    segments = flat_range_segments(manifest, start, length)
    lines = ["0->A"]
    for seg in segments:
        list_name = seg["list"]
        first = int(seg["offset"])
        last = first + int(seg["length"]) - 1
        lines += [
            *page_open_lines(list_name),
            f"For(N,{first},{last})",
            "A+1->A",
            f"{list_name}(N)->{ACC_LIST}(A)",
            "End",
            *page_close_lines(list_name),
        ]
    return lines


def load_vector_to_f_lines(manifest: dict[str, Any], tensor: dict[str, Any], length: int) -> list[str]:
    segments = flat_range_segments(manifest, int(tensor["start"]), length)
    lines = ["0->A"]
    for seg in segments:
        list_name = seg["list"]
        first = int(seg["offset"])
        last = first + int(seg["length"]) - 1
        lines += [
            *page_open_lines(list_name),
            f"For(N,{first},{last})",
            "A+1->A",
            f"{list_name}(N)->{LOGIT_LIST}(A)",
            "End",
            *page_close_lines(list_name),
        ]
    return lines


def load_matrix_a_lines(manifest: dict[str, Any], tensor_name: str) -> list[str]:
    tensor = manifest["tensors"][tensor_name]
    rows, cols = tensor["shape"]
    return [
        ti_dim_matrix("[A]", rows, cols),
        "0->A",
        *matrix_segment_copy_lines("[A]", rows, cols, tensor["segments"]),
    ]


def stream_tensor_add_to_list_lines(
    manifest: dict[str, Any],
    tensor_name: str,
    rows: int,
    cols: int,
    dest_list: str,
    src_expr: str,
) -> list[str]:
    tensor = manifest["tensors"][tensor_name]
    if tensor.get("layout") == "col_major":
        return stream_col_major_tensor_add_to_list_lines(manifest, tensor_name, rows, cols, dest_list, src_expr)
    return stream_row_major_tensor_add_to_list_lines(manifest, tensor_name, rows, cols, dest_list, src_expr)


def stream_row_major_tensor_add_to_list_lines(
    manifest: dict[str, Any],
    tensor_name: str,
    rows: int,
    cols: int,
    dest_list: str,
    src_expr: str,
) -> list[str]:
    segments = manifest["tensors"][tensor_name]["segments"]
    lines: list[str] = []
    abs_pos = 0
    for seg in segments:
        list_name = seg["list"]
        first = int(seg["offset"])
        length = int(seg["length"])
        lines += [
            *page_open_lines(list_name),
            f"{first}->N",
        ]
        remaining = length
        while remaining:
            row = abs_pos // cols + 1
            col = abs_pos % cols + 1
            if col == 1 and remaining >= cols:
                row_count = remaining // cols
                lines += [
                    f"For(R,{row},{row + row_count - 1})",
                    f"For(Q,1,{cols})",
                    f"{dest_list}(R)+{list_name}(N)*{src_expr}->{dest_list}(R)",
                    "N+1->N",
                    "End",
                    "End",
                ]
                advance = row_count * cols
            else:
                run = min(remaining, cols - col + 1)
                lines += [
                    f"For(Q,{col},{col + run - 1})",
                    f"{dest_list}({row})+{list_name}(N)*{src_expr}->{dest_list}({row})",
                    "N+1->N",
                    "End",
                ]
                advance = run
            abs_pos += advance
            remaining -= advance
        lines += page_close_lines(list_name)
    return lines


def stream_col_major_tensor_add_to_list_lines(
    manifest: dict[str, Any],
    tensor_name: str,
    rows: int,
    cols: int,
    dest_list: str,
    src_expr: str,
) -> list[str]:
    segments = manifest["tensors"][tensor_name]["segments"]
    lines: list[str] = []
    abs_pos = 0
    for seg in segments:
        list_name = seg["list"]
        first = int(seg["offset"])
        length = int(seg["length"])
        lines += [
            *page_open_lines(list_name),
            f"{first}->N",
        ]
        remaining = length
        while remaining:
            col = abs_pos // rows + 1
            row = abs_pos % rows + 1
            if row == 1 and remaining >= rows:
                col_count = remaining // rows
                lines += [
                    f"For(Q,{col},{col + col_count - 1})",
                    f"{src_expr}->V",
                    "If V!=0",
                    "Then",
                    f"For(R,1,{rows})",
                    f"{dest_list}(R)+{list_name}(N)*V->{dest_list}(R)",
                    "N+1->N",
                    "End",
                    "Else",
                    f"N+{rows}->N",
                    "End",
                    "End",
                ]
                advance = col_count * rows
            else:
                run = min(remaining, rows - row + 1)
                lines += [
                    f"{col}->Q",
                    f"{src_expr}->V",
                    "If V!=0",
                    "Then",
                    f"For(R,{row},{row + run - 1})",
                    f"{dest_list}(R)+{list_name}(N)*V->{dest_list}(R)",
                    "N+1->N",
                    "End",
                    "Else",
                    f"N+{run}->N",
                    "End",
                ]
                advance = run
            abs_pos += advance
            remaining -= advance
        lines += page_close_lines(list_name)
    return lines


def matvec_add_from_lx_lines(manifest: dict[str, Any], tensor_name: str, rows: int, cols: int, lx_start0: int, dest_list: str = ACC_LIST) -> list[str]:
    src = "Q" if lx_start0 == 0 else f"{lx_start0}+Q"
    return stream_tensor_add_to_list_lines(manifest, tensor_name, rows, cols, dest_list, f"|LX({src})")


def matvec_add_from_list_lines(manifest: dict[str, Any], tensor_name: str, rows: int, cols: int, src_list: str, src_start0: int, dest_list: str = ACC_LIST) -> list[str]:
    src = "Q" if src_start0 == 0 else f"{src_start0}+Q"
    return stream_tensor_add_to_list_lines(manifest, tensor_name, rows, cols, dest_list, f"{src_list}({src})")


def matvec_add_to_f_from_list_lines(manifest: dict[str, Any], tensor_name: str, rows: int, cols: int, src_list: str) -> list[str]:
    return matvec_add_from_list_lines(manifest, tensor_name, rows, cols, src_list, 0, LOGIT_LIST)


def matrix_column_add_to_f_lines(manifest: dict[str, Any], tensor_name: str, rows: int, id_var: str) -> list[str]:
    tensor = manifest["tensors"][tensor_name]
    cols = int(tensor["shape"][1])
    tensor_start = int(tensor["start"])
    lines: list[str] = []
    for col_id in range(1, cols + 1):
        lines += [
            f"If {id_var}={col_id}",
            "Then",
        ]
        by_list: dict[str, list[tuple[int, int]]] = {}
        for row in range(1, rows + 1):
            flat_start = tensor_start + (row - 1) * cols + (col_id - 1)
            seg = flat_range_segments(manifest, flat_start, 1)[0]
            list_name = seg["list"]
            offset = int(seg["offset"])
            by_list.setdefault(list_name, []).append((row, offset))
        for list_name, reads in by_list.items():
            lines += [
                *page_open_lines(list_name),
            ]
            for row, offset in reads:
                lines += [
                f"{LOGIT_LIST}({row})+{list_name}({offset})->{LOGIT_LIST}({row})",
                ]
            lines += [
                *page_close_lines(list_name),
            ]
        lines += ["End"]
    return lines


def copy_embedding_row_to_list_lines(
    manifest: dict[str, Any],
    tensor_name: str,
    id_var: str,
    cols: int,
    dest_list: str,
    dest_start0: int = 0,
) -> list[str]:
    tensor = manifest["tensors"][tensor_name]
    rows = int(tensor["shape"][0])
    tensor_start = int(tensor["start"])
    lines: list[str] = []
    for row_id in range(1, rows + 1):
        lines += [
            f"If {id_var}={row_id}",
            "Then",
            f"{dest_start0 + 1}->J",
        ]
        for seg in flat_range_segments(manifest, tensor_start + (row_id - 1) * cols, cols):
            list_name = seg["list"]
            first = int(seg["offset"])
            last = first + int(seg["length"]) - 1
            lines += [
                *page_open_lines(list_name),
                f"For(N,{first},{last})",
                f"{list_name}(N)->{dest_list}(J)",
                "J+1->J",
                "End",
                *page_close_lines(list_name),
            ]
        lines += ["End"]
    return lines


def load_embedding_row_to_b_lines(manifest: dict[str, Any], tensor_name: str, id_var: str, cols: int, dest_start0: int = 0) -> list[str]:
    return copy_embedding_row_to_list_lines(manifest, tensor_name, id_var, cols, COND_LIST, dest_start0)


def cond_matrix_add_to_f_from_b_lines(manifest: dict[str, Any], tensor_name: str, rows: int, cols: int) -> list[str]:
    # Conditional tensors are stored transposed relative to the logits: cols x rows.
    segments = manifest["tensors"][tensor_name]["segments"]
    lines = ["0->Z"]
    for seg in segments:
        list_name = seg["list"]
        first = int(seg["offset"])
        last = first + int(seg["length"]) - 1
        lines += [
            *page_open_lines(list_name),
            f"For(N,{first},{last})",
            "Z+1->Z",
            f"int((Z-1)/{rows})+1->Q",
            f"Z-{rows}*int((Z-1)/{rows})->R",
            f"{LOGIT_LIST}(R)+{COND_LIST}(Q)*{list_name}(N)->{LOGIT_LIST}(R)",
            "End",
            *page_close_lines(list_name),
        ]
    return lines


def vector_add_to_f_lines(manifest: dict[str, Any], tensor_name: str, rows: int) -> list[str]:
    return [
        *load_vector_to_c_lines(manifest, manifest["tensors"][tensor_name], 0, rows),
        f"For(R,1,{rows})",
        f"{LOGIT_LIST}(R)+{ACC_LIST}(R)->{LOGIT_LIST}(R)",
        "End",
    ]


def softmax_f_to_list_lines(rows: int, dest_list: str) -> list[str]:
    return [
        f"{LOGIT_LIST}(1)/T->M",
        f"For(I,2,{rows})",
        f"If {LOGIT_LIST}(I)/T>M",
        "Then",
        f"{LOGIT_LIST}(I)/T->M",
        "End",
        "End",
        "0->V",
        f"For(I,1,{rows})",
        f"10^(({LOGIT_LIST}(I)/T-M)/ln(10))->{LOGIT_LIST}(I)",
        f"V+{LOGIT_LIST}(I)->V",
        "End",
        f"For(I,1,{rows})",
        f"{LOGIT_LIST}(I)/V->{dest_list}(I)",
        "End",
    ]


def apply_nucleus_mask_lines() -> list[str]:
    return [
        "UnArchive |LNM",
        "0->V",
        "For(I,1,14)",
        "V+|LNM(14*(O-1)+I)->V",
        "End",
        "If V>0",
        "Then",
        "For(I,1,14)",
        "If |LNM(14*(O-1)+I)=0",
        "Then",
        f"~999999999->{LOGIT_LIST}(I)",
        "End",
        "End",
        "End",
        "Archive |LNM",
    ]


def coda_mask_read_lines() -> list[str]:
    lines = [
        "0->V",
        "int((Z-1)/999)+1->H",
        "Z-999*(H-1)->J",
    ]
    for i in range(1, 11):
        lines += [
            f"If H={i}",
            "Then",
            f"UnArchive |LCM{i}",
            f"|LCM{i}(J)->V",
            f"Archive |LCM{i}",
            "End",
        ]
    return lines


def apply_coda_mask_lines() -> list[str]:
    read = coda_mask_read_lines()
    return [
        "0->M",
        "For(I,1,24)",
        "((O-1)*336+(U-1)*24+I)->Z",
        *read,
        "M+V->M",
        "End",
        "If M>0",
        "Then",
        "For(I,1,24)",
        "((O-1)*336+(U-1)*24+I)->Z",
        *read,
        "If V=0",
        "Then",
        f"~999999999->{LOGIT_LIST}(I)",
        "End",
        "End",
        "End",
    ]


def apply_rep_penalty_lines(hist_list: str) -> list[str]:
    return [
        "If G>0",
        "Then",
        "For(I,1,15)",
        f"If {hist_list}(I)>0",
        "Then",
        f"{LOGIT_LIST}({hist_list}(I))-G->{LOGIT_LIST}({hist_list}(I))",
        "End",
        "End",
        "End",
    ]


def relu_store_lines(rows: int, dest_list: str, list_start0: int) -> list[str]:
    dest_idx = "R" if list_start0 == 0 else f"{list_start0}+R"
    return [
        f"For(R,1,{rows})",
        f"If {ACC_LIST}(R)<0",
        "Then",
        f"0->{ACC_LIST}(R)",
        "End",
        f"{ACC_LIST}(R)->{dest_list}({dest_idx})",
        "End",
    ]


def add_h1_precompute_table_lines(table: dict[str, Any]) -> list[str]:
    rows = int(table["rows"])
    lines = [
        f"{CONTEXT_ID_LIST}({int(table['context_id_index'])})->D",
        "If D>0",
        "Then",
        "iPart(D+.5)->D",
    ]
    for group in table["groups"]:
        first_id = int(group["first_id"])
        count = int(group["count"])
        last_id = first_id + count - 1
        list_name = group["list"]
        expected_dim = rows * count
        short_name = list_name.replace("|L", "")
        lines += [
            f"If D>={first_id}",
            "Then",
            f"If D<={last_id}",
            "Then",
            *page_open_lines(list_name),
            f"If dim({list_name})<{expected_dim}",
            "Then",
            f"Disp \"BAD {short_name}\"",
            "Disp \"DIM\"",
            f"Disp dim({list_name})",
            "Disp \"NEED\"",
            f"Disp {expected_dim}",
            "Pause ",
            "Stop",
            "End",
            f"{rows}*(D-{first_id})->J",
            "iPart(J)->J",
            f"J+{rows}->Q",
            "If J<0",
            "Then",
            f"Disp \"IDX {short_name}\"",
            "Disp \"J\"",
            "Disp J",
            "Pause ",
            "Stop",
            "End",
            f"If Q>dim({list_name})",
            "Then",
            f"Disp \"IDX {short_name}\"",
            "Disp \"Q\"",
            "Disp Q",
            "Disp \"DIM\"",
            f"Disp dim({list_name})",
            "Pause ",
            "Stop",
            "End",
            f"For(R,1,{rows})",
            "J+R->Q",
            f"{ACC_LIST}(R)+{list_name}(Q)->{ACC_LIST}(R)",
            "End",
            *page_close_lines(list_name),
            "End",
            "End",
        ]
    lines += ["End"]
    return lines


def generate_h1_precomputed(manifest: dict[str, Any]) -> list[str]:
    tables_by_chunk: dict[int, list[dict[str, Any]]] = {}
    for table in manifest["h1_precompute"]["tables"]:
        tables_by_chunk.setdefault(int(table["row_chunk"]), []).append(table)

    lines = [
        "ClrHome",
        "Output(1,1,\"TILM2 H1P\")",
        *runtime_open_lines("|LX", "|LH", ACC_LIST, CONTEXT_ID_LIST),
    ]
    for i in range(2):
        lines += [
            f"Output(2,1,\"H1P CHUNK {i + 1}\")",
            *load_vector_to_c_lines(manifest, manifest["tensors"]["b_h"], i * CHUNK_SIZE, CHUNK_SIZE),
        ]
        for table in sorted(tables_by_chunk.get(i, []), key=lambda t: (int(t["slot"]), str(t["part"]))):
            lines += add_h1_precompute_table_lines(table)
        lines += [
            f"Output(3,1,\"D H1 {i}\")",
            *matvec_add_from_lx_lines(manifest, f"W_disc_h1_{i}", CHUNK_SIZE, 16, DISCOURSE_OFFSET),
            f"Output(3,1,\"W H1 {i}\")",
            *matvec_add_from_lx_lines(manifest, f"W_word_h1_{i}", CHUNK_SIZE, 8, WORD_OFFSET),
            *relu_store_lines(CHUNK_SIZE, "|LH", i * CHUNK_SIZE),
        ]
    lines += [
        "Output(8,1,\"H1P READY\")",
        "Return",
    ]
    return lines


def generate_h1(manifest: dict[str, Any]) -> list[str]:
    if manifest.get("h1_precompute", {}).get("enabled"):
        return generate_h1_precomputed(manifest)

    lines = [
        "ClrHome",
        "Output(1,1,\"TILM2 H1\")",
        *runtime_open_lines("|LX", "|LH", ACC_LIST),
    ]
    input_col_lengths = [99] * 8 + [82]
    for i in range(2):
        lines += [
            f"Output(2,1,\"H1 CHUNK {i + 1}\")",
            *load_vector_to_c_lines(manifest, manifest["tensors"]["b_h"], i * CHUNK_SIZE, CHUNK_SIZE),
        ]
        for j, cols in enumerate(input_col_lengths):
            lines += [
                f"Output(3,1,\"WH {i}{j}\")",
                *matvec_add_from_lx_lines(manifest, f"W_h_r{i}_c{j}", CHUNK_SIZE, cols, j * CHUNK_SIZE),
            ]
        lines += [
            f"Output(3,1,\"D H1 {i}\")",
            *matvec_add_from_lx_lines(manifest, f"W_disc_h1_{i}", CHUNK_SIZE, 16, DISCOURSE_OFFSET),
            f"Output(3,1,\"W H1 {i}\")",
            *matvec_add_from_lx_lines(manifest, f"W_word_h1_{i}", CHUNK_SIZE, 8, WORD_OFFSET),
            *relu_store_lines(CHUNK_SIZE, "|LH", i * CHUNK_SIZE),
        ]
    lines += [
        "Output(8,1,\"H1 READY\")",
        "Return",
    ]
    return lines


def generate_h2(manifest: dict[str, Any]) -> list[str]:
    lines = [
        "ClrHome",
        "Output(1,1,\"TILM2 H2\")",
        *runtime_open_lines("|LX", "|LH", "|LM", ACC_LIST),
    ]
    for i in range(2):
        lines += [
            f"Output(2,1,\"H2 CHUNK {i + 1}\")",
            *load_vector_to_c_lines(manifest, manifest["tensors"]["b_h2"], i * CHUNK_SIZE, CHUNK_SIZE),
        ]
        for j in range(2):
            lines += [
                f"Output(3,1,\"W2 {i}{j}\")",
                *matvec_add_from_list_lines(manifest, f"W_h2_r{i}_c{j}", CHUNK_SIZE, CHUNK_SIZE, "|LH", j * CHUNK_SIZE),
            ]
        lines += [
            f"Output(3,1,\"D H2 {i}\")",
            *matvec_add_from_lx_lines(manifest, f"W_disc_h2_{i}", CHUNK_SIZE, 16, DISCOURSE_OFFSET),
            f"Output(3,1,\"W H2 {i}\")",
            *matvec_add_from_lx_lines(manifest, f"W_word_h2_{i}", CHUNK_SIZE, 8, WORD_OFFSET),
            *relu_store_lines(CHUNK_SIZE, "|LM", i * CHUNK_SIZE),
        ]
    lines += [
        "Output(8,1,\"H2 READY\")",
        "Return",
    ]
    return lines


def generate_check_hidden() -> list[str]:
    return [
        "ClrHome",
        "Output(1,1,\"CHK HIDDEN\")",
        *runtime_open_lines("|LH", "|LM"),
        "UnArchive |LHR",
        "UnArchive |LMR",
        "0->A",
        "For(I,1,198)",
        "abs(|LH(I)-|LHR(I))->B",
        "If B>A",
        "Then",
        "B->A",
        "End",
        "End",
        "0->C",
        "For(I,1,198)",
        "abs(|LM(I)-|LMR(I))->B",
        "If B>C",
        "Then",
        "B->C",
        "End",
        "End",
        "Output(3,1,\"MAX H1\")",
        "Disp A",
        "Output(5,1,\"MAX H2\")",
        "Disp C",
        "Archive |LHR",
        "Archive |LMR",
        "Pause ",
        "Return",
    ]


def generate_head_lines(
    manifest: dict[str, Any],
    label: str,
    dest_list: str,
    rows: int,
    bias_name: str,
    h_chunks: list[str],
    disc_name: str,
    word_name: str,
    extra_lines: list[str] | None = None,
) -> list[str]:
    lines = [
        f"Output(2,1,\"{label[:13]}\")",
        *load_vector_to_f_lines(manifest, manifest["tensors"][bias_name], rows),
    ]
    for i, chunk_name in enumerate(h_chunks):
        lines += [
            *matvec_add_from_list_lines(manifest, chunk_name, rows, CHUNK_SIZE, "|LM", i * CHUNK_SIZE, LOGIT_LIST),
        ]
    lines += [
        *matvec_add_from_lx_lines(manifest, disc_name, rows, 16, DISCOURSE_OFFSET, LOGIT_LIST),
        *matvec_add_from_lx_lines(manifest, word_name, rows, 8, WORD_OFFSET, LOGIT_LIST),
    ]
    if extra_lines:
        lines += extra_lines
    lines += softmax_f_to_list_lines(rows, dest_list)
    return lines


def generate_output_heads(manifest: dict[str, Any]) -> list[str]:
    lines = [
        "ClrHome",
        "Output(1,1,\"TILM2 OUT\")",
        *runtime_open_lines(
            "|LX", "|LM", ACC_LIST, LOGIT_LIST,
            "|LR", "|LW", "|LS", "|LN", "|LO", "|LC",
            "|LHO",
        ),
        "If T<=0",
        "Then",
        ".5->T",
        "End",
    ]
    lines += generate_head_lines(manifest, "ROLE", *HEAD_SPECS["role"])
    lines += generate_head_lines(
        manifest,
        "WB",
        *HEAD_SPECS["wb"],
        extra_lines=[
            *matvec_add_to_f_from_list_lines(manifest, "W_role_wb", 2, 5, "|LR"),
        ],
    )
    lines += generate_head_lines(
        manifest,
        "STRESS",
        *HEAD_SPECS["stress"],
        extra_lines=[
            *matvec_add_to_f_from_list_lines(manifest, "W_role_stress", 3, 5, "|LR"),
        ],
    )
    lines += generate_head_lines(
        manifest,
        "NUCLEUS",
        *HEAD_SPECS["nucleus"],
        extra_lines=[
            *matvec_add_to_f_from_list_lines(manifest, "W_role_nucleus", 14, 5, "|LR"),
            *matvec_add_to_f_from_list_lines(manifest, "W_wb_nucleus", 14, 2, "|LW"),
        ],
    )
    lines += generate_head_lines(
        manifest,
        "ONSET",
        *HEAD_SPECS["onset"],
        extra_lines=[
            *matvec_add_to_f_from_list_lines(manifest, "W_role_onset", 28, 5, "|LR"),
            *matvec_add_to_f_from_list_lines(manifest, "W_wb_onset", 28, 2, "|LW"),
            *matvec_add_to_f_from_list_lines(manifest, "W_nuc_gate", 28, 14, "|LN"),
            *vector_add_to_f_lines(manifest, "b_nuc_gate", 28),
            *matvec_add_to_f_from_list_lines(manifest, "W_stress_gate", 28, 3, "|LS"),
            *vector_add_to_f_lines(manifest, "b_stress_gate", 28),
            *apply_rep_penalty_lines("|LHO"),
        ],
    )
    lines += generate_head_lines(
        manifest,
        "CODA",
        *HEAD_SPECS["coda"],
        extra_lines=[
            *matvec_add_to_f_from_list_lines(manifest, "W_role_coda", 24, 5, "|LR"),
            *matvec_add_to_f_from_list_lines(manifest, "W_wb_coda", 24, 2, "|LW"),
        ],
    )
    lines += [
        "Output(8,1,\"OUT READY\")",
        "Return",
    ]
    return lines


def generate_generation_outputs(manifest: dict[str, Any]) -> list[str]:
    """Python generate() style sampled cascade."""
    lines = [
        "ClrHome",
        "Output(1,1,\"TILM2 OGEN\")",
        *runtime_open_lines(
            "|LX", "|LM", ACC_LIST, LOGIT_LIST, SCRATCH_LIST, COND_LIST,
            "|LR", "|LW", "|LS", "|LN", "|LO", "|LC",
            "|LHO", "|LHU", "|LHC",
        ),
        "If T<=0",
        "Then",
        ".5->T",
        "End",
    ]
    lines += generate_head_lines(manifest, "ROLE", *HEAD_SPECS["role"])
    lines += generate_head_lines(
        manifest,
        "WB",
        *HEAD_SPECS["wb"],
        extra_lines=[
            *matvec_add_to_f_from_list_lines(manifest, "W_role_wb", 2, 5, "|LR"),
        ],
    )
    lines += sample_from_list_lines("|LW", 2, "W", "WB")
    lines += generate_head_lines(
        manifest,
        "STRESS",
        *HEAD_SPECS["stress"],
        extra_lines=[
            *matvec_add_to_f_from_list_lines(manifest, "W_role_stress", 3, 5, "|LR"),
        ],
    )
    lines += sample_from_list_lines("|LS", 3, "S", "STR")

    # Preliminary nucleus distribution for onset gating.
    lines += generate_head_lines(
        manifest,
        "NUC RAW",
        *HEAD_SPECS["nucleus"],
        extra_lines=[
            *matvec_add_to_f_from_list_lines(manifest, "W_role_nucleus", 14, 5, "|LR"),
            *matrix_column_add_to_f_lines(manifest, "W_wb_nucleus", 14, "W"),
        ],
    )

    lines += generate_head_lines(
        manifest,
        "ONSET",
        *HEAD_SPECS["onset"],
        extra_lines=[
            *matvec_add_to_f_from_list_lines(manifest, "W_role_onset", 28, 5, "|LR"),
            *matrix_column_add_to_f_lines(manifest, "W_wb_onset", 28, "W"),
            *matvec_add_to_f_from_list_lines(manifest, "W_nuc_gate", 28, 14, "|LN"),
            *vector_add_to_f_lines(manifest, "b_nuc_gate", 28),
            *matvec_add_to_f_from_list_lines(manifest, "W_stress_gate", 28, 3, "|LS"),
            *vector_add_to_f_lines(manifest, "b_stress_gate", 28),
            *apply_rep_penalty_lines("|LHO"),
        ],
    )
    lines += sample_from_list_lines("|LO", 28, "O", "ONS")
    lines += load_embedding_row_to_b_lines(manifest, "E_onset", "O", 21, 0)

    lines += generate_head_lines(
        manifest,
        "NUCLEUS",
        *HEAD_SPECS["nucleus"],
        extra_lines=[
            *matvec_add_to_f_from_list_lines(manifest, "W_role_nucleus", 14, 5, "|LR"),
            *matrix_column_add_to_f_lines(manifest, "W_wb_nucleus", 14, "W"),
            *cond_matrix_add_to_f_from_b_lines(manifest, "W_nuc_cond", 14, 21),
            *apply_nucleus_mask_lines(),
            *apply_rep_penalty_lines("|LHU"),
        ],
    )
    lines += sample_from_list_lines("|LN", 14, "U", "NUC")
    lines += load_embedding_row_to_b_lines(manifest, "E_nucleus", "U", 21, 21)

    lines += generate_head_lines(
        manifest,
        "CODA",
        *HEAD_SPECS["coda"],
        extra_lines=[
            *matvec_add_to_f_from_list_lines(manifest, "W_role_coda", 24, 5, "|LR"),
            *matrix_column_add_to_f_lines(manifest, "W_wb_coda", 24, "W"),
            *cond_matrix_add_to_f_from_b_lines(manifest, "W_coda_cond", 24, 42),
            *apply_coda_mask_lines(),
            *apply_rep_penalty_lines("|LHC"),
        ],
    )
    lines += sample_from_list_lines("|LC", 24, "C", "COD")
    lines += [
        "Output(8,1,\"OGEN READY\")",
        "Return",
    ]
    return lines


def generate_check_outputs() -> list[str]:
    checks = [
        ("|LR", "|LRR", 5, "ROLE"),
        ("|LW", "|LWR", 2, "WB"),
        ("|LS", "|LSR", 3, "STR"),
        ("|LN", "|LNR", 14, "NUC"),
        ("|LO", "|LOR", 28, "ONS"),
        ("|LC", "|LCR", 24, "COD"),
    ]
    lines = [
        "ClrHome",
        "Output(1,1,\"CHK OUT\")",
        *runtime_open_lines("|LR", "|LW", "|LS", "|LN", "|LO", "|LC"),
        "UnArchive |LRR",
        "UnArchive |LWR",
        "UnArchive |LSR",
        "UnArchive |LNR",
        "UnArchive |LOR",
        "UnArchive |LCR",
        "0->A",
    ]
    row = 2
    for got, ref, rows, label in checks:
        lines += [
            "0->C",
            f"For(I,1,{rows})",
            f"abs({got}(I)-{ref}(I))->B",
            "If B>C",
            "Then",
            "B->C",
            "End",
            "End",
            "If C>A",
            "Then",
            "C->A",
            "End",
            f"Output({row},1,\"{label}\")",
            "Disp C",
        ]
        row += 1
    lines += [
        "Output(8,1,\"MAX\")",
        "Disp A",
        "Archive |LRR",
        "Archive |LWR",
        "Archive |LSR",
        "Archive |LNR",
        "Archive |LOR",
        "Archive |LCR",
        "Pause ",
        "Return",
    ]
    return lines


def sample_from_list_lines(src_list: str, rows: int, dest_var: str, label: str) -> list[str]:
    return [
        f"Output(2,1,\"S {label}\")",
        f"For(I,1,{rows})",
        f"{src_list}(I)->{LOGIT_LIST}(I)",
        f"0->{SCRATCH_LIST}(I)",
        "End",
        "K->H",
        "If H<1",
        "Then",
        "4->H",
        "End",
        f"If H>{rows}",
        "Then",
        f"{rows}->H",
        "End",
        "For(J,1,H)",
        "~1->M",
        "0->Y",
        f"For(I,1,{rows})",
        f"If {LOGIT_LIST}(I)>M",
        "Then",
        f"{LOGIT_LIST}(I)->M",
        "I->Y",
        "End",
        "End",
        "If Y>0",
        "Then",
        f"{LOGIT_LIST}(Y)->{SCRATCH_LIST}(Y)",
        f"~1->{LOGIT_LIST}(Y)",
        "End",
        "End",
        "0->M",
        f"For(I,1,{rows})",
        f"M+{SCRATCH_LIST}(I)->M",
        "End",
        "rand*M->Z",
        "0->M",
        f"0->{dest_var}",
        f"For(I,1,{rows})",
        f"M+{SCRATCH_LIST}(I)->M",
        f"If {dest_var}=0",
        "Then",
        "If Z<=M",
        "Then",
        f"I->{dest_var}",
        "End",
        "End",
        "End",
        f"If {dest_var}=0",
        "Then",
        f"{rows}->{dest_var}",
        "End",
    ]


def generate_sample() -> list[str]:
    return [
        "ClrHome",
        "Output(1,1,\"TILM2 SAMPLE\")",
        *runtime_open_lines(
            LOGIT_LIST, SCRATCH_LIST,
            "|LR", "|LW", "|LS", "|LN", "|LO", "|LC",
        ),
        *sample_from_list_lines("|LR", 5, "D", "ROLE"),
        *sample_from_list_lines("|LW", 2, "W", "WB"),
        *sample_from_list_lines("|LS", 3, "S", "STR"),
        *sample_from_list_lines("|LN", 14, "U", "NUC"),
        *sample_from_list_lines("|LO", 28, "O", "ONS"),
        *sample_from_list_lines("|LC", 24, "C", "COD"),
        "Output(8,1,\"SAMP READY\")",
        "Return",
    ]


def encode_embedding_row_lines(manifest: dict[str, Any], tensor_name: str, id_var: str, cols: int, dest_start0: int) -> list[str]:
    return copy_embedding_row_to_list_lines(manifest, tensor_name, id_var, cols, ENCODE_LIST, dest_start0)


def generate_encode(manifest: dict[str, Any]) -> list[str]:
    return [
        "ClrHome",
        "Output(1,1,\"TILM2 ENCODE\")",
        *runtime_open_lines(ENCODE_LIST),
        *encode_embedding_row_lines(manifest, "E_onset", "O", 21, 0),
        *encode_embedding_row_lines(manifest, "E_nucleus", "U", 21, 21),
        *encode_embedding_row_lines(manifest, "E_coda", "C", 21, 42),
        *encode_embedding_row_lines(manifest, "E_stress", "S", 21, 63),
        *encode_embedding_row_lines(manifest, "E_wb", "W", 1, 84),
        "Output(8,1,\"ENC READY\")",
        "Return",
    ]


def tanh_store_context_lines(rows: int, offset0: int) -> list[str]:
    lines = [
        f"For(R,1,{rows})",
        f"If {ACC_LIST}(R)>10",
        "Then",
        f"1->{ACC_LIST}(R)",
        "Else",
        f"If {ACC_LIST}(R)<~10",
        "Then",
        f"~1->{ACC_LIST}(R)",
        "Else",
        f"tanh({ACC_LIST}(R))->{ACC_LIST}(R)",
        "End",
        "End",
    ]
    lines += [
        f"{ACC_LIST}(R)->|LX({offset0}+R)",
        "End",
    ]
    return lines


def generate_context_update(manifest: dict[str, Any]) -> list[str]:
    return [
        "ClrHome",
        "Output(1,1,\"TILM2 CTX\")",
        *runtime_open_lines("|LX", CONTEXT_ID_LIST, ENCODE_LIST, ACC_LIST),
        f"For(I,1,{TOKEN_CONTEXT_DIM - 85})",
        "|LX(I+85)->|LX(I)",
        "End",
        "For(I,1,85)",
        f"{ENCODE_LIST}(I)->|LX(765+I)",
        "End",
        "For(I,1,45)",
        f"{CONTEXT_ID_LIST}(I+5)->{CONTEXT_ID_LIST}(I)",
        "End",
        f"O->{CONTEXT_ID_LIST}(46)",
        f"U->{CONTEXT_ID_LIST}(47)",
        f"C->{CONTEXT_ID_LIST}(48)",
        f"S->{CONTEXT_ID_LIST}(49)",
        f"W->{CONTEXT_ID_LIST}(50)",
        "Output(2,1,\"DISC\")",
        *load_vector_to_c_lines(manifest, manifest["tensors"]["b_disc"], 0, 16),
        *matvec_add_from_list_lines(manifest, "W_disc_in", 16, 85, ENCODE_LIST, 0),
        *matvec_add_from_lx_lines(manifest, "W_disc_h", 16, 16, DISCOURSE_OFFSET),
        *tanh_store_context_lines(16, DISCOURSE_OFFSET),
        "Output(3,1,\"WORD\")",
        *load_vector_to_c_lines(manifest, manifest["tensors"]["b_word"], 0, 8),
        *matvec_add_from_list_lines(manifest, "W_word_in", 8, 85, ENCODE_LIST, 0),
        "If P=0",
        "Then",
        *matvec_add_from_lx_lines(manifest, "W_word_h", 8, 8, WORD_OFFSET),
        "End",
        *matvec_add_from_lx_lines(manifest, "W_word_disc", 8, 16, DISCOURSE_OFFSET),
        *tanh_store_context_lines(8, WORD_OFFSET),
        "If W=2",
        "Then",
        "1->P",
        "Else",
        "0->P",
        "End",
        "Output(8,1,\"CTX READY\")",
        "Return",
    ]


def generate_step() -> list[str]:
    return [
        "prgmT2H1",
        "prgmT2H2",
        "prgmT2OGEN",
        "prgmT2REC",
        "prgmT2ENC",
        "prgmT2CTX",
        "Return",
    ]


def generate_gen() -> list[str]:
    return [
        "ClrHome",
        "Output(1,1,\"TILM2 GEN\")",
        *runtime_open_lines("|LID", "|LHO", "|LHU", "|LHC", "|LI"),
        "Fill(0,|LHO)",
        "Fill(0,|LHU)",
        "Fill(0,|LHC)",
        "For(K,1,10)",
        "|LI(5*K-4)->|LHO(K+5)",
        "|LI(5*K-3)->|LHU(K+5)",
        "|LI(5*K-2)->|LHC(K+5)",
        "End",
        "Fill(0,|LDR)",
        "Fill(0,|LDW)",
        "Fill(0,|LDS)",
        "Fill(0,|LDN)",
        "Fill(0,|LDO)",
        "Fill(0,|LDC)",
        "1->P",
        f"For(X,1,{OUTPUT_SYLLABLES})",
        "Output(2,1,\"SYL\")",
        "Disp X",
        "prgmT2STEP",
        "5*X-4->Y",
        "O->|LID(Y)",
        "U->|LID(Y+1)",
        "C->|LID(Y+2)",
        "S->|LID(Y+3)",
        "W->|LID(Y+4)",
        "For(I,1,14)",
        "|LHO(I+1)->|LHO(I)",
        "|LHU(I+1)->|LHU(I)",
        "|LHC(I+1)->|LHC(I)",
        "End",
        "O->|LHO(15)",
        "U->|LHU(15)",
        "C->|LHC(15)",
        "End",
        "Output(8,1,\"GEN DONE\")",
        "Pause ",
        "Return",
    ]


def generate_prob_debug() -> list[str]:
    return [
        "ClrHome",
        "Output(1,1,\"PROB DBG\")",
        "prgmT2H1",
        "prgmT2H2",
        "prgmT2OGEN",
        "ClrHome",
        "Disp \"IDS O U C S W\"",
        "Disp O",
        "Disp U",
        "Disp C",
        "Disp S",
        "Disp W",
        "Pause ",
        "ClrHome",
        "Disp \"ON 2 1 7 8\"",
        "Disp |LO(2)",
        "Disp |LO(1)",
        "Disp |LO(7)",
        "Disp |LO(8)",
        "Pause ",
        "ClrHome",
        "Disp \"NUC 6 10 12\"",
        "Disp |LN(6)",
        "Disp |LN(10)",
        "Disp |LN(12)",
        "Pause ",
        "ClrHome",
        "Disp \"COD 1 12\"",
        "Disp |LC(1)",
        "Disp |LC(12)",
        "Pause ",
        "Return",
    ]


def generate_recorder() -> list[str]:
    lines = [
        "If X<1",
        "Then",
        "Return",
        "End",
        f"If X>{OUTPUT_SYLLABLES}",
        "Then",
        "Return",
        "End",
    ]
    for src, dst, count in [
        ("|LR", "|LDR", 5),
        ("|LW", "|LDW", 2),
        ("|LS", "|LDS", 3),
        ("|LN", "|LDN", 14),
        ("|LO", "|LDO", 28),
        ("|LC", "|LDC", 24),
    ]:
        lines += [
            f"{count}*(X-1)->A",
            f"For(I,1,{count})",
            f"{src}(I)->{dst}(A+I)",
            "End",
        ]
    return lines


def debug_pages(list_name: str, total: int, title: str, base_expr: str, chunk: int = 6) -> list[str]:
    lines: list[str] = []
    for start in range(1, total + 1, chunk):
        end = min(total, start + chunk - 1)
        lines += [
            "ClrHome",
            f'Disp "{title}"',
            f'Disp "{start}-{end}"',
        ]
        for i in range(start, end + 1):
            lines.append(f"Disp {list_name}({base_expr}+{i})")
        lines += [
            "Pause ",
        ]
    return lines


def generate_debug_browser() -> list[str]:
    lines = [
        "ClrHome",
        "Disp \"TILM2 DEBUG\"",
        "Disp \"STEP 1-8\"",
        "Input X",
        "If X<1",
        "Then",
        "Return",
        "End",
        f"If X>{OUTPUT_SYLLABLES}",
        "Then",
        "Return",
        "End",
        "ClrHome",
        "Disp \"RAW IDS\"",
        "Disp \"O U C S W\"",
        "Disp |LID(5*X-4)",
        "Disp |LID(5*X-3)",
        "Disp |LID(5*X-2)",
        "Disp |LID(5*X-1)",
        "Disp |LID(5*X)",
        "Pause ",
    ]
    lines += debug_pages("|LDR", 5, "ROLE", "5*(X-1)")
    lines += debug_pages("|LDW", 2, "WB", "2*(X-1)")
    lines += debug_pages("|LDS", 3, "STR", "3*(X-1)")
    lines += debug_pages("|LDO", 28, "ONS", "28*(X-1)")
    lines += debug_pages("|LDN", 14, "NUC", "14*(X-1)")
    lines += debug_pages("|LDC", 24, "COD", "24*(X-1)")
    lines += ["Return"]
    return lines


def compare_block(current: str, reference: str, rows: int, label: str) -> list[str]:
    return [
        f"0->C",
        f"For(I,1,{rows})",
        f"abs({current}(I)-{reference}(I))->B",
        "If B>C",
        "Then",
        "B->C",
        "End",
        "End",
        f'Output(2,1,"{label}")',
        "Disp \"MAX DIFF\"",
        "Disp C",
        "Pause ",
    ]


def generate_compare_step1() -> list[str]:
    lines = [
        "ClrHome",
        "Disp \"CMP STEP1\"",
        "Disp \"REF LISTS\"",
        "Disp \"LRR LWR\"",
        "Disp \"LSR LNR\"",
        "Disp \"LOR LCR\"",
        "Pause ",
        "prgmT2H1",
        "prgmT2H2",
        "prgmT2OUT",
        "UnArchive |LR",
        "UnArchive |LW",
        "UnArchive |LS",
        "UnArchive |LN",
        "UnArchive |LO",
        "UnArchive |LC",
        "UnArchive |LDRR",
        "UnArchive |LDWR",
        "UnArchive |LDSR",
        "UnArchive |LDNR",
        "UnArchive |LDOR",
        "UnArchive |LDCR",
        "0->A",
    ]
    lines += compare_block("|LR", "|LDRR", 5, "ROLE")
    lines += compare_block("|LW", "|LDWR", 2, "WB")
    lines += compare_block("|LS", "|LDSR", 3, "STR")
    lines += compare_block("|LN", "|LDNR", 14, "NUC")
    lines += compare_block("|LO", "|LDOR", 28, "ONS")
    lines += compare_block("|LC", "|LDCR", 24, "COD")
    lines += [
        "Output(8,1,\"MAX\")",
        "Disp A",
        "Archive |LDRR",
        "Archive |LDWR",
        "Archive |LDSR",
        "Archive |LDNR",
        "Archive |LDOR",
        "Archive |LDCR",
        "Pause ",
        "Return",
    ]
    return lines


def generate_main(manifest: dict[str, Any]) -> list[str]:
    has_load_debug = "W_h_r0_c0" in manifest["tensors"]
    load_menu = "Menu(\"TILM2\",\"INIT\",1,\"INFO\",2,\"SMOKE\",3,\"LOAD W00\",4,\"RUN\",5,\"SEED\",12,\"DRAW\",13,\"QUIT\",8)"
    no_load_menu = "Menu(\"TILM2\",\"INIT\",1,\"INFO\",2,\"SMOKE\",3,\"RUN\",5,\"SEED\",12,\"DRAW\",13,\"QUIT\",8)"
    lines = [
        ".5->T",
        "3->K",
        ".5->G",
        "1->P",
        "Lbl 0",
        load_menu if has_load_debug else no_load_menu,
        "Lbl 1",
        "prgmT2INIT",
        "Goto 0",
        "Lbl 2",
        "prgmT2INFO",
        "Goto 0",
        "Lbl 3",
        "prgmT2SMOKE",
        "Goto 0",
    ]
    if has_load_debug:
        lines += [
        "Lbl 4",
        "prgmT2LA00",
        "Goto 0",
        ]
    lines += [
        "Lbl 5",
        "Menu(\"RUN\",\"H1\",6,\"H2\",7,\"OUT\",9,\"GEN\",11,\"CHK\",10,\"DBG\",14,\"CMP\",15,\"BACK\",0,\"QUIT\",8)",
        "Lbl 6",
        "prgmT2H1",
        "Goto 0",
        "Lbl 7",
        "prgmT2H2",
        "Goto 0",
        "Lbl 9",
        "prgmT2OUT",
        "Goto 0",
        "Lbl 10",
        "prgmT2CHK",
        "prgmT2CKO",
        "Goto 0",
        "Lbl 11",
        "prgmT2GEN",
        "Goto 0",
        "Lbl 14",
        "prgmT2DBG",
        "Goto 0",
        "Lbl 15",
        "prgmT2CMP",
        "Goto 0",
        "Lbl 12",
        "prgmT2SEED",
        "Goto 0",
        "Lbl 13",
        "prgmT2DRAW",
        "Goto 0",
        "Lbl 8",
        "Stop",
    ]
    return lines


def write_readme(outdir: Path, manifest: dict[str, Any], include_debug: bool) -> None:
    programs = [
        "TILM2", "T2INIT", "T2INFO", "T2SMOKE", "T2GET", "T2H1", "T2H2",
        "T2OUT", "T2OGEN", "T2SAMP", "T2ENC", "T2CTX", "T2STEP", "T2GEN",
    ]
    if include_debug:
        programs += ["T2LA00", "T2REC", "T2PDBG", "T2DBG", "T2CMP", "T2CHK", "T2CKO"]

    readme = f"""# TILM2 TI-BASIC Runtime Export

Generated programs contain no JSON dependency. JSON is consumed only by
`generate_ti_basic_runtime.py` on the PC.

Transfer/import these calculator objects:

- Programs: {", ".join(f"`{name}`" for name in programs)}
- Lists: `L1` through `L6`, then `|LL7` through `{manifest['list_names'][-1]}`
- Test/generation support lists: `|LX`, `|LHR`, `|LMR`, `|LRR`, `|LWR`,
  `|LSR`, `|LNR`, `|LOR`, `|LCR`, `|LNM`, `|LCM1` through `|LCM10`

Current generated layer:

- `TILM2`: menu
- `T2INIT`: initializes scalar settings, scratch lists, and debug buffers
- `T2INFO`: displays compiled architecture/export facts
- `T2SMOKE`: checks representative list dimensions and reads first/last values
- `T2GET`: correctness-oriented flat parameter getter using `Z` input and `V` output
- `T2H1`: computes hidden layer 1 from a preloaded 874-value `|LX`
- `T2H2`: computes hidden layer 2 from `T2H1` output
- `T2OUT`: computes all forward-pass output-head probabilities from `|LM`
- `T2OGEN`: computes Python-generate-style sampled cascade probabilities/IDs
  with hard phonotactic masks
- `T2SAMP`: top-k samples component IDs from probability lists
- `T2ENC`: encodes sampled IDs into an 85-value token vector in `|LG`
- `T2CTX`: shifts `|LX` and updates discourse/word state
- `T2STEP`: runs one syllable step
- `T2GEN`: loops `N` syllables and writes sampled IDs to `|LID`

The runtime is list-paged: generated hot loops `UnArchive` only the packed list
chunk currently being read, accumulate into small RAM lists, and `Archive` the
chunk before moving on. No dense matrix workspace is required. `T2GEN` also
maintains 15-step repetition penalty histories in `|LHO`, `|LHU`, and `|LHC`;
scalar `G` is the repetition penalty. `T2ENC` currently uses `T2GET` for 85
embedding values per sampled token, which is acceptable as a first complete
implementation but can be optimized.

Debug capture keeps per-syllable snapshots in `|LDR`, `|LDW`, `|LDS`,
`|LDN`, `|LDO`, and `|LDC`, with `|LID` holding the sampled IDs. Run
{("Debug capture keeps per-syllable snapshots in `|LDR`, `|LDW`, `|LDS`, `|LDN`, `|LDO`, and `|LDC`, with `|LID` holding the sampled IDs. Run `T2DBG` after a generation pass to inspect one step at a time. `T2CMP` checks the step-1 debug lists against exported PC reference lists `|LDRR`, `|LDWR`, `|LDSR`, `|LDNR`, `|LDOR`, and `|LDCR`." if include_debug else "Debug and compare programs are omitted from the default runtime build to keep the transfer set small. Use the debug build only when parity inspection is needed.")} 
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exportdir", default="ti_packed_export")
    parser.add_argument("--outdir", default="ti_basic_runtime")
    parser.add_argument("--include-debug", action="store_true")
    args = parser.parse_args()

    exportdir = Path(args.exportdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((exportdir / "packed_manifest.json").read_text(encoding="utf-8"))
    vocab = json.loads((exportdir / "vocab.json").read_text(encoding="utf-8"))

    write_program(outdir, "TILM2", generate_main(manifest))
    write_program(outdir, "T2INIT", generate_init(manifest))
    write_program(outdir, "T2INFO", generate_info(manifest, vocab))
    write_program(outdir, "T2SMOKE", generate_smoke(manifest))
    write_program(outdir, "T2GET", generate_getter(manifest))
    if args.include_debug and "W_h_r0_c0" in manifest["tensors"]:
        write_program(outdir, "T2LA00", generate_load_matrix_a("W_h_r0_c0", manifest["tensors"]["W_h_r0_c0"]))
    write_program(outdir, "T2H1", generate_h1(manifest))
    write_program(outdir, "T2H2", generate_h2(manifest))
    write_program(outdir, "T2OUT", generate_output_heads(manifest))
    write_program(outdir, "T2OGEN", generate_generation_outputs(manifest))
    write_program(outdir, "T2SAMP", generate_sample())
    write_program(outdir, "T2ENC", generate_encode(manifest))
    write_program(outdir, "T2CTX", generate_context_update(manifest))
    write_program(outdir, "T2STEP", generate_step())
    write_program(outdir, "T2GEN", generate_gen())
    if args.include_debug:
        write_program(outdir, "T2REC", generate_recorder())
        write_program(outdir, "T2PDBG", generate_prob_debug())
        write_program(outdir, "T2DBG", generate_debug_browser())
        write_program(outdir, "T2CMP", generate_compare_step1())
        write_program(outdir, "T2CHK", generate_check_hidden())
        write_program(outdir, "T2CKO", generate_check_outputs())
    write_readme(outdir, manifest, args.include_debug)

    print(f"Wrote TI-BASIC runtime scaffold to {outdir}")
    programs = ["TILM2", "T2INIT", "T2INFO", "T2SMOKE", "T2GET", "T2H1", "T2H2", "T2OUT", "T2OGEN", "T2SAMP", "T2ENC", "T2CTX", "T2STEP", "T2GEN"]
    if args.include_debug:
        programs += ["T2LA00", "T2REC", "T2PDBG", "T2DBG", "T2CMP", "T2CHK", "T2CKO"]
    print("Programs: " + ", ".join(programs))
    print(f"Compiled {manifest['list_count']} list names and {manifest['total_values']} parameter slots into TI-BASIC.")


if __name__ == "__main__":
    main()
