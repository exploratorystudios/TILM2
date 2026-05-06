"""
PC simulator for a TI-84-style packed TILM2 runtime.

The goal is to preserve the trained desktop model while changing storage:
all parameters are flattened into calculator-style lists, then rehydrated one
tensor/chunk at a time for PC parity checks. This is not yet a TI-BASIC
program; it is the validation layer before writing one.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from tilm2_model import TILM2, Vocab


BUILTIN_LIST_NAMES = [f"L{i}" for i in range(1, 7)]


def custom_list_name(index: int) -> str:
    return f"|LL{index}"


def ti_list_names(count: int) -> list[str]:
    names: list[str] = []
    for i in range(count):
        if i < len(BUILTIN_LIST_NAMES):
            names.append(BUILTIN_LIST_NAMES[i])
        else:
            names.append(custom_list_name(i + 1))
    return names


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = x / max(temperature, 1e-6)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -10, 10))


@dataclass
class TensorRef:
    name: str
    shape: tuple[int, ...]
    start: int
    length: int


class PackedStore:
    def __init__(self, list_size: int = 999):
        self.list_size = list_size
        self.flat: list[float] = []
        self.refs: dict[str, TensorRef] = {}

    def add(self, name: str, arr: np.ndarray) -> None:
        flat = np.asarray(arr, dtype=np.float64).reshape(-1)
        start = len(self.flat)
        self.flat.extend(float(v) for v in flat)
        self.refs[name] = TensorRef(
            name=name,
            shape=tuple(int(v) for v in arr.shape),
            start=start,
            length=int(flat.shape[0]),
        )

    def tensor(self, name: str) -> np.ndarray:
        ref = self.refs[name]
        data = np.array(self.flat[ref.start:ref.start + ref.length], dtype=np.float64)
        return data.reshape(ref.shape)

    def list_count(self) -> int:
        return (len(self.flat) + self.list_size - 1) // self.list_size

    def manifest(self, precision: int | None = None) -> dict[str, Any]:
        names = ti_list_names(self.list_count())
        tensors = {}
        for name, ref in self.refs.items():
            first = ref.start // self.list_size
            last = (ref.start + ref.length - 1) // self.list_size if ref.length else first
            tensors[name] = {
                "shape": list(ref.shape),
                "start": ref.start,
                "length": ref.length,
                "segments": [
                    {
                        "list": names[i],
                        "offset": 1 if i != first else ref.start % self.list_size + 1,
                        "length": min((i + 1) * self.list_size, ref.start + ref.length) - max(i * self.list_size, ref.start),
                    }
                    for i in range(first, last + 1)
                ],
            }
        manifest = {
            "list_size": self.list_size,
            "list_count": self.list_count(),
            "list_names": names,
            "total_values": len(self.flat),
            "tensor_count": len(self.refs),
            "tensors": tensors,
        }
        if precision is not None:
            manifest["export_precision"] = precision
        return manifest

    def write(self, outdir: str, precision: int = 6) -> None:
        os.makedirs(outdir, exist_ok=True)
        names = ti_list_names(self.list_count())
        for i, name in enumerate(names):
            start = i * self.list_size
            end = min(start + self.list_size, len(self.flat))
            filename = list_filename(name)
            with open(os.path.join(outdir, filename), "w", encoding="utf-8") as f:
                f.write("\n".join(f"{v:.{precision}f}" for v in self.flat[start:end]))
        with open(os.path.join(outdir, "packed_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(self.manifest(precision=precision), f, indent=2)


def list_filename(name: str) -> str:
    return f"{name}.csv"


def matrix_filename(name: str) -> str:
    return f"{name}.csv"


def write_zero_matrix(path: str, rows: int, cols: int, precision: int) -> None:
    row = ",".join(f"{0.0:.{precision}f}" for _ in range(cols))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(row for _ in range(rows)))


def workspace_matrix_specs(model: TILM2, vocab: Vocab) -> list[dict[str, Any]]:
    max_logits = max(vocab.n_onsets, vocab.n_nuclei, vocab.n_codas, vocab.n_roles, 3, 2)
    return [
        {
            "name": "[A]",
            "shape": [model.chunk_size, model.chunk_size],
            "role": "staged weight matrix chunk",
        },
        {
            "name": "[B]",
            "shape": [model.chunk_size, 1],
            "role": "staged input or hidden vector chunk",
        },
        {
            "name": "[C]",
            "shape": [model.chunk_size, 1],
            "role": "partial accumulator vector",
        },
        {
            "name": "[D]",
            "shape": [model.chunk_size, model.n_chunks],
            "role": "h1 workspace, one 99-value hidden chunk per column",
        },
        {
            "name": "[E]",
            "shape": [model.chunk_size, model.n_chunks],
            "role": "h2 workspace, one 99-value hidden chunk per column",
        },
        {
            "name": "[F]",
            "shape": [max_logits, 1],
            "role": "logits/probability workspace for output heads",
        },
        {
            "name": "[G]",
            "shape": [model.token_dim, 1],
            "role": "encoded syllable/context token workspace",
        },
        {
            "name": "[H]",
            "shape": [model.discourse_state_dim, 1],
            "role": "discourse state workspace",
        },
        {
            "name": "[I]",
            "shape": [model.word_state_dim, 1],
            "role": "word state workspace",
        },
        {
            "name": "[J]",
            "shape": [model.chunk_size, 1],
            "role": "scratch vector workspace",
        },
    ]


def write_workspace_matrices(outdir: str, model: TILM2, vocab: Vocab, precision: int) -> None:
    specs = workspace_matrix_specs(model, vocab)
    for spec in specs:
        path = os.path.join(outdir, matrix_filename(spec["name"]))
        write_zero_matrix(path, spec["shape"][0], spec["shape"][1], precision)

    manifest = {
        "matrix_count": len(specs),
        "matrices": [
            {
                **spec,
                "path": matrix_filename(spec["name"]),
                "contains_parameters": False,
            }
            for spec in specs
        ],
        "note": "Model parameters are stored in lists. These matrices are runtime workspaces/staging buffers named for TI-BASIC.",
    }
    with open(os.path.join(outdir, "matrix_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_packed_store(exportdir: str) -> PackedStore:
    with open(os.path.join(exportdir, "packed_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    store = PackedStore(list_size=manifest["list_size"])
    flat: list[float] = []
    for name in manifest["list_names"]:
        path = os.path.join(exportdir, list_filename(name))
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            flat.extend(float(x) for x in text.replace(",", " ").split())
    store.flat = flat

    for name, ref in manifest["tensors"].items():
        store.refs[name] = TensorRef(
            name=name,
            shape=tuple(ref["shape"]),
            start=int(ref["start"]),
            length=int(ref["length"]),
        )
    return store


def pack_model(model: TILM2, list_size: int = 999) -> PackedStore:
    store = PackedStore(list_size=list_size)

    for i, row_chunks in enumerate(model.W_h_chunks):
        for j, chunk in enumerate(row_chunks):
            store.add(f"W_h_r{i}_c{j}", chunk)
    for i, row_chunks in enumerate(model.W_h2_chunks):
        for j, chunk in enumerate(row_chunks):
            store.add(f"W_h2_r{i}_c{j}", chunk)

    for base, chunks in [
        ("W_disc_h1", model.W_disc_h1_chunks),
        ("W_word_h1", model.W_word_h1_chunks),
        ("W_disc_h2", model.W_disc_h2_chunks),
        ("W_word_h2", model.W_word_h2_chunks),
        ("W_onset", model.W_onset_chunks),
        ("W_nucleus", model.W_nucleus_chunks),
        ("W_coda", model.W_coda_chunks),
        ("W_stress", model.W_stress_chunks),
        ("W_wb", model.W_wb_chunks),
        ("W_role", model.W_role_chunks),
    ]:
        for i, chunk in enumerate(chunks):
            store.add(f"{base}_{i}", chunk)

    for name in [
        "E_onset", "E_nucleus", "E_coda", "E_stress", "E_wb",
        "W_disc_in", "W_disc_h", "W_word_in", "W_word_h", "W_word_disc",
        "W_disc_onset", "W_disc_nucleus", "W_disc_coda", "W_disc_stress", "W_disc_wb", "W_disc_role",
        "W_word_onset", "W_word_nucleus", "W_word_coda", "W_word_stress", "W_word_wb", "W_word_role",
        "W_role_onset", "W_role_nucleus", "W_role_coda", "W_role_stress", "W_role_wb",
        "W_wb_onset", "W_wb_nucleus", "W_wb_coda",
        "W_nuc_gate", "W_stress_gate", "W_nuc_cond", "W_coda_cond",
        "b_h", "b_h2", "b_onset", "b_nucleus", "b_coda", "b_stress", "b_wb", "b_role",
        "b_disc", "b_word", "b_nuc_gate", "b_stress_gate",
    ]:
        store.add(name, getattr(model, name))

    return store


class PackedTILM2:
    def __init__(self, vocab: Vocab, store: PackedStore, hparams: dict[str, int]):
        self.vocab = vocab
        self.store = store
        self.context_len = hparams["context_len"]
        self.embed_dim = hparams["embed_dim"]
        self.hidden_dim = hparams["hidden_dim"]
        self.discourse_state_dim = hparams["discourse_state_dim"]
        self.word_state_dim = hparams["word_state_dim"]
        self.state_dim = self.discourse_state_dim + self.word_state_dim
        self.token_dim = 4 * self.embed_dim + 1
        self.input_dim = self.context_len * self.token_dim + self.state_dim
        self.chunk_size = hparams["chunk_size"]
        self.n_chunks = hparams["n_chunks"]
        self.n_input_col_chunks = hparams["n_input_col_chunks"]
        self.n_h_col_chunks = hparams["n_h_col_chunks"]
        self.input_col_starts = [j * self.chunk_size for j in range(self.n_input_col_chunks)] + [self.input_dim]
        self.h_col_starts = [j * self.chunk_size for j in range(self.n_chunks)] + [self.hidden_dim]
        self.stage_reads = 0
        self._on_nuc_mask = None
        self._on_nuc_cod_mask = None
        self._coda_freq_bonus = None

    def t(self, name: str) -> np.ndarray:
        self.stage_reads += 1
        return self.store.tensor(name)

    def encode_token(self, token: dict) -> np.ndarray:
        oi, ni, ci, si, wb = self.vocab.token_to_indices(token)
        return np.concatenate([
            self.t("E_onset")[oi],
            self.t("E_nucleus")[ni],
            self.t("E_coda")[ci],
            self.t("E_stress")[si],
            self.t("E_wb")[wb],
        ])

    def state_summaries_from_tokens(self, context: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        disc = np.zeros(self.discourse_state_dim)
        word = np.zeros(self.word_state_dim)
        prev_wb = 1
        for tok in context[-self.context_len:]:
            vec = self.encode_token(tok)
            word_prev = np.zeros_like(word) if prev_wb else word
            disc = tanh(self.t("W_disc_in") @ vec + self.t("W_disc_h") @ disc + self.t("b_disc"))
            word = tanh(self.t("W_word_in") @ vec + self.t("W_word_h") @ word_prev + self.t("W_word_disc") @ disc + self.t("b_word"))
            prev_wb = 1 if tok.get("word_boundary", False) else 0
        return disc, word

    def encode_context(self, context: list[dict]) -> np.ndarray:
        ctx = context[-self.context_len:]
        vecs = [self.encode_token(t) for t in ctx]
        while len(vecs) < self.context_len:
            vecs.insert(0, np.zeros(self.token_dim))
        disc, word = self.state_summaries_from_tokens(ctx)
        return np.concatenate(vecs + [disc, word])

    def head_logits(self, prefix: str, bias_name: str, h2: np.ndarray) -> np.ndarray:
        out = self.t(bias_name).copy()
        for i in range(self.n_h_col_chunks):
            s = i * self.chunk_size
            e = s + self.t(f"{prefix}_{i}").shape[1]
            out += self.t(f"{prefix}_{i}") @ h2[s:e]
        return out

    def forward(self, context: list[dict], temperature: float = 1.0) -> dict:
        self.stage_reads = 0
        x = self.encode_context(context)
        disc_state = x[-self.state_dim:-self.word_state_dim]
        word_state = x[-self.word_state_dim:]

        h1_parts = []
        for i in range(self.n_chunks):
            row0 = self.t(f"W_h_r{i}_c0")
            s = i * self.chunk_size
            e = s + row0.shape[0]
            pre = self.t("b_h")[s:e].copy()
            for j in range(self.n_input_col_chunks):
                chunk = self.t(f"W_h_r{i}_c{j}")
                pre += chunk @ x[self.input_col_starts[j]:self.input_col_starts[j + 1]]
            pre += self.t(f"W_disc_h1_{i}") @ disc_state + self.t(f"W_word_h1_{i}") @ word_state
            h1_parts.append(relu(pre))
        h1 = np.concatenate(h1_parts)

        h2_parts = []
        for i in range(self.n_chunks):
            row0 = self.t(f"W_h2_r{i}_c0")
            s = i * self.chunk_size
            e = s + row0.shape[0]
            pre = self.t("b_h2")[s:e].copy()
            for j in range(self.n_chunks):
                chunk = self.t(f"W_h2_r{i}_c{j}")
                pre += chunk @ h1[self.h_col_starts[j]:self.h_col_starts[j + 1]]
            pre += self.t(f"W_disc_h2_{i}") @ disc_state + self.t(f"W_word_h2_{i}") @ word_state
            h2_parts.append(relu(pre))
        h2 = np.concatenate(h2_parts)

        role_logits = self.head_logits("W_role", "b_role", h2)
        role_logits += self.t("W_disc_role") @ disc_state + self.t("W_word_role") @ word_state
        role_probs = softmax(role_logits, temperature)

        wb_logits = self.head_logits("W_wb", "b_wb", h2)
        wb_logits += self.t("W_disc_wb") @ disc_state + self.t("W_word_wb") @ word_state
        wb_logits += self.t("W_role_wb") @ role_probs
        wb_probs = softmax(wb_logits, temperature)

        stress_logits = self.head_logits("W_stress", "b_stress", h2)
        stress_logits += self.t("W_disc_stress") @ disc_state + self.t("W_word_stress") @ word_state
        stress_logits += self.t("W_role_stress") @ role_probs
        stress_probs = softmax(stress_logits, temperature)

        nuc_logits = self.head_logits("W_nucleus", "b_nucleus", h2)
        nuc_logits += self.t("W_disc_nucleus") @ disc_state + self.t("W_word_nucleus") @ word_state
        nuc_logits += self.t("W_role_nucleus") @ role_probs
        nuc_logits += self.t("W_wb_nucleus") @ wb_probs
        nuc_probs = softmax(nuc_logits, temperature)

        onset_logits = self.head_logits("W_onset", "b_onset", h2)
        onset_logits += self.t("W_disc_onset") @ disc_state + self.t("W_word_onset") @ word_state
        onset_logits += self.t("W_role_onset") @ role_probs
        onset_logits += self.t("W_wb_onset") @ wb_probs
        onset_logits += self.t("W_nuc_gate") @ nuc_probs + self.t("b_nuc_gate")
        onset_logits += self.t("W_stress_gate") @ stress_probs + self.t("b_stress_gate")
        onset_probs = softmax(onset_logits, temperature)

        coda_logits = self.head_logits("W_coda", "b_coda", h2)
        coda_logits += self.t("W_disc_coda") @ disc_state + self.t("W_word_coda") @ word_state
        coda_logits += self.t("W_role_coda") @ role_probs
        coda_logits += self.t("W_wb_coda") @ wb_probs
        coda_probs = softmax(coda_logits, temperature)

        return {
            "hidden": h2,
            "onset_probs": onset_probs,
            "nucleus_probs": nuc_probs,
            "coda_probs": coda_probs,
            "stress_probs": stress_probs,
            "wb_probs": wb_probs,
            "role_probs": role_probs,
            "stage_reads": self.stage_reads,
        }

    def build_phonotactic_masks(
        self,
        windows: list[dict],
        min_freq: int = 3,
        freq_weight: float = 0.5,
    ) -> None:
        v = self.vocab
        counts = np.zeros((v.n_onsets, v.n_nuclei, v.n_codas), dtype=np.int32)
        for w in windows:
            for tok in w["context"] + [w["target"]]:
                oi, ni, ci, *_ = v.token_to_indices(tok)
                counts[oi, ni, ci] += 1
        self._on_nuc_mask = counts.any(axis=2) if min_freq <= 1 else (counts >= min_freq).any(axis=2)
        self._on_nuc_cod_mask = counts >= min_freq
        self._coda_freq_bonus = np.log1p(counts).astype(np.float64) * freq_weight

    def sample_token(self, probs: np.ndarray, top_k: int = 0) -> int:
        if top_k > 0:
            top_idx = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs)
            mask[top_idx] = probs[top_idx]
            if mask.sum() > 0:
                probs = mask / mask.sum()
        return int(np.random.choice(len(probs), p=probs))

    def generate(
        self,
        seed_context: list[dict],
        n_syllables: int = 20,
        temperature: float = 1.0,
        top_k: int = 5,
        rep_penalty: float = 0.5,
        rep_window: int = 15,
    ) -> list[dict]:
        context = list(seed_context)[-self.context_len:]
        generated = []
        history: list[dict] = list(seed_context)

        def apply_rep_penalty(logits: np.ndarray, component: str) -> np.ndarray:
            if rep_penalty <= 0 or not history:
                return logits
            logits = logits.copy()
            idx_map = {
                "onset": self.vocab.onset_to_idx,
                "nucleus": self.vocab.nucleus_to_idx,
                "coda": self.vocab.coda_to_idx,
            }[component]
            for tok in history[-rep_window:]:
                idx = idx_map.get(tok[component], 0)
                logits[idx] -= rep_penalty
            return logits

        for _ in range(n_syllables):
            x = self.encode_context(context)
            disc_state = x[-self.state_dim:-self.word_state_dim]
            word_state = x[-self.word_state_dim:]

            h1_parts = []
            for i in range(self.n_chunks):
                row0 = self.t(f"W_h_r{i}_c0")
                s = i * self.chunk_size
                e = s + row0.shape[0]
                pre = self.t("b_h")[s:e].copy()
                for j in range(self.n_input_col_chunks):
                    chunk = self.t(f"W_h_r{i}_c{j}")
                    pre += chunk @ x[self.input_col_starts[j]:self.input_col_starts[j + 1]]
                pre += self.t(f"W_disc_h1_{i}") @ disc_state + self.t(f"W_word_h1_{i}") @ word_state
                h1_parts.append(relu(pre))
            h1 = np.concatenate(h1_parts)

            h2_parts = []
            for i in range(self.n_chunks):
                row0 = self.t(f"W_h2_r{i}_c0")
                s = i * self.chunk_size
                e = s + row0.shape[0]
                pre = self.t("b_h2")[s:e].copy()
                for j in range(self.n_chunks):
                    chunk = self.t(f"W_h2_r{i}_c{j}")
                    pre += chunk @ h1[self.h_col_starts[j]:self.h_col_starts[j + 1]]
                pre += self.t(f"W_disc_h2_{i}") @ disc_state + self.t(f"W_word_h2_{i}") @ word_state
                h2_parts.append(relu(pre))
            h2 = np.concatenate(h2_parts)

            role_logits = self.head_logits("W_role", "b_role", h2)
            role_logits += self.t("W_disc_role") @ disc_state + self.t("W_word_role") @ word_state
            role_probs = softmax(role_logits, temperature)

            wb_logits = self.head_logits("W_wb", "b_wb", h2)
            wb_logits += self.t("W_disc_wb") @ disc_state + self.t("W_word_wb") @ word_state
            wb_logits += self.t("W_role_wb") @ role_probs
            wb_probs = softmax(wb_logits, temperature)
            wb_idx = self.sample_token(wb_probs, top_k)
            wb_feat = np.zeros(2, dtype=np.float64)
            wb_feat[wb_idx] = 1.0

            stress_logits = self.head_logits("W_stress", "b_stress", h2)
            stress_logits += self.t("W_disc_stress") @ disc_state + self.t("W_word_stress") @ word_state
            stress_logits += self.t("W_role_stress") @ role_probs
            stress_probs = softmax(stress_logits, temperature)
            stress_idx = self.sample_token(stress_probs, top_k)

            nuc_raw_logits = self.head_logits("W_nucleus", "b_nucleus", h2)
            nuc_raw_logits += self.t("W_disc_nucleus") @ disc_state + self.t("W_word_nucleus") @ word_state
            nuc_raw_logits += self.t("W_role_nucleus") @ role_probs
            nuc_raw_logits += self.t("W_wb_nucleus") @ wb_feat
            nuc_raw = softmax(nuc_raw_logits, temperature)

            onset_logits = self.head_logits("W_onset", "b_onset", h2)
            onset_logits += self.t("W_disc_onset") @ disc_state + self.t("W_word_onset") @ word_state
            onset_logits += self.t("W_role_onset") @ role_probs
            onset_logits += self.t("W_wb_onset") @ wb_feat
            onset_logits += self.t("W_nuc_gate") @ nuc_raw + self.t("b_nuc_gate")
            onset_logits += self.t("W_stress_gate") @ stress_probs + self.t("b_stress_gate")
            onset_logits = apply_rep_penalty(onset_logits, "onset")
            onset_probs = softmax(onset_logits, temperature)
            onset_idx = self.sample_token(onset_probs, top_k)
            onset_emb = self.t("E_onset")[onset_idx]

            nuc_logits = self.head_logits("W_nucleus", "b_nucleus", h2)
            nuc_logits += self.t("W_disc_nucleus") @ disc_state + self.t("W_word_nucleus") @ word_state
            nuc_logits += self.t("W_role_nucleus") @ role_probs
            nuc_logits += self.t("W_wb_nucleus") @ wb_feat
            nuc_logits += onset_emb @ self.t("W_nuc_cond")
            if self._on_nuc_mask is not None:
                valid = self._on_nuc_mask[onset_idx]
                if valid.any():
                    nuc_logits[~valid] = -1e9
            nuc_logits = apply_rep_penalty(nuc_logits, "nucleus")
            nuc_probs = softmax(nuc_logits, temperature)
            nuc_idx = self.sample_token(nuc_probs, top_k)
            nuc_emb = self.t("E_nucleus")[nuc_idx]

            coda_logits = self.head_logits("W_coda", "b_coda", h2)
            coda_logits += self.t("W_disc_coda") @ disc_state + self.t("W_word_coda") @ word_state
            coda_logits += self.t("W_role_coda") @ role_probs
            coda_logits += self.t("W_wb_coda") @ wb_feat
            coda_logits += np.concatenate([onset_emb, nuc_emb]) @ self.t("W_coda_cond")
            if self._coda_freq_bonus is not None:
                coda_logits += self._coda_freq_bonus[onset_idx, nuc_idx]
            if self._on_nuc_cod_mask is not None:
                valid = self._on_nuc_cod_mask[onset_idx, nuc_idx]
                if valid.any():
                    coda_logits[~valid] = -1e9
            coda_logits = apply_rep_penalty(coda_logits, "coda")
            coda_probs = softmax(coda_logits, temperature)
            coda_idx = self.sample_token(coda_probs, top_k)

            token = {
                "onset": self.vocab.onsets[onset_idx],
                "nucleus": self.vocab.nuclei[nuc_idx],
                "coda": self.vocab.codas[coda_idx],
                "stress": stress_idx,
                "word_boundary": bool(wb_idx),
            }
            generated.append(token)
            history.append(token)
            context = (context + [token])[-self.context_len:]

        return generated


def load_model(weights: str, data_path: str, context_len: int, embed_dim: int, hidden_dim: int) -> tuple[TILM2, Vocab, dict]:
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    vocab_dict = dict(data["vocab"])
    if "role_vocab" in data:
        vocab_dict["role_vocab"] = data["role_vocab"]
    vocab = Vocab(vocab_dict)
    model = TILM2(vocab, context_len=context_len, embed_dim=embed_dim, hidden_dim=hidden_dim)
    model.load(weights)
    return model, vocab, data


def compare_forward(model: TILM2, packed: PackedTILM2, context: list[dict], temperature: float) -> dict[str, float]:
    ref = model.forward(context, temperature=temperature)
    got = packed.forward(context, temperature=temperature)
    keys = ["hidden", "role_probs", "wb_probs", "stress_probs", "onset_probs", "nucleus_probs", "coda_probs"]
    return {key: float(np.max(np.abs(ref[key] - got[key]))) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--list-size", type=int, default=999)
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=21)
    parser.add_argument("--hidden-dim", type=int, default=198)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, vocab, data = load_model(args.weights, args.data, args.context_len, args.embed_dim, args.hidden_dim)
    store = pack_model(model, list_size=args.list_size)
    hparams = {
        "context_len": model.context_len,
        "embed_dim": model.embed_dim,
        "hidden_dim": model.hidden_dim,
        "discourse_state_dim": model.discourse_state_dim,
        "word_state_dim": model.word_state_dim,
        "chunk_size": model.chunk_size,
        "n_chunks": model.n_chunks,
        "n_input_col_chunks": model.n_input_col_chunks,
        "n_h_col_chunks": model.n_h_col_chunks,
    }
    packed = PackedTILM2(vocab, store, hparams)

    window = data["windows"][args.window_index]
    context = window["context"][-model.context_len:]
    diffs = compare_forward(model, packed, context, args.temperature)

    print("Packed TI-layout simulator")
    print(f"  total values: {len(store.flat)}")
    print(f"  list size:    {store.list_size}")
    print(f"  list count:   {store.list_count()}")
    print(f"  tensor count: {len(store.refs)}")
    print(f"  stage reads for one forward: {packed.stage_reads}")
    print("  max abs diffs:")
    for key, value in diffs.items():
        print(f"    {key:<14} {value:.12g}")

    if args.outdir:
        store.write(args.outdir, precision=args.precision)
        write_workspace_matrices(args.outdir, model, vocab, precision=args.precision)
        with open(os.path.join(args.outdir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump({
                "onsets": vocab.onsets,
                "nuclei": vocab.nuclei,
                "codas": vocab.codas,
                "roles": vocab.roles,
            }, f, indent=2)
        print(f"  wrote packed lists to {args.outdir}")


if __name__ == "__main__":
    main()
