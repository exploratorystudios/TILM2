"""
TILM2 neural network — syllable-level autoregressive language model.

Architecture (all fits within TI-84 99×99 matrix constraints via factored heads):

  Input: context_len × syllable_embedding_dim  (concatenated embeddings)
  Hidden: 99 neurons (sigmoid)
  Output heads (factored — reconstruct syllable from components):
    Head 1: Onset     (~30 classes)
    Head 2: Nucleus   (~16 classes)
    Head 3: Coda      (~25 classes)
    Head 4: Stress    (3 classes: 0=unstressed, 1=primary, 2=secondary)
    Head 5: Word-boundary (2 classes)

  Phonotactic gate: nucleus probs modulate onset head
  Stress gate:      stress probs modulate onset head

TI-84 constraint: all matrices must be ≤ 99×99.
  Default: context_len=3, embed_dim=8 → input_dim = 3×33 = 99 (exact fit).
  Larger configs auto-split W_h into ≤99-column chunk matrices.

Training uses mini-batches for speed — same math, gradients averaged over the batch.
"""

import numpy as np
import json
import argparse
import time

# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------

class Vocab:
    def __init__(self, vocab_dict: dict):
        self.onsets  = vocab_dict["onsets"]
        self.nuclei  = vocab_dict["nuclei"]
        self.codas   = vocab_dict["codas"]
        self.roles   = vocab_dict.get("roles") or vocab_dict.get("role_vocab") or ["other"]

        self.onset_to_idx   = {v: i for i, v in enumerate(self.onsets)}
        self.nucleus_to_idx = {v: i for i, v in enumerate(self.nuclei)}
        self.coda_to_idx    = {v: i for i, v in enumerate(self.codas)}
        self.role_to_idx    = {v: i for i, v in enumerate(self.roles)}

        self.n_onsets  = len(self.onsets)
        self.n_nuclei  = len(self.nuclei)
        self.n_codas   = len(self.codas)
        self.n_stress  = 3
        self.n_wb      = 2
        self.n_roles   = len(self.roles)

    def token_to_indices(self, token: dict) -> tuple[int, int, int, int, int]:
        oi = self.onset_to_idx.get(token["onset"], 0)
        ni = self.nucleus_to_idx.get(token["nucleus"], 0)
        ci = self.coda_to_idx.get(token["coda"], 0)
        si = min(token["stress"], 2)
        wb = 1 if token.get("word_boundary", False) else 0
        return oi, ni, ci, si, wb

    def role_to_index(self, role: str) -> int:
        return self.role_to_idx.get(role, 0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TILM2:
    def __init__(
        self,
        vocab: Vocab,
        context_len: int = 10,
        embed_dim: int = 21,
        hidden_dim: int = 198,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.vocab       = vocab
        self.context_len = context_len
        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim
        self._on_nuc_mask     = None   # (n_onsets, n_nuclei) bool
        self._on_nuc_cod_mask = None   # (n_onsets, n_nuclei, n_codas) bool
        self._coda_freq_bonus = None   # (n_onsets, n_nuclei, n_codas) float — log-freq bonus

        self.token_dim = 4 * embed_dim + 1
        self.discourse_state_dim = 16
        self.word_state_dim = 8
        self.role_loss_weight = 0.35
        self.state_dim = self.discourse_state_dim + self.word_state_dim
        self.input_dim = context_len * self.token_dim + self.state_dim

        scale = np.sqrt(2.0 / self.input_dim)  # He init — correct scale for ReLU
        self.E_onset   = np.random.randn(vocab.n_onsets,  embed_dim) * scale
        self.E_nucleus = np.random.randn(vocab.n_nuclei,  embed_dim) * scale
        self.E_coda    = np.random.randn(vocab.n_codas,   embed_dim) * scale
        self.E_stress  = np.random.randn(vocab.n_stress,  embed_dim) * scale
        self.E_wb      = np.random.randn(2, 1) * scale

        # W_h: 2D grid of ≤99×99 chunks [row_chunk][col_chunk].
        # Row-chunked for hidden_dim > 99; col-chunked for input_dim > 99.
        # h[row_i] = sigmoid( sum_j( W_h[row_i][col_j] @ x[col_j_slice] ) + b_h )
        self.chunk_size         = 99
        self.n_chunks           = (hidden_dim + self.chunk_size - 1) // self.chunk_size
        self.n_input_col_chunks = (self.input_dim + self.chunk_size - 1) // self.chunk_size
        self.input_col_starts   = [j * self.chunk_size for j in range(self.n_input_col_chunks)] + [self.input_dim]
        self.W_h_chunks: list[list[np.ndarray]] = []
        for i in range(self.n_chunks):
            s_row = i * self.chunk_size
            e_row = min(s_row + self.chunk_size, hidden_dim)
            row_chunks = []
            for j in range(self.n_input_col_chunks):
                s_col = self.input_col_starts[j]
                e_col = self.input_col_starts[j + 1]
                row_chunks.append(np.random.randn(e_row - s_row, e_col - s_col) * scale)
            self.W_h_chunks.append(row_chunks)
        self.b_h = np.zeros(hidden_dim)
        self.W_disc_h1_chunks = []
        self.W_word_h1_chunks = []
        for i in range(self.n_chunks):
            s_row = i * self.chunk_size
            e_row = min(s_row + self.chunk_size, hidden_dim)
            rows = e_row - s_row
            self.W_disc_h1_chunks.append(np.random.randn(rows, self.discourse_state_dim) * scale * 0.1)
            self.W_word_h1_chunks.append(np.random.randn(rows, self.word_state_dim) * scale * 0.1)

        # W_h2: second hidden layer (hidden_dim → hidden_dim), 2D-chunked.
        # Both row and col chunks align with n_chunks (same hidden_dim on both sides).
        self.h_col_starts = [j * self.chunk_size for j in range(self.n_chunks)] + [self.hidden_dim]
        self.W_h2_chunks: list[list[np.ndarray]] = []
        for i in range(self.n_chunks):
            s_row = i * self.chunk_size
            e_row = min(s_row + self.chunk_size, hidden_dim)
            row_chunks = []
            for j in range(self.n_chunks):
                s_col = self.h_col_starts[j]
                e_col = self.h_col_starts[j + 1]
                row_chunks.append(np.random.randn(e_row - s_row, e_col - s_col) * scale)
            self.W_h2_chunks.append(row_chunks)
        self.b_h2 = np.zeros(hidden_dim)
        self.W_disc_h2_chunks = []
        self.W_word_h2_chunks = []
        for i in range(self.n_chunks):
            s_row = i * self.chunk_size
            e_row = min(s_row + self.chunk_size, hidden_dim)
            rows = e_row - s_row
            self.W_disc_h2_chunks.append(np.random.randn(rows, self.discourse_state_dim) * scale * 0.1)
            self.W_word_h2_chunks.append(np.random.randn(rows, self.word_state_dim) * scale * 0.1)

        # Output heads — column-chunked into ≤99-column slices for TI-84.
        # Each head is a list of (n_classes × chunk_size) matrices.
        # logits = sum(chunk @ H[:, s:e].T) + bias
        def make_head(n_classes: int) -> list[np.ndarray]:
            chunks = []
            for i in range(self.n_h_col_chunks):
                s = i * self.chunk_size
                e = min(s + self.chunk_size, hidden_dim)
                chunks.append(np.random.randn(n_classes, e - s) * scale)
            return chunks

        self.n_h_col_chunks = (hidden_dim + self.chunk_size - 1) // self.chunk_size

        self.W_onset_chunks  = make_head(vocab.n_onsets)
        self.b_onset         = np.zeros(vocab.n_onsets)
        self.W_nucleus_chunks = make_head(vocab.n_nuclei)
        self.b_nucleus        = np.zeros(vocab.n_nuclei)
        self.W_coda_chunks   = make_head(vocab.n_codas)
        self.b_coda          = np.zeros(vocab.n_codas)
        self.W_stress_chunks = make_head(vocab.n_stress)
        self.b_stress        = np.zeros(vocab.n_stress)
        self.W_wb_chunks     = make_head(2)
        self.b_wb            = np.zeros(2)
        self.W_role_chunks   = make_head(vocab.n_roles)
        self.b_role          = np.zeros(vocab.n_roles)

        # Tiny state encoders: a discourse track plus a resettable within-word track.
        self.W_disc_in   = np.random.randn(self.discourse_state_dim, self.token_dim) * scale * 0.1
        self.W_disc_h    = np.random.randn(self.discourse_state_dim, self.discourse_state_dim) * scale * 0.1
        self.b_disc      = np.zeros(self.discourse_state_dim)
        self.W_word_in   = np.random.randn(self.word_state_dim, self.token_dim) * scale * 0.1
        self.W_word_h    = np.random.randn(self.word_state_dim, self.word_state_dim) * scale * 0.1
        self.W_word_disc = np.random.randn(self.word_state_dim, self.discourse_state_dim) * scale * 0.1
        self.b_word      = np.zeros(self.word_state_dim)

        # Direct state conditioning helps keep sentence role and word assembly coherent.
        self.W_disc_onset   = np.random.randn(vocab.n_onsets,  self.discourse_state_dim) * scale * 0.1
        self.W_disc_nucleus = np.random.randn(vocab.n_nuclei,  self.discourse_state_dim) * scale * 0.1
        self.W_disc_coda    = np.random.randn(vocab.n_codas,   self.discourse_state_dim) * scale * 0.1
        self.W_disc_stress  = np.random.randn(vocab.n_stress,  self.discourse_state_dim) * scale * 0.1
        self.W_disc_wb      = np.random.randn(2,               self.discourse_state_dim) * scale * 0.1
        self.W_disc_role    = np.random.randn(vocab.n_roles,   self.discourse_state_dim) * scale * 0.1
        self.W_word_onset   = np.random.randn(vocab.n_onsets,  self.word_state_dim) * scale * 0.1
        self.W_word_nucleus = np.random.randn(vocab.n_nuclei,  self.word_state_dim) * scale * 0.1
        self.W_word_coda    = np.random.randn(vocab.n_codas,   self.word_state_dim) * scale * 0.1
        self.W_word_stress  = np.random.randn(vocab.n_stress,  self.word_state_dim) * scale * 0.1
        self.W_word_wb      = np.random.randn(2,               self.word_state_dim) * scale * 0.1
        self.W_word_role    = np.random.randn(vocab.n_roles,   self.word_state_dim) * scale * 0.1

        self.W_role_onset   = np.random.randn(vocab.n_onsets,  vocab.n_roles) * scale * 0.1
        self.W_role_nucleus = np.random.randn(vocab.n_nuclei,  vocab.n_roles) * scale * 0.1
        self.W_role_coda    = np.random.randn(vocab.n_codas,   vocab.n_roles) * scale * 0.1
        self.W_role_stress  = np.random.randn(vocab.n_stress,  vocab.n_roles) * scale * 0.1
        self.W_role_wb      = np.random.randn(2,               vocab.n_roles) * scale * 0.1

        # Boundary is predicted first and then conditions the syllable heads.
        self.W_wb_onset   = np.random.randn(vocab.n_onsets, 2) * scale * 0.1
        self.W_wb_nucleus = np.random.randn(vocab.n_nuclei, 2) * scale * 0.1
        self.W_wb_coda    = np.random.randn(vocab.n_codas,  2) * scale * 0.1

        # Gates connect nucleus/stress → onset; independent of hidden_dim
        self.W_nuc_gate    = np.random.randn(vocab.n_onsets, vocab.n_nuclei) * scale * 0.1
        self.b_nuc_gate    = np.zeros(vocab.n_onsets)
        self.W_stress_gate = np.random.randn(vocab.n_onsets, vocab.n_stress) * scale * 0.1
        self.b_stress_gate = np.zeros(vocab.n_onsets)

        # Within-syllable autoregressive conditioning (both well within 99×99)
        # W_nuc_cond:  (embed_dim, n_nuclei)  = (7, 14)  — onset embedding → nucleus logits
        # W_coda_cond: (2*embed_dim, n_codas) = (14, 90) — [onset_emb, nuc_emb] → coda logits
        self.W_nuc_cond  = np.random.randn(embed_dim,     vocab.n_nuclei) * scale * 0.1
        self.W_coda_cond = np.random.randn(2 * embed_dim, vocab.n_codas)  * scale * 0.1

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        x = x / max(temperature, 1e-6)
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def _softmax_batch(self, x: np.ndarray) -> np.ndarray:
        """Softmax over last axis for a (B, C) matrix."""
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    def _encode_token(self, token: dict) -> np.ndarray:
        oi, ni, ci, si, wb = self.vocab.token_to_indices(token)
        return np.concatenate([
            self.E_onset[oi], self.E_nucleus[ni],
            self.E_coda[ci],  self.E_stress[si], self.E_wb[wb],
        ])

    def _tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(np.clip(x, -10, 10))

    def _state_summaries_from_tokens(self, context: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        disc = np.zeros(self.discourse_state_dim)
        word = np.zeros(self.word_state_dim)
        prev_wb = 1
        for tok in context[-self.context_len:]:
            vec = self._encode_token(tok)
            if prev_wb:
                word_prev = np.zeros_like(word)
            else:
                word_prev = word
            disc = self._tanh(self.W_disc_in @ vec + self.W_disc_h @ disc + self.b_disc)
            word = self._tanh(self.W_word_in @ vec + self.W_word_h @ word_prev + self.W_word_disc @ disc + self.b_word)
            prev_wb = 1 if tok.get("word_boundary", False) else 0
        return disc, word

    def _encode_context(self, context: list[dict]) -> np.ndarray:
        ctx = context[-self.context_len:]
        vecs = [self._encode_token(t) for t in ctx]
        while len(vecs) < self.context_len:
            vecs.insert(0, np.zeros(self.token_dim))
        disc, word = self._state_summaries_from_tokens(ctx)
        return np.concatenate(vecs + [disc, word])

    # -----------------------------------------------------------------------
    # Shared helpers for inference
    # -----------------------------------------------------------------------

    def _h2_from_context(self, context: list[dict]) -> np.ndarray:
        x = self._encode_context(context)
        disc_state = x[-self.state_dim:-self.word_state_dim]
        word_state = x[-self.word_state_dim:]
        h1_parts = []
        for i, row_chunks in enumerate(self.W_h_chunks):
            s = i * self.chunk_size; e = s + row_chunks[0].shape[0]
            pre = self.b_h[s:e].copy()
            for j, chunk in enumerate(row_chunks):
                pre += chunk @ x[self.input_col_starts[j]:self.input_col_starts[j + 1]]
            pre += self.W_disc_h1_chunks[i] @ disc_state + self.W_word_h1_chunks[i] @ word_state
            h1_parts.append(self._relu(pre))
        h1 = np.concatenate(h1_parts)
        h2_parts = []
        for i, row_chunks in enumerate(self.W_h2_chunks):
            s = i * self.chunk_size; e = s + row_chunks[0].shape[0]
            pre = self.b_h2[s:e].copy()
            for j, chunk in enumerate(row_chunks):
                pre += chunk @ h1[self.h_col_starts[j]:self.h_col_starts[j + 1]]
            pre += self.W_disc_h2_chunks[i] @ disc_state + self.W_word_h2_chunks[i] @ word_state
            h2_parts.append(self._relu(pre))
        return np.concatenate(h2_parts)

    def _head_logits(self, chunks: list, bias: np.ndarray, h2: np.ndarray) -> np.ndarray:
        out = bias.copy()
        for i, chunk in enumerate(chunks):
            s = i * self.chunk_size; e = s + chunk.shape[1]
            out += chunk @ h2[s:e]
        return out

    # -----------------------------------------------------------------------
    # Single-sample forward (inference only)
    # -----------------------------------------------------------------------

    def forward(self, context: list[dict], temperature: float = 1.0) -> dict:
        x = self._encode_context(context)
        disc_state = x[-self.state_dim:-self.word_state_dim]
        word_state = x[-self.word_state_dim:]

        h1_parts = []
        for i, row_chunks in enumerate(self.W_h_chunks):
            s = i * self.chunk_size
            e = s + row_chunks[0].shape[0]
            pre = self.b_h[s:e].copy()
            for j, chunk in enumerate(row_chunks):
                pre += chunk @ x[self.input_col_starts[j]:self.input_col_starts[j + 1]]
            pre += self.W_disc_h1_chunks[i] @ disc_state + self.W_word_h1_chunks[i] @ word_state
            h1_parts.append(self._relu(pre))
        h1 = np.concatenate(h1_parts)

        h2_parts = []
        for i, row_chunks in enumerate(self.W_h2_chunks):
            s = i * self.chunk_size
            e = s + row_chunks[0].shape[0]
            pre = self.b_h2[s:e].copy()
            for j, chunk in enumerate(row_chunks):
                pre += chunk @ h1[self.h_col_starts[j]:self.h_col_starts[j + 1]]
            pre += self.W_disc_h2_chunks[i] @ disc_state + self.W_word_h2_chunks[i] @ word_state
            h2_parts.append(self._relu(pre))
        h2 = np.concatenate(h2_parts)

        def head_logits_1d(chunks, bias):
            out = bias.copy()
            for i, chunk in enumerate(chunks):
                s = i * self.chunk_size
                e = s + chunk.shape[1]
                out += chunk @ h2[s:e]
            return out

        role_logits = head_logits_1d(self.W_role_chunks, self.b_role)
        role_logits += self.W_disc_role @ disc_state + self.W_word_role @ word_state
        role_probs = self._softmax(role_logits, temperature)

        wb_logits   = head_logits_1d(self.W_wb_chunks, self.b_wb)
        wb_logits  += self.W_disc_wb @ disc_state + self.W_word_wb @ word_state
        wb_logits  += self.W_role_wb @ role_probs
        wb_probs    = self._softmax(wb_logits, temperature)

        stress_logits = head_logits_1d(self.W_stress_chunks, self.b_stress)
        stress_logits += self.W_disc_stress @ disc_state + self.W_word_stress @ word_state
        stress_logits += self.W_role_stress @ role_probs
        stress_probs = self._softmax(stress_logits, temperature)

        nuc_logits   = head_logits_1d(self.W_nucleus_chunks, self.b_nucleus)
        nuc_logits  += self.W_disc_nucleus @ disc_state + self.W_word_nucleus @ word_state
        nuc_logits  += self.W_role_nucleus @ role_probs
        nuc_logits  += self.W_wb_nucleus @ wb_probs
        nuc_probs    = self._softmax(nuc_logits, temperature)

        onset_logits  = head_logits_1d(self.W_onset_chunks, self.b_onset)
        onset_logits += self.W_disc_onset @ disc_state + self.W_word_onset @ word_state
        onset_logits += self.W_role_onset @ role_probs
        onset_logits += self.W_wb_onset @ wb_probs
        onset_logits += self.W_nuc_gate @ nuc_probs + self.b_nuc_gate
        onset_logits += self.W_stress_gate @ stress_probs + self.b_stress_gate
        onset_probs   = self._softmax(onset_logits, temperature)

        coda_logits = head_logits_1d(self.W_coda_chunks, self.b_coda)
        coda_logits += self.W_disc_coda @ disc_state + self.W_word_coda @ word_state
        coda_logits += self.W_role_coda @ role_probs
        coda_logits += self.W_wb_coda @ wb_probs
        coda_probs  = self._softmax(coda_logits, temperature)

        return {
            "hidden": h2, "_x": x,
            "onset_probs": onset_probs, "nucleus_probs": nuc_probs,
            "coda_probs": coda_probs, "stress_probs": stress_probs, "wb_probs": wb_probs,
            "role_probs": role_probs,
            "discourse_state": disc_state, "word_state": word_state,
        }

    # -----------------------------------------------------------------------
    # Batched forward + backward (training)
    # -----------------------------------------------------------------------

    def _forward_batch(self, X: np.ndarray, T: np.ndarray, R: np.ndarray) -> dict:
        """
        X: (B, input_dim) pre-encoded context vectors.
        T: (B, 5) int target indices — used for teacher-forced conditioning.
        Returns activations needed for backward.
        """
        # Layer 1: (B, input_dim) → (B, hidden_dim)
        B = X.shape[0]
        disc_state = X[:, -self.state_dim:-self.word_state_dim]
        word_state = X[:, -self.word_state_dim:]
        H1_parts = []
        for i, row_chunks in enumerate(self.W_h_chunks):
            s = i * self.chunk_size
            e = s + row_chunks[0].shape[0]
            pre = np.tile(self.b_h[s:e], (B, 1))
            for j, chunk in enumerate(row_chunks):
                js = self.input_col_starts[j]
                je = self.input_col_starts[j + 1]
                pre += X[:, js:je] @ chunk.T
            pre += disc_state @ self.W_disc_h1_chunks[i].T + word_state @ self.W_word_h1_chunks[i].T
            H1_parts.append(self._relu(pre))
        H1 = np.concatenate(H1_parts, axis=1)

        # Layer 2: (B, hidden_dim) → (B, hidden_dim)
        H2_parts = []
        for i, row_chunks in enumerate(self.W_h2_chunks):
            s = i * self.chunk_size
            e = s + row_chunks[0].shape[0]
            pre = np.tile(self.b_h2[s:e], (B, 1))
            for j, chunk in enumerate(row_chunks):
                js = self.h_col_starts[j]
                je = self.h_col_starts[j + 1]
                pre += H1[:, js:je] @ chunk.T
            pre += disc_state @ self.W_disc_h2_chunks[i].T + word_state @ self.W_word_h2_chunks[i].T
            H2_parts.append(self._relu(pre))
        H2 = np.concatenate(H2_parts, axis=1)

        def head_logits_batch(chunks, bias):
            out = np.tile(bias, (B, 1))
            for i, chunk in enumerate(chunks):
                s = i * self.chunk_size
                e = s + chunk.shape[1]
                out += H2[:, s:e] @ chunk.T
            return out

        role_logits = head_logits_batch(self.W_role_chunks, self.b_role)
        role_logits += disc_state @ self.W_disc_role.T + word_state @ self.W_word_role.T
        role_probs = self._softmax_batch(role_logits)
        role_onehot = np.zeros_like(role_probs)
        role_onehot[np.arange(B), R] = 1.0

        wb_logits = head_logits_batch(self.W_wb_chunks, self.b_wb)
        wb_logits += disc_state @ self.W_disc_wb.T + word_state @ self.W_word_wb.T
        wb_logits += role_onehot @ self.W_role_wb.T
        wb_probs  = self._softmax_batch(wb_logits)
        wb_onehot = np.zeros_like(wb_probs)
        wb_onehot[np.arange(B), T[:, 4]] = 1.0

        stress_logits = head_logits_batch(self.W_stress_chunks, self.b_stress)
        stress_logits += disc_state @ self.W_disc_stress.T + word_state @ self.W_word_stress.T
        stress_logits += role_onehot @ self.W_role_stress.T
        stress_probs  = self._softmax_batch(stress_logits)

        # Nucleus: conditioned on true (teacher-forced) onset embedding
        onset_cond   = self.E_onset[T[:, 0]]      # (B, embed_dim)
        nuc_logits   = head_logits_batch(self.W_nucleus_chunks, self.b_nucleus)
        nuc_logits  += disc_state @ self.W_disc_nucleus.T + word_state @ self.W_word_nucleus.T
        nuc_logits  += role_onehot @ self.W_role_nucleus.T
        nuc_logits  += wb_onehot @ self.W_wb_nucleus.T
        nuc_logits  += onset_cond @ self.W_nuc_cond   # (B, n_nuclei)
        nuc_probs    = self._softmax_batch(nuc_logits)

        onset_logits  = head_logits_batch(self.W_onset_chunks, self.b_onset)
        onset_logits += disc_state @ self.W_disc_onset.T + word_state @ self.W_word_onset.T
        onset_logits += role_onehot @ self.W_role_onset.T
        onset_logits += wb_onehot @ self.W_wb_onset.T
        onset_logits += nuc_probs @ self.W_nuc_gate.T + self.b_nuc_gate
        onset_logits += stress_probs @ self.W_stress_gate.T + self.b_stress_gate
        onset_probs   = self._softmax_batch(onset_logits)

        # Coda: conditioned on true onset + true nucleus embeddings
        nucleus_cond = self.E_nucleus[T[:, 1]]     # (B, embed_dim)
        coda_cond    = np.concatenate([onset_cond, nucleus_cond], axis=1)  # (B, 2*embed_dim)
        coda_logits  = head_logits_batch(self.W_coda_chunks, self.b_coda)
        coda_logits += disc_state @ self.W_disc_coda.T + word_state @ self.W_word_coda.T
        coda_logits += role_onehot @ self.W_role_coda.T
        coda_logits += wb_onehot @ self.W_wb_coda.T
        coda_logits += coda_cond @ self.W_coda_cond   # (B, n_codas)
        coda_probs   = self._softmax_batch(coda_logits)

        return {
            "H1": H1, "H2": H2, "X": X,
            "onset_probs": onset_probs, "nucleus_probs": nuc_probs,
            "coda_probs": coda_probs, "stress_probs": stress_probs, "wb_probs": wb_probs,
            "role_probs": role_probs, "role_onehot": role_onehot,
            "onset_cond": onset_cond, "coda_cond": coda_cond,
            "discourse_state": disc_state, "word_state": word_state, "wb_onehot": wb_onehot,
        }

    def _backward_batch(
        self,
        fwd: dict,
        targets: np.ndarray,          # (B, 5) int: [oi, ni, ci, si, wb]
        roles: np.ndarray,            # (B,) int target role indices
        ctx_indices: np.ndarray,      # (B, context_len, 5) int
        valid_mask: np.ndarray,       # (B, context_len) bool
        active_word_mask: np.ndarray, # (B, context_len) bool
        lr: float,
        weight_decay: float,
    ) -> float:
        """
        Batched backward pass. Returns mean loss over the batch.
        """
        H1, H2, X = fwd["H1"], fwd["H2"], fwd["X"]
        disc_state = fwd["discourse_state"]
        word_state = fwd["word_state"]
        wb_onehot = fwd["wb_onehot"]
        role_onehot = fwd["role_onehot"]
        B = H2.shape[0]
        decay = weight_decay * lr

        # ---- Softmax-CE gradients: (probs - onehot) / B ----
        def delta(probs: np.ndarray, idx: np.ndarray, scale: float = 1.0) -> np.ndarray:
            d = probs.copy()
            d[np.arange(B), idx] -= 1.0
            return d * (scale / B)

        oi, ni, ci, si, wb = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3], targets[:, 4]

        # Loss (for logging)
        loss = (
            -np.log(fwd["onset_probs"][np.arange(B), oi] + 1e-10).mean()
            + -np.log(fwd["nucleus_probs"][np.arange(B), ni] + 1e-10).mean()
            + -np.log(fwd["coda_probs"][np.arange(B), ci] + 1e-10).mean()
            + 1.0 * -np.log(fwd["wb_probs"][np.arange(B), wb] + 1e-10).mean()
            + 0.8 * -np.log(fwd["stress_probs"][np.arange(B), si] + 1e-10).mean()
            + self.role_loss_weight * -np.log(fwd["role_probs"][np.arange(B), roles] + 1e-10).mean()
        )

        d_onset  = delta(fwd["onset_probs"],   oi,  1.0)   # (B, n_onsets)
        d_nuc    = delta(fwd["nucleus_probs"], ni,  1.0)   # (B, n_nuclei)
        d_coda   = delta(fwd["coda_probs"],    ci,  1.0)   # (B, n_codas)
        d_stress = delta(fwd["stress_probs"],  si,  0.8)   # (B, n_stress)
        d_wb     = delta(fwd["wb_probs"],      wb,  1.0)   # (B, n_wb)
        d_role   = delta(fwd["role_probs"],    roles, self.role_loss_weight)

        # Gate feedback
        d_nuc    += d_onset @ self.W_nuc_gate        # (B, n_nuclei)
        d_stress += d_onset @ self.W_stress_gate     # (B, n_stress)

        # Error at H2 from output heads — accumulated per column chunk
        err_H2 = np.zeros_like(H2)
        for i in range(self.n_h_col_chunks):
            s = i * self.chunk_size
            e = s + self.W_onset_chunks[i].shape[1]
            err_H2[:, s:e] += (d_onset @ self.W_onset_chunks[i]
                               + d_nuc   @ self.W_nucleus_chunks[i]
                               + d_coda  @ self.W_coda_chunks[i]
                               + d_stress@ self.W_stress_chunks[i]
                               + d_wb    @ self.W_wb_chunks[i]
                               + d_role  @ self.W_role_chunks[i])
        delta_H2 = err_H2 * (H2 > 0)

        # ---- Output head weight updates (column chunks of H2) ----
        for i in range(self.n_h_col_chunks):
            s = i * self.chunk_size
            e = s + self.W_onset_chunks[i].shape[1]
            H2slice = H2[:, s:e]
            self.W_onset_chunks[i]   -= lr * (d_onset.T  @ H2slice) + decay * self.W_onset_chunks[i]
            self.W_nucleus_chunks[i] -= lr * (d_nuc.T    @ H2slice) + decay * self.W_nucleus_chunks[i]
            self.W_coda_chunks[i]    -= lr * (d_coda.T   @ H2slice) + decay * self.W_coda_chunks[i]
            self.W_stress_chunks[i]  -= lr * (d_stress.T @ H2slice) + decay * self.W_stress_chunks[i]
            self.W_wb_chunks[i]      -= lr * (d_wb.T     @ H2slice) + decay * self.W_wb_chunks[i]
            self.W_role_chunks[i]    -= lr * (d_role.T   @ H2slice) + decay * self.W_role_chunks[i]

        self.b_onset   -= lr * d_onset.sum(0)
        self.b_nucleus -= lr * d_nuc.sum(0)
        self.b_coda    -= lr * d_coda.sum(0)
        self.b_stress  -= lr * d_stress.sum(0)
        self.b_wb      -= lr * d_wb.sum(0)
        self.b_role    -= lr * d_role.sum(0)

        self.W_disc_onset   -= lr * (d_onset.T @ disc_state) + decay * self.W_disc_onset
        self.W_disc_nucleus -= lr * (d_nuc.T   @ disc_state) + decay * self.W_disc_nucleus
        self.W_disc_coda    -= lr * (d_coda.T  @ disc_state) + decay * self.W_disc_coda
        self.W_disc_stress  -= lr * (d_stress.T@ disc_state) + decay * self.W_disc_stress
        self.W_disc_wb      -= lr * (d_wb.T    @ disc_state) + decay * self.W_disc_wb
        self.W_disc_role    -= lr * (d_role.T  @ disc_state) + decay * self.W_disc_role
        self.W_word_onset   -= lr * (d_onset.T @ word_state) + decay * self.W_word_onset
        self.W_word_nucleus -= lr * (d_nuc.T   @ word_state) + decay * self.W_word_nucleus
        self.W_word_coda    -= lr * (d_coda.T  @ word_state) + decay * self.W_word_coda
        self.W_word_stress  -= lr * (d_stress.T@ word_state) + decay * self.W_word_stress
        self.W_word_wb      -= lr * (d_wb.T    @ word_state) + decay * self.W_word_wb
        self.W_word_role    -= lr * (d_role.T  @ word_state) + decay * self.W_word_role
        self.W_role_onset   -= lr * (d_onset.T @ role_onehot) + decay * self.W_role_onset
        self.W_role_nucleus -= lr * (d_nuc.T   @ role_onehot) + decay * self.W_role_nucleus
        self.W_role_coda    -= lr * (d_coda.T  @ role_onehot) + decay * self.W_role_coda
        self.W_role_stress  -= lr * (d_stress.T@ role_onehot) + decay * self.W_role_stress
        self.W_role_wb      -= lr * (d_wb.T    @ role_onehot) + decay * self.W_role_wb
        self.W_wb_onset     -= lr * (d_onset.T @ wb_onehot) + decay * self.W_wb_onset
        self.W_wb_nucleus   -= lr * (d_nuc.T   @ wb_onehot) + decay * self.W_wb_nucleus
        self.W_wb_coda      -= lr * (d_coda.T  @ wb_onehot) + decay * self.W_wb_coda

        gate_lr = lr * 0.3
        self.W_nuc_gate    -= gate_lr * (d_onset.T @ fwd["nucleus_probs"]) + decay * self.W_nuc_gate
        self.b_nuc_gate    -= gate_lr * d_onset.sum(0)
        self.W_stress_gate -= gate_lr * (d_onset.T @ fwd["stress_probs"])  + decay * self.W_stress_gate
        self.b_stress_gate -= gate_lr * d_onset.sum(0)

        # Within-syllable conditioning weight updates (slow lr like gates; no embedding backprop
        # through this path — context path already updates E_onset/E_nucleus adequately)
        cond_lr = lr * 0.3
        onset_cond = fwd["onset_cond"]   # (B, embed_dim)
        coda_cond  = fwd["coda_cond"]    # (B, 2*embed_dim)
        self.W_nuc_cond  -= cond_lr * (onset_cond.T @ d_nuc)  + decay * self.W_nuc_cond
        self.W_coda_cond -= cond_lr * (coda_cond.T  @ d_coda) + decay * self.W_coda_cond

        # Layer 2 backward: update W_h2, propagate error to H1
        err_H1 = np.zeros_like(H1)
        grad_disc_hidden = np.zeros_like(disc_state)
        grad_word_hidden = np.zeros_like(word_state)
        for i, row_chunks in enumerate(self.W_h2_chunks):
            s = i * self.chunk_size
            e = s + row_chunks[0].shape[0]
            dH2_slice = delta_H2[:, s:e]
            self.b_h2[s:e] -= lr * dH2_slice.sum(0)
            grad_disc_hidden += dH2_slice @ self.W_disc_h2_chunks[i]
            grad_word_hidden += dH2_slice @ self.W_word_h2_chunks[i]
            self.W_disc_h2_chunks[i] -= lr * (dH2_slice.T @ disc_state) + decay * self.W_disc_h2_chunks[i]
            self.W_word_h2_chunks[i] -= lr * (dH2_slice.T @ word_state) + decay * self.W_word_h2_chunks[i]
            for j in range(len(row_chunks)):
                js = self.h_col_starts[j]
                je = self.h_col_starts[j + 1]
                chunk = row_chunks[j]
                H1j   = H1[:, js:je]
                err_H1[:, js:je] += dH2_slice @ chunk         # pre-update gradient
                row_chunks[j] -= lr * (dH2_slice.T @ H1j) + decay * chunk
        delta_H1 = err_H1 * (H1 > 0)

        # Layer 1 backward: update W_h, propagate error to embeddings
        grad_X = np.zeros((B, self.input_dim))
        for i, row_chunks in enumerate(self.W_h_chunks):
            s = i * self.chunk_size
            e = s + row_chunks[0].shape[0]
            dH1_slice = delta_H1[:, s:e]
            self.b_h[s:e] -= lr * dH1_slice.sum(0)
            grad_disc_hidden += dH1_slice @ self.W_disc_h1_chunks[i]
            grad_word_hidden += dH1_slice @ self.W_word_h1_chunks[i]
            self.W_disc_h1_chunks[i] -= lr * (dH1_slice.T @ disc_state) + decay * self.W_disc_h1_chunks[i]
            self.W_word_h1_chunks[i] -= lr * (dH1_slice.T @ word_state) + decay * self.W_word_h1_chunks[i]
            for j in range(len(row_chunks)):
                js = self.input_col_starts[j]
                je = self.input_col_starts[j + 1]
                chunk = row_chunks[j]
                Xj    = X[:, js:je]
                grad_X[:, js:je] += dH1_slice @ chunk         # pre-update gradient
                row_chunks[j] -= lr * (dH1_slice.T @ Xj) + decay * chunk

        # Direct head conditioning on active-word summary contributes additional
        # input gradient on the summary slice.
        disc_slice = slice(self.input_dim - self.state_dim, self.input_dim - self.word_state_dim)
        word_slice = slice(self.input_dim - self.word_state_dim, self.input_dim)
        grad_disc = grad_X[:, disc_slice] + (
            d_onset @ self.W_disc_onset
            + d_nuc @ self.W_disc_nucleus
            + d_coda @ self.W_disc_coda
            + d_stress @ self.W_disc_stress
            + d_wb @ self.W_disc_wb
            + d_role @ self.W_disc_role
        ) + grad_disc_hidden
        grad_word = grad_X[:, word_slice] + (
            d_onset @ self.W_word_onset
            + d_nuc @ self.W_word_nucleus
            + d_coda @ self.W_word_coda
            + d_stress @ self.W_word_stress
            + d_wb @ self.W_word_wb
            + d_role @ self.W_word_role
        ) + grad_word_hidden

        for pos in range(self.context_len):
            s = pos * self.token_dim
            g_o = grad_X[:, s                : s + self.embed_dim]
            g_n = grad_X[:, s + self.embed_dim     : s + 2*self.embed_dim]
            g_c = grad_X[:, s + 2*self.embed_dim   : s + 3*self.embed_dim]
            g_s = grad_X[:, s + 3*self.embed_dim   : s + 4*self.embed_dim]
            g_w = grad_X[:, s + 4*self.embed_dim   : s + self.token_dim]

            pos_idx = ctx_indices[:, pos, :]   # (B, 5)
            valid = valid_mask[:, pos]
            if valid.any():
                np.add.at(self.E_onset,   pos_idx[valid, 0], -lr * g_o[valid])
                np.add.at(self.E_nucleus, pos_idx[valid, 1], -lr * g_n[valid])
                np.add.at(self.E_coda,    pos_idx[valid, 2], -lr * g_c[valid])
                np.add.at(self.E_stress,  pos_idx[valid, 3], -lr * g_s[valid])
                np.add.at(self.E_wb,      pos_idx[valid, 4], -lr * g_w[valid])

        # Backprop through the tiny state encoders.
        token_vecs = np.zeros((B, self.context_len, self.token_dim), dtype=np.float64)
        for pos in range(self.context_len):
            s = pos * self.token_dim
            token_vecs[:, pos] = X[:, s:s + self.token_dim]

        disc_states = np.zeros((self.context_len + 1, B, self.discourse_state_dim), dtype=np.float64)
        word_states = np.zeros((self.context_len + 1, B, self.word_state_dim), dtype=np.float64)
        reset_masks = np.ones((self.context_len, B, 1), dtype=np.float64)
        prev_wb = np.ones((B, 1), dtype=np.float64)
        for pos in range(self.context_len):
            valid = valid_mask[:, pos].astype(np.float64)[:, None]
            reset_masks[pos] = prev_wb
            word_prev = word_states[pos] * (1.0 - reset_masks[pos])
            disc_next = self._tanh(token_vecs[:, pos] @ self.W_disc_in.T + disc_states[pos] @ self.W_disc_h.T + self.b_disc)
            word_next = self._tanh(token_vecs[:, pos] @ self.W_word_in.T + word_prev @ self.W_word_h.T + disc_next @ self.W_word_disc.T + self.b_word)
            disc_states[pos + 1] = disc_next * valid + disc_states[pos] * (1.0 - valid)
            word_states[pos + 1] = word_next * valid + word_states[pos] * (1.0 - valid)
            prev_wb = ctx_indices[:, pos, 4:5].astype(np.float64) * valid + np.ones((B, 1)) * (1.0 - valid)

        g_disc = grad_disc
        g_word = grad_word
        g_W_disc_in = np.zeros_like(self.W_disc_in)
        g_W_disc_h = np.zeros_like(self.W_disc_h)
        g_b_disc = np.zeros_like(self.b_disc)
        g_W_word_in = np.zeros_like(self.W_word_in)
        g_W_word_h = np.zeros_like(self.W_word_h)
        g_W_word_disc = np.zeros_like(self.W_word_disc)
        g_b_word = np.zeros_like(self.b_word)

        for pos in range(self.context_len - 1, -1, -1):
            valid = valid_mask[:, pos].astype(np.float64)[:, None]
            token_vec = token_vecs[:, pos]
            disc_prev = disc_states[pos]
            disc_cur = disc_states[pos + 1]
            word_prev_raw = word_states[pos]
            word_cur = word_states[pos + 1]
            reset = reset_masks[pos]
            word_prev = word_prev_raw * (1.0 - reset)

            d_disc_prev = g_disc * (1.0 - valid)
            d_word_prev = g_word * (1.0 - valid)

            dz_word = g_word * (1.0 - word_cur ** 2) * valid
            g_W_word_in += dz_word.T @ token_vec
            g_W_word_h += dz_word.T @ word_prev
            g_W_word_disc += dz_word.T @ disc_cur
            g_b_word += dz_word.sum(axis=0)

            token_grad = dz_word @ self.W_word_in
            d_word_prev += (dz_word @ self.W_word_h) * (1.0 - reset)
            g_disc_cur = g_disc * valid + dz_word @ self.W_word_disc

            dz_disc = g_disc_cur * (1.0 - disc_cur ** 2) * valid
            g_W_disc_in += dz_disc.T @ token_vec
            g_W_disc_h += dz_disc.T @ disc_prev
            g_b_disc += dz_disc.sum(axis=0)

            token_grad += dz_disc @ self.W_disc_in
            d_disc_prev += dz_disc @ self.W_disc_h

            s = pos * self.token_dim
            grad_X[:, s:s + self.token_dim] += token_grad
            g_disc = d_disc_prev
            g_word = d_word_prev

        self.W_disc_in   -= lr * g_W_disc_in   + decay * self.W_disc_in
        self.W_disc_h    -= lr * g_W_disc_h    + decay * self.W_disc_h
        self.b_disc      -= lr * g_b_disc
        self.W_word_in   -= lr * g_W_word_in   + decay * self.W_word_in
        self.W_word_h    -= lr * g_W_word_h    + decay * self.W_word_h
        self.W_word_disc -= lr * g_W_word_disc + decay * self.W_word_disc
        self.b_word      -= lr * g_b_word

        return float(loss)

    # -----------------------------------------------------------------------
    # Pre-encode all windows into numpy arrays (done once before training)
    # -----------------------------------------------------------------------

    def encode_dataset(self, windows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          X:   (N, input_dim)          — encoded context vectors
          T:   (N, 5)                  — target indices [oi, ni, ci, si, wb]
          R:   (N,)                    — target role indices
          CI:  (N, context_len, 5)     — context token indices for embedding updates
          VM:  (N, context_len)        — valid (non-padding) positions
          AW:  (N, context_len)        — positions belonging to the active word
        """
        N = len(windows)
        vocab = self.vocab
        X  = np.zeros((N, self.input_dim),                    dtype=np.float32)
        T  = np.zeros((N, 5),                                 dtype=np.int32)
        R  = np.zeros((N,),                                   dtype=np.int32)
        CI = np.zeros((N, self.context_len, 5),               dtype=np.int32)
        VM = np.zeros((N, self.context_len),                  dtype=bool)
        AW = np.zeros((N, self.context_len),                  dtype=bool)

        print("  Pre-encoding dataset...")
        for i, w in enumerate(windows):
            context = w["context"]
            # Pad context
            ctx = context[-self.context_len:]
            pad = self.context_len - len(ctx)
            last_word_start = 0

            # Context indices
            for p, tok in enumerate(ctx):
                CI[i, pad + p] = vocab.token_to_indices(tok)
                VM[i, pad + p] = True
                if tok.get("word_boundary", False):
                    last_word_start = p

            # Context vector
            vecs = [np.zeros(self.token_dim)] * pad
            for tok in ctx:
                oi, ni, ci, si, wb = vocab.token_to_indices(tok)
                vecs.append(np.concatenate([
                    self.E_onset[oi], self.E_nucleus[ni],
                    self.E_coda[ci],  self.E_stress[si], self.E_wb[wb],
                ]))
            if ctx:
                AW[i, pad + last_word_start : pad + len(ctx)] = True
            disc, word = self._state_summaries_from_tokens(ctx)
            X[i] = np.concatenate(vecs + [disc, word])

            T[i] = vocab.token_to_indices(w["target"])
            R[i] = vocab.role_to_index(w.get("target_role", "other"))

            if (i + 1) % 50000 == 0:
                print(f"    {i+1}/{N}")

        print(f"  Done. X shape: {X.shape}")
        return X, T, R, CI, VM, AW

    def precompute_indices(self, windows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Pre-compute T and CI index arrays — never stale, no embedding dependency.
        Returns:
          T:  (N, 5)                 int — target indices
          R:  (N,)                   int — target role indices
          CI: (N, context_len, 5)    int — context token indices (0 = padding)
          VM: (N, context_len)       bool — valid (non-padding) positions
          AW: (N, context_len)       bool — active-word positions in the context
        """
        N     = len(windows)
        vocab = self.vocab
        T  = np.zeros((N, 5),                   dtype=np.int32)
        R  = np.zeros((N,),                     dtype=np.int32)
        CI = np.zeros((N, self.context_len, 5), dtype=np.int32)
        VM = np.zeros((N, self.context_len),    dtype=bool)
        AW = np.zeros((N, self.context_len),    dtype=bool)

        for i, w in enumerate(windows):
            ctx = w["context"][-self.context_len:]
            pad = self.context_len - len(ctx)
            last_word_start = 0
            for p, tok in enumerate(ctx):
                CI[i, pad + p] = vocab.token_to_indices(tok)
                VM[i, pad + p] = True
                if tok.get("word_boundary", False):
                    last_word_start = p
            if ctx:
                AW[i, pad + last_word_start : pad + len(ctx)] = True
            T[i] = vocab.token_to_indices(w["target"])
            R[i] = vocab.role_to_index(w.get("target_role", "other"))

        print(f"  Indices pre-computed. T: {T.shape}, R: {R.shape}, CI: {CI.shape}, VM: {VM.shape}, AW: {AW.shape}")
        return T, R, CI, VM, AW

    def _encode_from_indices(self, CI: np.ndarray, VM: np.ndarray, AW: np.ndarray) -> np.ndarray:
        """
        Reconstruct X batch from CI: (B, context_len, 5) int.
        Uses current embedding matrices — always fresh, no staleness.
        """
        B = CI.shape[0]
        X = np.zeros((B, self.input_dim), dtype=np.float64)
        ed = self.embed_dim
        for pos in range(self.context_len):
            s   = pos * self.token_dim
            idx = CI[:, pos, :]
            valid = VM[:, pos].astype(np.float64)[:, None]
            X[:, s          : s +   ed] = self.E_onset[idx[:, 0]] * valid
            X[:, s +   ed   : s + 2*ed] = self.E_nucleus[idx[:, 1]] * valid
            X[:, s + 2*ed   : s + 3*ed] = self.E_coda[idx[:, 2]] * valid
            X[:, s + 3*ed   : s + 4*ed] = self.E_stress[idx[:, 3]] * valid
            X[:, s + 4*ed   : s + self.token_dim] = self.E_wb[idx[:, 4]] * valid
        disc = np.zeros((B, self.discourse_state_dim), dtype=np.float64)
        word = np.zeros((B, self.word_state_dim), dtype=np.float64)
        prev_wb = np.ones((B, 1), dtype=np.float64)
        for pos in range(self.context_len):
            s = pos * self.token_dim
            token_vec = X[:, s:s + self.token_dim]
            valid = VM[:, pos].astype(np.float64)[:, None]
            word_prev = word * (1.0 - prev_wb)
            disc_next = self._tanh(token_vec @ self.W_disc_in.T + disc @ self.W_disc_h.T + self.b_disc)
            word_next = self._tanh(token_vec @ self.W_word_in.T + word_prev @ self.W_word_h.T + disc_next @ self.W_word_disc.T + self.b_word)
            disc = disc_next * valid + disc * (1.0 - valid)
            word = word_next * valid + word * (1.0 - valid)
            prev_wb = CI[:, pos, 4:5].astype(np.float64) * valid + np.ones((B, 1)) * (1.0 - valid)
        X[:, -self.state_dim:-self.word_state_dim] = disc
        X[:, -self.word_state_dim:] = word
        return X

    # -----------------------------------------------------------------------
    # Inference / generation
    # -----------------------------------------------------------------------

    def build_phonotactic_masks(
        self,
        windows: list[dict],
        min_freq: int = 3,
        freq_weight: float = 0.5,
    ) -> None:
        """
        Build frequency-weighted phonotactic masks from training windows.

        min_freq:    triples appearing fewer than this many times are excluded
                     entirely (hard cutoff). Raise to 5-10 for tighter output.
        freq_weight: soft log-frequency bonus added to coda logits so common
                     words are naturally preferred over rare-but-valid ones.
                     Set to 0.0 to disable.
        """
        v = self.vocab
        counts = np.zeros((v.n_onsets, v.n_nuclei, v.n_codas), dtype=np.int32)

        for w in windows:
            for tok in w["context"] + [w["target"]]:
                oi, ni, ci, *_ = v.token_to_indices(tok)
                counts[oi, ni, ci] += 1

        on_nuc_cod = counts >= min_freq
        on_nuc     = on_nuc_cod.any(axis=2)

        self._on_nuc_mask     = on_nuc
        self._on_nuc_cod_mask = on_nuc_cod
        # Soft bonus: log(count+1) * weight, added to coda logits at generation time
        self._coda_freq_bonus = np.log1p(counts).astype(np.float64) * freq_weight

        n_pairs   = int(on_nuc.sum())
        n_triples = int(on_nuc_cod.sum())
        total     = v.n_onsets * v.n_nuclei * v.n_codas
        print(f"Phonotactic masks: {n_pairs} onset-nucleus pairs, "
              f"{n_triples}/{total} triples (min_freq={min_freq}, freq_weight={freq_weight})")

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

        def _apply_rep_penalty(logits: np.ndarray, component: str) -> np.ndarray:
            if rep_penalty <= 0 or not history:
                return logits
            logits = logits.copy()
            idx_map = {"onset": self.vocab.onset_to_idx,
                       "nucleus": self.vocab.nucleus_to_idx,
                       "coda": self.vocab.coda_to_idx}[component]
            for tok in history[-rep_window:]:
                idx = idx_map.get(tok[component], 0)
                logits[idx] -= rep_penalty
            return logits

        for _ in range(n_syllables):
            x = self._encode_context(context)
            disc_state = x[-self.state_dim:-self.word_state_dim]
            word_state = x[-self.word_state_dim:]
            h2 = self._h2_from_context(context)

            role_logits = self._head_logits(self.W_role_chunks, self.b_role, h2)
            role_logits += self.W_disc_role @ disc_state + self.W_word_role @ word_state
            role_probs = self._softmax(role_logits, temperature)

            # Stress and word boundary are independent of phoneme conditioning
            wb_logits    = self._head_logits(self.W_wb_chunks, self.b_wb, h2)
            wb_logits   += self.W_disc_wb @ disc_state + self.W_word_wb @ word_state
            wb_logits   += self.W_role_wb @ role_probs
            wb_probs     = self._softmax(wb_logits, temperature)
            wb_idx       = self.sample_token(wb_probs,     top_k)
            wb_feat      = np.zeros(2, dtype=np.float64)
            wb_feat[wb_idx] = 1.0

            stress_logits  = self._head_logits(self.W_stress_chunks, self.b_stress, h2)
            stress_logits += self.W_disc_stress @ disc_state + self.W_word_stress @ word_state
            stress_logits += self.W_role_stress @ role_probs
            stress_probs   = self._softmax(stress_logits, temperature)
            stress_idx     = self.sample_token(stress_probs, top_k)

            # Onset — gated by stress and a preliminary nucleus estimate
            nuc_raw_logits = self._head_logits(self.W_nucleus_chunks, self.b_nucleus, h2)
            nuc_raw_logits += self.W_disc_nucleus @ disc_state + self.W_word_nucleus @ word_state
            nuc_raw_logits += self.W_role_nucleus @ role_probs
            nuc_raw_logits += self.W_wb_nucleus @ wb_feat
            nuc_raw      = self._softmax(nuc_raw_logits, temperature)
            onset_logits  = self._head_logits(self.W_onset_chunks, self.b_onset, h2)
            onset_logits += self.W_disc_onset @ disc_state + self.W_word_onset @ word_state
            onset_logits += self.W_role_onset @ role_probs
            onset_logits += self.W_wb_onset @ wb_feat
            onset_logits += self.W_nuc_gate @ nuc_raw + self.b_nuc_gate
            onset_logits += self.W_stress_gate @ stress_probs + self.b_stress_gate
            onset_logits  = _apply_rep_penalty(onset_logits, "onset")
            onset_probs   = self._softmax(onset_logits, temperature)
            onset_idx     = self.sample_token(onset_probs, top_k)
            onset_emb     = self.E_onset[onset_idx]

            # Nucleus conditioned on sampled onset embedding
            nuc_logits  = self._head_logits(self.W_nucleus_chunks, self.b_nucleus, h2)
            nuc_logits += self.W_disc_nucleus @ disc_state + self.W_word_nucleus @ word_state
            nuc_logits += self.W_role_nucleus @ role_probs
            nuc_logits += self.W_wb_nucleus @ wb_feat
            nuc_logits += onset_emb @ self.W_nuc_cond
            if self._on_nuc_mask is not None:
                valid = self._on_nuc_mask[onset_idx]
                if valid.any():
                    nuc_logits[~valid] = -1e9
            nuc_logits  = _apply_rep_penalty(nuc_logits, "nucleus")
            nuc_probs   = self._softmax(nuc_logits, temperature)
            nuc_idx     = self.sample_token(nuc_probs, top_k)
            nuc_emb     = self.E_nucleus[nuc_idx]

            # Coda conditioned on sampled onset + nucleus embeddings
            coda_logits  = self._head_logits(self.W_coda_chunks, self.b_coda, h2)
            coda_logits += self.W_disc_coda @ disc_state + self.W_word_coda @ word_state
            coda_logits += self.W_role_coda @ role_probs
            coda_logits += self.W_wb_coda @ wb_feat
            coda_logits += np.concatenate([onset_emb, nuc_emb]) @ self.W_coda_cond
            if self._coda_freq_bonus is not None:
                coda_logits += self._coda_freq_bonus[onset_idx, nuc_idx]
            if self._on_nuc_cod_mask is not None:
                valid = self._on_nuc_cod_mask[onset_idx, nuc_idx]
                if valid.any():
                    coda_logits[~valid] = -1e9
            coda_logits  = _apply_rep_penalty(coda_logits, "coda")
            coda_probs   = self._softmax(coda_logits, temperature)
            coda_idx     = self.sample_token(coda_probs, top_k)

            token = {
                "onset":         self.vocab.onsets[onset_idx],
                "nucleus":       self.vocab.nuclei[nuc_idx],
                "coda":          self.vocab.codas[coda_idx],
                "stress":        stress_idx,
                "word_boundary": bool(wb_idx),
            }
            generated.append(token)
            history.append(token)
            context = (context + [token])[-self.context_len:]
        return generated

    @staticmethod
    def syllable_to_str(token: dict) -> str:
        mark = {0: "", 1: "ˈ", 2: "ˌ"}.get(token["stress"], "")
        return f"{mark}{token['onset']}{token['nucleus']}{token['coda']}"

    def tokens_to_text(self, tokens: list[dict]) -> str:
        parts = []
        for t in tokens:
            syl = self.syllable_to_str(t)
            parts.append((" " + syl) if (t.get("word_boundary") and parts) else syl)
        return "".join(parts)

    # -----------------------------------------------------------------------
    # Save / load
    # -----------------------------------------------------------------------

    def _save_chunks(self, arrays: dict, name: str, chunks: list[np.ndarray]) -> None:
        for i, c in enumerate(chunks):
            arrays[f"{name}_{i}"] = c

    def _load_chunks(self, d, name: str) -> list[np.ndarray]:
        chunks = []
        i = 0
        while f"{name}_{i}" in d:
            chunks.append(d[f"{name}_{i}"])
            i += 1
        return chunks

    def _save_wh_chunks(self, arrays: dict) -> None:
        for i, row_chunks in enumerate(self.W_h_chunks):
            for j, chunk in enumerate(row_chunks):
                arrays[f"W_h_r{i}_c{j}"] = chunk

    def _load_wh_chunks(self, d) -> list[list[np.ndarray]]:
        chunks = []
        i = 0
        while f"W_h_r{i}_c0" in d:
            row = []
            j = 0
            while f"W_h_r{i}_c{j}" in d:
                row.append(d[f"W_h_r{i}_c{j}"])
                j += 1
            chunks.append(row)
            i += 1
        return chunks

    def _save_2d_chunks(self, arrays: dict, name: str, chunks: list[list[np.ndarray]]) -> None:
        for i, row_chunks in enumerate(chunks):
            for j, chunk in enumerate(row_chunks):
                arrays[f"{name}_r{i}_c{j}"] = chunk

    def _load_2d_chunks(self, d, name: str) -> list[list[np.ndarray]]:
        chunks = []
        i = 0
        while f"{name}_r{i}_c0" in d:
            row = []
            j = 0
            while f"{name}_r{i}_c{j}" in d:
                row.append(d[f"{name}_r{i}_c{j}"])
                j += 1
            chunks.append(row)
            i += 1
        return chunks

    def save(self, path: str) -> None:
        arrays = dict(
            E_onset=self.E_onset, E_nucleus=self.E_nucleus,
            E_coda=self.E_coda,   E_stress=self.E_stress, E_wb=self.E_wb,
            b_h=self.b_h,
            b_onset=self.b_onset,   b_nucleus=self.b_nucleus,
            b_coda=self.b_coda,     b_stress=self.b_stress,  b_wb=self.b_wb, b_role=self.b_role,
            W_disc_in=self.W_disc_in, W_disc_h=self.W_disc_h, b_disc=self.b_disc,
            W_word_in=self.W_word_in, W_word_h=self.W_word_h, W_word_disc=self.W_word_disc, b_word=self.b_word,
            W_disc_onset=self.W_disc_onset, W_disc_nucleus=self.W_disc_nucleus,
            W_disc_coda=self.W_disc_coda, W_disc_stress=self.W_disc_stress, W_disc_wb=self.W_disc_wb, W_disc_role=self.W_disc_role,
            W_word_onset=self.W_word_onset, W_word_nucleus=self.W_word_nucleus,
            W_word_coda=self.W_word_coda,   W_word_stress=self.W_word_stress, W_word_wb=self.W_word_wb, W_word_role=self.W_word_role,
            W_role_onset=self.W_role_onset, W_role_nucleus=self.W_role_nucleus, W_role_coda=self.W_role_coda,
            W_role_stress=self.W_role_stress, W_role_wb=self.W_role_wb,
            W_wb_onset=self.W_wb_onset, W_wb_nucleus=self.W_wb_nucleus, W_wb_coda=self.W_wb_coda,
            W_nuc_gate=self.W_nuc_gate,       b_nuc_gate=self.b_nuc_gate,
            W_stress_gate=self.W_stress_gate, b_stress_gate=self.b_stress_gate,
            W_nuc_cond=self.W_nuc_cond, W_coda_cond=self.W_coda_cond,
            b_h2=self.b_h2,
            _hparams=np.array([self.context_len, self.embed_dim, self.hidden_dim,
                               self.discourse_state_dim, self.word_state_dim]),
        )
        self._save_wh_chunks(arrays)
        self._save_2d_chunks(arrays, "W_h2", self.W_h2_chunks)
        self._save_chunks(arrays, "W_disc_h1", self.W_disc_h1_chunks)
        self._save_chunks(arrays, "W_word_h1", self.W_word_h1_chunks)
        self._save_chunks(arrays, "W_disc_h2", self.W_disc_h2_chunks)
        self._save_chunks(arrays, "W_word_h2", self.W_word_h2_chunks)
        self._save_chunks(arrays, "W_onset",  self.W_onset_chunks)
        self._save_chunks(arrays, "W_nucleus",self.W_nucleus_chunks)
        self._save_chunks(arrays, "W_coda",   self.W_coda_chunks)
        self._save_chunks(arrays, "W_stress", self.W_stress_chunks)
        self._save_chunks(arrays, "W_wb",     self.W_wb_chunks)
        self._save_chunks(arrays, "W_role",   self.W_role_chunks)
        np.savez(path, **arrays)
        print(f"Saved weights to {path}.npz")

    def load(self, path: str) -> None:
        if not path.endswith(".npz"):
            path += ".npz"
        d = np.load(path)
        # Restore hyperparams saved at training time to avoid silent shape mismatches.
        if "_hparams" in d:
            if len(d["_hparams"]) < 5:
                raise ValueError(
                    "This checkpoint uses the older pre-state architecture and cannot "
                    "be loaded into the current TILM2. Regenerate training data and retrain."
                )
            ctx, emb, hid = int(d["_hparams"][0]), int(d["_hparams"][1]), int(d["_hparams"][2])
            disc_dim = int(d["_hparams"][3]) if len(d["_hparams"]) > 3 else 16
            word_dim = int(d["_hparams"][4]) if len(d["_hparams"]) > 4 else 8
            if (ctx, emb, hid) != (self.context_len, self.embed_dim, self.hidden_dim):
                print(f"  [load] hparam mismatch — overriding with saved values: "
                      f"context_len={ctx}, embed_dim={emb}, hidden_dim={hid}")
            self.context_len        = ctx
            self.embed_dim          = emb
            self.hidden_dim         = hid
            self.token_dim          = 4 * emb + 1
            self.discourse_state_dim = disc_dim
            self.word_state_dim      = word_dim
            self.state_dim           = disc_dim + word_dim
            self.input_dim           = ctx * self.token_dim + self.state_dim
            self.chunk_size         = 99
            self.n_chunks           = (hid + self.chunk_size - 1) // self.chunk_size
            self.n_input_col_chunks = (self.input_dim + self.chunk_size - 1) // self.chunk_size
            self.input_col_starts   = [j * self.chunk_size for j in range(self.n_input_col_chunks)] + [self.input_dim]
            self.h_col_starts       = [j * self.chunk_size for j in range(self.n_chunks)] + [self.hidden_dim]
        self.E_onset = d["E_onset"]; self.E_nucleus = d["E_nucleus"]
        self.E_coda  = d["E_coda"];  self.E_stress  = d["E_stress"]; self.E_wb = d["E_wb"]
        self.b_h     = d["b_h"]
        self.b_onset = d["b_onset"]; self.b_nucleus = d["b_nucleus"]
        self.b_coda  = d["b_coda"];  self.b_stress  = d["b_stress"]; self.b_wb = d["b_wb"]
        self.b_role  = d["b_role"] if "b_role" in d else np.zeros(self.vocab.n_roles)
        self.W_disc_in   = d["W_disc_in"] if "W_disc_in" in d else np.zeros((self.discourse_state_dim, self.token_dim))
        self.W_disc_h    = d["W_disc_h"] if "W_disc_h" in d else np.zeros((self.discourse_state_dim, self.discourse_state_dim))
        self.b_disc      = d["b_disc"] if "b_disc" in d else np.zeros(self.discourse_state_dim)
        self.W_word_in   = d["W_word_in"] if "W_word_in" in d else np.zeros((self.word_state_dim, self.token_dim))
        self.W_word_h    = d["W_word_h"] if "W_word_h" in d else np.zeros((self.word_state_dim, self.word_state_dim))
        self.W_word_disc = d["W_word_disc"] if "W_word_disc" in d else np.zeros((self.word_state_dim, self.discourse_state_dim))
        self.b_word      = d["b_word"] if "b_word" in d else np.zeros(self.word_state_dim)
        self.W_disc_onset   = d["W_disc_onset"]   if "W_disc_onset"   in d else np.zeros((self.vocab.n_onsets, self.discourse_state_dim))
        self.W_disc_nucleus = d["W_disc_nucleus"] if "W_disc_nucleus" in d else np.zeros((self.vocab.n_nuclei, self.discourse_state_dim))
        self.W_disc_coda    = d["W_disc_coda"]    if "W_disc_coda"    in d else np.zeros((self.vocab.n_codas, self.discourse_state_dim))
        self.W_disc_stress  = d["W_disc_stress"]  if "W_disc_stress"  in d else np.zeros((self.vocab.n_stress, self.discourse_state_dim))
        self.W_disc_wb      = d["W_disc_wb"]      if "W_disc_wb"      in d else np.zeros((2, self.discourse_state_dim))
        self.W_disc_role    = d["W_disc_role"]    if "W_disc_role"    in d else np.zeros((self.vocab.n_roles, self.discourse_state_dim))
        self.W_word_onset   = d["W_word_onset"]   if "W_word_onset"   in d else np.zeros((self.vocab.n_onsets, self.word_state_dim))
        self.W_word_nucleus = d["W_word_nucleus"] if "W_word_nucleus" in d else np.zeros((self.vocab.n_nuclei, self.word_state_dim))
        self.W_word_coda    = d["W_word_coda"]    if "W_word_coda"    in d else np.zeros((self.vocab.n_codas, self.word_state_dim))
        self.W_word_stress  = d["W_word_stress"]  if "W_word_stress"  in d else np.zeros((self.vocab.n_stress, self.word_state_dim))
        self.W_word_wb      = d["W_word_wb"]      if "W_word_wb"      in d else np.zeros((2, self.word_state_dim))
        self.W_word_role    = d["W_word_role"]    if "W_word_role"    in d else np.zeros((self.vocab.n_roles, self.word_state_dim))
        self.W_role_onset   = d["W_role_onset"]   if "W_role_onset"   in d else np.zeros((self.vocab.n_onsets, self.vocab.n_roles))
        self.W_role_nucleus = d["W_role_nucleus"] if "W_role_nucleus" in d else np.zeros((self.vocab.n_nuclei, self.vocab.n_roles))
        self.W_role_coda    = d["W_role_coda"]    if "W_role_coda"    in d else np.zeros((self.vocab.n_codas, self.vocab.n_roles))
        self.W_role_stress  = d["W_role_stress"]  if "W_role_stress"  in d else np.zeros((self.vocab.n_stress, self.vocab.n_roles))
        self.W_role_wb      = d["W_role_wb"]      if "W_role_wb"      in d else np.zeros((2, self.vocab.n_roles))
        self.W_wb_onset     = d["W_wb_onset"]     if "W_wb_onset"     in d else np.zeros((self.vocab.n_onsets, 2))
        self.W_wb_nucleus   = d["W_wb_nucleus"]   if "W_wb_nucleus"   in d else np.zeros((self.vocab.n_nuclei, 2))
        self.W_wb_coda      = d["W_wb_coda"]      if "W_wb_coda"      in d else np.zeros((self.vocab.n_codas, 2))
        self.W_nuc_gate    = d["W_nuc_gate"];    self.b_nuc_gate    = d["b_nuc_gate"]
        self.W_stress_gate = d["W_stress_gate"]; self.b_stress_gate = d["b_stress_gate"]
        # New conditioning matrices — fall back to zero-init if loading old weights
        ed = self.embed_dim
        self.W_nuc_cond  = d["W_nuc_cond"]  if "W_nuc_cond"  in d else np.zeros((ed,     self.vocab.n_nuclei))
        self.W_coda_cond = d["W_coda_cond"] if "W_coda_cond" in d else np.zeros((2 * ed, self.vocab.n_codas))
        self.W_h_chunks       = self._load_wh_chunks(d)
        self.W_h2_chunks      = self._load_2d_chunks(d, "W_h2")
        self.W_disc_h1_chunks = self._load_chunks(d, "W_disc_h1")
        self.W_word_h1_chunks = self._load_chunks(d, "W_word_h1")
        self.W_disc_h2_chunks = self._load_chunks(d, "W_disc_h2")
        self.W_word_h2_chunks = self._load_chunks(d, "W_word_h2")
        self.b_h2             = d["b_h2"]
        self.W_onset_chunks   = self._load_chunks(d, "W_onset")
        self.W_nucleus_chunks = self._load_chunks(d, "W_nucleus")
        self.W_coda_chunks    = self._load_chunks(d, "W_coda")
        self.W_stress_chunks  = self._load_chunks(d, "W_stress")
        self.W_wb_chunks      = self._load_chunks(d, "W_wb")
        self.W_role_chunks    = self._load_chunks(d, "W_role")
        if not self.W_disc_h1_chunks:
            self.W_disc_h1_chunks = [np.zeros((row[0].shape[0], self.discourse_state_dim)) for row in self.W_h_chunks]
        if not self.W_word_h1_chunks:
            self.W_word_h1_chunks = [np.zeros((row[0].shape[0], self.word_state_dim)) for row in self.W_h_chunks]
        if not self.W_disc_h2_chunks:
            self.W_disc_h2_chunks = [np.zeros((row[0].shape[0], self.discourse_state_dim)) for row in self.W_h2_chunks]
        if not self.W_word_h2_chunks:
            self.W_word_h2_chunks = [np.zeros((row[0].shape[0], self.word_state_dim)) for row in self.W_h2_chunks]
        if not self.W_role_chunks:
            self.W_role_chunks = [np.zeros((self.vocab.n_roles, chunk.shape[1])) for chunk in self.W_onset_chunks]
        self.n_chunks       = len(self.W_h_chunks)
        self.n_h_col_chunks = len(self.W_onset_chunks)
        print(f"Loaded weights from {path}")


# ---------------------------------------------------------------------------
# Training loop (mini-batch)
# ---------------------------------------------------------------------------

def train(
    model: TILM2,
    windows: list[dict],
    epochs: int = 5,
    lr: float = 0.01,
    lr_decay: float = 0.5,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    log_every: int = 500,
    save_path: str = "tilm2_weights",
    checkpoint_every: int = 1,
) -> None:
    N = len(windows)
    rng = np.random.default_rng(42)

    # Pre-compute integer index arrays once — independent of embeddings, never stale.
    # X is reconstructed fresh each batch from current embeddings via _encode_from_indices.
    print("Pre-computing dataset indices (one time)...")
    T, R, CI, VM, AW = model.precompute_indices(windows)

    for epoch in range(1, epochs + 1):
        epoch_lr = lr * (lr_decay ** (epoch - 1))
        t0 = time.time()

        # Shuffle
        idx     = rng.permutation(N)
        T_shuf  = T[idx]
        R_shuf  = R[idx]
        CI_shuf = CI[idx]
        VM_shuf = VM[idx]
        AW_shuf = AW[idx]

        total_loss = 0.0
        n_batches  = (N + batch_size - 1) // batch_size

        for b in range(n_batches):
            s   = b * batch_size
            e   = min(s + batch_size, N)
            Tb  = T_shuf[s:e]
            Rb  = R_shuf[s:e]
            CIb = CI_shuf[s:e]
            VMb = VM_shuf[s:e]
            AWb = AW_shuf[s:e]
            Xb  = model._encode_from_indices(CIb, VMb, AWb)   # always uses current embeddings

            fwd  = model._forward_batch(Xb, Tb, Rb)
            loss = model._backward_batch(
                fwd, Tb, Rb, CIb, VMb, AWb, lr=epoch_lr, weight_decay=weight_decay
            )
            total_loss += loss

            if (b + 1) % log_every == 0:
                avg = total_loss / (b + 1)
                elapsed = time.time() - t0
                print(f"  step {b+1}/{n_batches} | loss {avg:.4f} | {elapsed:.1f}s")

        avg_loss = total_loss / n_batches
        elapsed  = time.time() - t0
        print(f"Epoch {epoch}/{epochs} complete | avg loss: {avg_loss:.4f} | lr: {epoch_lr:.6f} | {elapsed:.1f}s")

        if epoch % checkpoint_every == 0:
            model.save(f"{save_path}_ep{epoch}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train TILM2")
    parser.add_argument("--data",        default="training_data.json")
    parser.add_argument("--weights",     default=None, help="Resume from .npz")
    parser.add_argument("--save",        default="tilm2_weights")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--lr",          type=float, default=0.01)
    parser.add_argument("--lr-decay",    type=float, default=0.5)
    parser.add_argument("--wd",          type=float, default=1e-4)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--hidden-dim",  type=int,   default=198)
    parser.add_argument("--embed-dim",   type=int,   default=21)
    parser.add_argument("--context-len", type=int,   default=10)
    parser.add_argument("--log-every",   type=int,   default=500, help="Log every N batches")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    print(f"Loading training data from {args.data}...")
    with open(args.data) as f:
        data = json.load(f)

    vocab_dict = dict(data["vocab"])
    if "role_vocab" in data:
        vocab_dict["role_vocab"] = data["role_vocab"]
    vocab   = Vocab(vocab_dict)
    windows = data["windows"]
    print(f"  {len(windows)} windows | vocab: {vocab.n_onsets} onsets, {vocab.n_nuclei} nuclei, {vocab.n_codas} codas")

    model = TILM2(vocab, context_len=args.context_len, embed_dim=args.embed_dim,
                  hidden_dim=args.hidden_dim, seed=args.seed)
    if args.weights:
        model.load(args.weights)

    train(
        model, windows,
        epochs=args.epochs,
        lr=args.lr,
        lr_decay=args.lr_decay,
        weight_decay=args.wd,
        batch_size=args.batch_size,
        log_every=args.log_every,
        save_path=args.save,
    )

    model.save(args.save)
    print("Training complete.")


if __name__ == "__main__":
    main()
