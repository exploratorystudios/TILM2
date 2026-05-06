"""
Debug step-1 generation on PC reference model.
Prints each additive term in the onset logit computation so we can
cross-reference against the calculator's debug output.

Usage:
    python3 debug_step1.py \
        --weights "../TILM2 (Copy 2)/tilm2_weights.npz" \
        --data "../TILM2 (Copy 2)/training_data.json" \
        --seed-text "the moon rose by the" \
        --temperature 0.5
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/home/thewindmage/Desktop/TILM2/TILM2")
from syllabifier import load_cmu_dict, tokenize_text
from tilm2_model import TILM2, Vocab


def softmax(logits, temp):
    z = logits / temp
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def top5(probs, labels):
    idx = np.argsort(probs)[::-1][:5]
    return [(labels[i], float(probs[i])) for i in idx]


def run_debug(model: TILM2, context: list[dict], temperature: float):
    print("=== STEP 1 DEBUG ===")
    x = model._encode_context(context)
    disc_state = x[-model.state_dim:-model.word_state_dim]
    word_state = x[-model.word_state_dim:]

    print(f"disc_state[:4]: {disc_state[:4]}")
    print(f"word_state[:4]: {word_state[:4]}")

    h2 = model._h2_from_context(context)
    print(f"h2[:8]: {h2[:8]}")
    print(f"h2 nonzero count: {np.count_nonzero(h2)} / {len(h2)}")

    # --- Role ---
    role_logits = model._head_logits(model.W_role_chunks, model.b_role, h2)
    role_logits += model.W_disc_role @ disc_state + model.W_word_role @ word_state
    role_probs = softmax(role_logits, temperature)
    print(f"\nrole_probs: {[(model.vocab.roles[i], round(float(role_probs[i]),4)) for i in range(len(role_probs))]}")

    # --- WB ---
    wb_logits = model._head_logits(model.W_wb_chunks, model.b_wb, h2)
    wb_logits += model.W_disc_wb @ disc_state + model.W_word_wb @ word_state
    wb_logits += model.W_role_wb @ role_probs
    wb_probs = softmax(wb_logits, temperature)
    print(f"wb_probs: {[round(float(p),4) for p in wb_probs]}")
    wb_idx = int(np.argmax(wb_probs))
    wb_feat = np.zeros(2)
    wb_feat[wb_idx] = 1.0
    print(f"wb_idx={wb_idx} (one-hot wb_feat={wb_feat})")

    # --- Stress ---
    stress_logits = model._head_logits(model.W_stress_chunks, model.b_stress, h2)
    stress_logits += model.W_disc_stress @ disc_state + model.W_word_stress @ word_state
    stress_logits += model.W_role_stress @ role_probs
    stress_probs = softmax(stress_logits, temperature)
    print(f"stress_probs: {[round(float(p),4) for p in stress_probs]}")
    stress_idx = int(np.argmax(stress_probs))
    print(f"stress_idx={stress_idx}")

    # --- nuc_raw (preliminary nucleus for onset gate) ---
    nuc_raw_logits = model._head_logits(model.W_nucleus_chunks, model.b_nucleus, h2)
    nuc_raw_logits += model.W_disc_nucleus @ disc_state + model.W_word_nucleus @ word_state
    nuc_raw_logits += model.W_role_nucleus @ role_probs
    nuc_raw_logits += model.W_wb_nucleus @ wb_feat
    nuc_raw = softmax(nuc_raw_logits, temperature)
    print(f"nuc_raw top5: {top5(nuc_raw, model.vocab.nuclei)}")

    # --- Onset logit breakdown ---
    print("\n--- ONSET LOGIT BREAKDOWN ---")

    base = model._head_logits(model.W_onset_chunks, model.b_onset, h2)
    print(f"After b_onset + W_onset@h2, top5: {top5(softmax(base,temperature), model.vocab.onsets)}")
    print(f"  [3]D={base[3]:.4f}, [4]DH={base[4]:.4f}, [11]L={base[11]:.4f}, [15]R={base[15]:.4f}, [16]S={base[16]:.4f}")

    base += model.W_disc_onset @ disc_state + model.W_word_onset @ word_state
    print(f"After +disc/word state, top5: {top5(softmax(base,temperature), model.vocab.onsets)}")
    print(f"  [3]D={base[3]:.4f}, [4]DH={base[4]:.4f}, [11]L={base[11]:.4f}, [15]R={base[15]:.4f}, [16]S={base[16]:.4f}")

    base += model.W_role_onset @ role_probs
    print(f"After +role, top5: {top5(softmax(base,temperature), model.vocab.onsets)}")
    print(f"  [3]D={base[3]:.4f}, [4]DH={base[4]:.4f}, [11]L={base[11]:.4f}, [15]R={base[15]:.4f}, [16]S={base[16]:.4f}")

    base += model.W_wb_onset @ wb_feat
    print(f"After +wb (one-hot wb_idx={wb_idx}), top5: {top5(softmax(base,temperature), model.vocab.onsets)}")
    print(f"  [3]D={base[3]:.4f}, [4]DH={base[4]:.4f}, [11]L={base[11]:.4f}, [15]R={base[15]:.4f}, [16]S={base[16]:.4f}")

    nuc_gate_contrib = model.W_nuc_gate @ nuc_raw + model.b_nuc_gate
    base += nuc_gate_contrib
    print(f"After +nuc_gate, top5: {top5(softmax(base,temperature), model.vocab.onsets)}")
    print(f"  [3]D={base[3]:.4f}, [4]DH={base[4]:.4f}, [11]L={base[11]:.4f}, [15]R={base[15]:.4f}, [16]S={base[16]:.4f}")
    print(f"  nuc_gate_contrib[3]={nuc_gate_contrib[3]:.4f}, [4]={nuc_gate_contrib[4]:.4f}")

    stress_gate_contrib = model.W_stress_gate @ stress_probs + model.b_stress_gate
    base += stress_gate_contrib
    print(f"After +stress_gate, top5: {top5(softmax(base,temperature), model.vocab.onsets)}")
    print(f"  [3]D={base[3]:.4f}, [4]DH={base[4]:.4f}, [11]L={base[11]:.4f}, [15]R={base[15]:.4f}, [16]S={base[16]:.4f}")
    print(f"  stress_gate_contrib[3]={stress_gate_contrib[3]:.4f}, [4]={stress_gate_contrib[4]:.4f}")

    # Rep penalty (generate mode): last 15 tokens in history = seed context
    print(f"\nSeed context (last 15 for rep penalty):")
    for tok in context[-15:]:
        print(f"  onset='{tok['onset']}' (idx={model.vocab.onset_to_idx.get(tok['onset'],0)})")

    rep_logits = base.copy()
    for tok in context[-15:]:
        idx = model.vocab.onset_to_idx.get(tok["onset"], 0)
        rep_logits[idx] -= 0.5
    print(f"After rep_penalty (-0.5 per occurrence), top5: {top5(softmax(rep_logits,temperature), model.vocab.onsets)}")
    print(f"  [3]D={rep_logits[3]:.4f}, [4]DH={rep_logits[4]:.4f}, [11]L={rep_logits[11]:.4f}, [15]R={rep_logits[15]:.4f}, [16]S={rep_logits[16]:.4f}")

    final_probs = softmax(rep_logits, temperature)
    print(f"\nFINAL onset_probs top5: {top5(final_probs, model.vocab.onsets)}")
    print(f"  D prob={final_probs[3]:.6f}")

    # Also print context IDs for cross-check with |LI
    print("\n=== CONTEXT IDs (|LI equivalent) ===")
    print("Seed context tokens:")
    for i, tok in enumerate(context):
        on_id = model.vocab.onset_to_idx.get(tok["onset"], 0) + 1  # TI 1-indexed
        nu_id = model.vocab.nucleus_to_idx.get(tok["nucleus"], 0) + 1
        co_id = model.vocab.coda_to_idx.get(tok["coda"], 0) + 1
        st_id = tok.get("stress", 0) + 1
        wb_id = 2 if tok.get("word_boundary", False) else 1
        print(f"  slot{i+1}: onset_id={on_id}({tok['onset']}), nuc_id={nu_id}({tok['nucleus']}), coda_id={co_id}({tok['coda']}), stress={st_id}, wb={wb_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed-text", default="the moon rose by the")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--cmu-dict", default="cmudict.dict")
    parser.add_argument("--allow-missing-cmu", action="store_true")
    args = parser.parse_args()

    load_cmu_dict(args.cmu_dict, required=not args.allow_missing_cmu)
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    vocab_dict = dict(data["vocab"])
    if "role_vocab" in data:
        vocab_dict["role_vocab"] = data["role_vocab"]
    vocab = Vocab(vocab_dict)
    model = TILM2(vocab, context_len=10, embed_dim=21, hidden_dim=198)
    model.load(args.weights)
    if "windows" in data:
        model.build_phonotactic_masks(data["windows"], min_freq=1, freq_weight=0.0)

    context = tokenize_text(args.seed_text) if args.seed_text else []
    context = context[-10:]
    print(f"Tokenized seed: {[t['onset']+t['nucleus']+t['coda'] for t in context]}")
    run_debug(model, context, args.temperature)


if __name__ == "__main__":
    main()
