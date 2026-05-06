"""
Estimate whether a TILM2 shape fits the TI-84 matrix/list storage contract.

This does not inspect trained weights. It sizes the architecture from
training_data.json vocabulary counts and proposed model dimensions.
"""

import argparse
import json
import math


def prod(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training_data.json")
    parser.add_argument("--context-len", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=198)
    parser.add_argument("--disc-dim", type=int, default=16)
    parser.add_argument("--word-dim", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=99)
    parser.add_argument(
        "--hidden-input-mode",
        choices=["concat", "state"],
        default="concat",
        help="'concat' uses the current full context vector; 'state' feeds only discourse+word state to W_h.",
    )
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)

    vocab = data["vocab"]
    n_onsets = len(vocab["onsets"])
    n_nuclei = len(vocab["nuclei"])
    n_codas = len(vocab["codas"])
    n_roles = len(data.get("role_vocab", vocab.get("roles", ["other"])))
    n_stress = 3
    n_wb = 2

    token_dim = 4 * args.embed_dim + 1
    state_dim = args.disc_dim + args.word_dim
    full_input_dim = args.context_len * token_dim + state_dim
    hidden_input_dim = full_input_dim if args.hidden_input_mode == "concat" else state_dim
    n_hidden_chunks = math.ceil(args.hidden_dim / args.chunk_size)
    n_input_chunks = math.ceil(hidden_input_dim / args.chunk_size)

    w_h_mats = n_hidden_chunks * n_input_chunks
    w_h2_mats = n_hidden_chunks * n_hidden_chunks
    hidden_matrix_count = w_h_mats + w_h2_mats

    matrix_cells = args.hidden_dim * hidden_input_dim + args.hidden_dim * args.hidden_dim

    list_shapes = {
        "embeddings": [
            (n_onsets, args.embed_dim),
            (n_nuclei, args.embed_dim),
            (n_codas, args.embed_dim),
            (n_stress, args.embed_dim),
            (n_wb, 1),
        ],
        "recurrent_state": [
            (args.disc_dim, token_dim),
            (args.disc_dim, args.disc_dim),
            (args.word_dim, token_dim),
            (args.word_dim, args.word_dim),
            (args.word_dim, args.disc_dim),
        ],
        "state_to_hidden": [
            (args.hidden_dim, args.disc_dim),
            (args.hidden_dim, args.word_dim),
            (args.hidden_dim, args.disc_dim),
            (args.hidden_dim, args.word_dim),
        ],
        "hidden_to_heads": [
            (n_onsets, args.hidden_dim),
            (n_nuclei, args.hidden_dim),
            (n_codas, args.hidden_dim),
            (n_stress, args.hidden_dim),
            (n_wb, args.hidden_dim),
            (n_roles, args.hidden_dim),
        ],
        "direct_head_conditioning": [
            (n_onsets, args.disc_dim),
            (n_nuclei, args.disc_dim),
            (n_codas, args.disc_dim),
            (n_stress, args.disc_dim),
            (n_wb, args.disc_dim),
            (n_roles, args.disc_dim),
            (n_onsets, args.word_dim),
            (n_nuclei, args.word_dim),
            (n_codas, args.word_dim),
            (n_stress, args.word_dim),
            (n_wb, args.word_dim),
            (n_roles, args.word_dim),
        ],
        "small_conditioning": [
            (n_onsets, n_roles),
            (n_nuclei, n_roles),
            (n_codas, n_roles),
            (n_stress, n_roles),
            (n_wb, n_roles),
            (n_onsets, n_wb),
            (n_nuclei, n_wb),
            (n_codas, n_wb),
            (n_onsets, n_nuclei),
            (n_onsets, n_stress),
            (args.embed_dim, n_nuclei),
            (2 * args.embed_dim, n_codas),
        ],
        "biases": [
            (args.hidden_dim,),
            (args.hidden_dim,),
            (n_onsets,),
            (n_nuclei,),
            (n_codas,),
            (n_stress,),
            (n_wb,),
            (n_roles,),
            (args.disc_dim,),
            (args.word_dim,),
            (n_onsets,),
            (n_onsets,),
        ],
    }

    list_cells_by_group = {
        name: sum(prod(shape) for shape in shapes)
        for name, shapes in list_shapes.items()
    }
    list_cells = sum(list_cells_by_group.values())
    total_cells = matrix_cells + list_cells

    print("TILM2 TI layout estimate")
    print(f"  context_len: {args.context_len}")
    print(f"  embed_dim:   {args.embed_dim}")
    print(f"  hidden_dim:  {args.hidden_dim}")
    print(f"  token_dim:   {token_dim}")
    print(f"  full input_dim:   {full_input_dim}")
    print(f"  hidden input mode:{args.hidden_input_mode}")
    print(f"  hidden input_dim: {hidden_input_dim}")
    print()
    print("Matrix plan")
    print(f"  W_h matrices:       {w_h_mats}")
    print(f"  W_h2 matrices:      {w_h2_mats}")
    print(f"  hidden matrices:    {hidden_matrix_count}/10")
    print(f"  hidden matrix cells:{matrix_cells}")
    print(f"  fits 10 matrices:   {'yes' if hidden_matrix_count <= 10 else 'no'}")
    print()
    print("List plan")
    for name, cells in list_cells_by_group.items():
        print(f"  {name:<25} {cells:>7}")
    print(f"  {'total list cells':<25} {list_cells:>7}")
    print()
    print(f"Total parameter cells: {total_cells}")


if __name__ == "__main__":
    main()
