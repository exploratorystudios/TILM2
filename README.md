# TILM2 — Syllable-Level Language Model for TI-84 Plus CE

A complete pipeline for training a small autoregressive language model on a PC and deploying it to a TI-84 Plus CE graphing calculator. The model generates poetry and prose using syllable-level tokens, with a 198-dimensional hidden layer factored into 6 parallel output heads (onset, nucleus, coda, stress, word-boundary, role). The architecture is built around the TI-84's matrix and list limits and ~150 KB RAM constraint.

---


NOTE: When running inference, be sure to use only words in vocab_words.txt. The model has only been trained on those 78 words.

## Requirements

```
pip install numpy tivars
```

Python 3.8 or later. The `tivars` library is only required for the final native file conversion step.

---

## Full Pipeline

### Step 1 — Generate a Corpus

```bash
python3 corpus_gen.py --words 400000 --output corpus/thematic.txt
```

Generates a synthetic English prose corpus using scene-plan generation with persistent world state. Output is written to `corpus/thematic.txt`. Adjust `--words` for the target corpus size.

---

### Step 2 — Process the Corpus into Syllable Tokens

```bash
python3 corpus_processor.py --corpus-dir corpus --cmu-dict data/cmudict.dict --output data/training_data.json
```

Reads all `.txt` files from `corpus/`, segments them into syllables using `data/cmudict.dict`, and assigns onset, nucleus, coda, stress, word-boundary, and role tags to each token. Output is written to `data/training_data.json`.

---

### Step 2b — Build the Project Lexicon

```bash
python3 build_project_lexicon.py --cmu-dict data/cmudict.dict --source corpus/thematic.txt --source data/seeds.txt --output data/project_lexicon.json
```

Builds the word→syllable map used by inference and the calculator seed encoder. Run this once after generating a corpus.

---

### Step 3 — Train the Model

```bash
python3 tilm2_model.py --data data/training_data.json --save data/tilm2_weights --epochs 20 --lr 0.01 --lr-decay 0.95
```

Trains the TILM2 model and saves checkpoints as `data/tilm2_weights.npz`. Key flags:

| Flag | Default | Description |
|---|---|---|
| `--data` | `training_data.json` | Path to tokenized training data |
| `--save` | `tilm2_weights` | Output prefix for `.npz` checkpoint |
| `--epochs` | 5 | Training epochs |
| `--lr` | 0.01 | Initial learning rate |
| `--lr-decay` | 0.5 | Per-epoch learning rate decay multiplier |

---

### Step 4 — Run Inference on PC

**Interactive:**
```bash
python3 inference.py --weights data/tilm2_weights.npz --data data/training_data.json --cmu-dict data/cmudict.dict --decode-lexicon data/project_lexicon.json
```
Enter a seed phrase at the prompt and the model will generate a syllable sequence interactively.

**Batch:**
```bash
python3 batch_inference.py --weights data/tilm2_weights.npz --data data/training_data.json --seeds data/seeds.txt --cmu-dict data/cmudict.dict --decode-lexicon data/project_lexicon.json --output results.txt --runs 3 --temperature 0.7 --top-k 3
```

Runs `--runs` completions per seed line in `data/seeds.txt` and writes all results to `results.txt`. Adjust `--temperature` (higher = more varied) and `--top-k` (lower = more focused) to tune output quality.

**Validate results:**
```bash
python3 analyze_results.py --input results.txt
python3 check_results.py --input results.txt
```

`analyze_results.py` reports attractor tokens, degenerate loops, and vocabulary coverage. `check_results.py` checks n-gram overlap, OOV rate, and verbatim run lengths.

---

### Step 5 — Export to TI-84 Plus CE

This step converts a trained `.npz` checkpoint into native TI variable files ready to transfer to the calculator.

**Recommended export (H1 precompute + H2 column-major + English runtime):**
```bash
python3 build_ti_runtime.py --weights data/tilm2_weights.npz --data data/training_data.json --out-prefix ti_model --cmu-dict data/cmudict.dict --english-words data/vocab_words.txt --h1-precompute --h2-colmajor --english-runtime
```


#### What the flags do

| Flag | Effect |
|---|---|
| `--h1-precompute` | Precomputes token-context H1 contribution lists so the calculator adds vectors instead of multiplying matrices. Required for reasonable generation speed. |
| `--h2-colmajor` | Repacks W_h2 tensors in column-major order so T2H2 reads each H1 activation once and skips zeros. Further speeds up the output head computation. |
| `--english-runtime` | Adds T2SEED, T2WORD, and T2DRAW programs for vocabulary-bounded English seed input and graph-screen display. |
| `--keep-h1-base-weights` | Keeps the original W_h base weight lists after precomputation (larger transfer, used for debugging). Omit this to prune them. |
| `--include-debug` | Adds debug programs to the TI-BASIC runtime. |
| `--include-compare-fixture` | Generates a step-1 compare fixture in `ti_model_native/compare_lists/` for parity checking. |

#### Output directories

After running `build_ti_runtime.py`:

| Directory | Contents |
|---|---|
| `ti_model_packed` | Base CSV weights and lists |
| `ti_model_packed_h1pre` | + H1 precompute pages |
| `ti_model_packed_h1pre_pruned` | H1 precompute with base W_h pruned (recommended) |
| `ti_model_basic` | Generated TI-BASIC source programs (`.txt`) |
| `ti_model_native` | Native TI transfer files (`.8xl`, `.8xp`, `.8xg`) |

---

### Step 6 — Transfer to Calculator

Transfer files are organized into categorized subfolders inside `ti_model_native/`:

| Folder | Contents | When to use |
|---|---|---|
| `programs/` | All TI-BASIC programs (`.8xp`) | Every transfer |
| `runtime_lists/` | Generation output lists (R, W, S, N, O, C, etc.) | Every transfer |
| `support_lists/` | Fixture, context, mask, and reference lists | Every transfer |
| `precompute_lists/` | Loose P* token-context H1 contribution pages | If transferring individually instead of via groups |
| `state_precompute/` | PD* (disc state) and PW* (word state) H1 combined weights | Every transfer when `--h1-precompute` is used |
| `compare_lists/` | Step-1 reference values for parity checking | Debug only |
| `loose_weight_lists/` | Loose L* base weight lists | Debug / individual transfer |

**Bulk transfer with TILP:**
```bash
tilp --calc=ti84+ --cable=DirectLink --port=1 --no-gui --silent ti_model_native/programs/*
tilp --calc=ti84+ --cable=DirectLink --port=1 --no-gui --silent ti_model_native/state_precompute/*
tilp --calc=ti84+ --cable=DirectLink --port=1 --no-gui --silent ti_model_native/support_lists/*
tilp --calc=ti84+ --cable=DirectLink --port=1 --no-gui --silent ti_model_native/runtime_lists/*
```

---

## Running the Model on the Calculator

**Total generation time: approximately 2.5–3 hours.** Plan accordingly and do not turn the calculator off mid-run.

All operations are driven through the `TILM2` main menu program. Run it with `[PRGM]` → `TILM2` → `[ENTER]`. You will never need to call the sub-programs (T2INIT, T2SEED, etc.) directly.

### Main menu

```
TILM2
  INIT    — load weights from flash into RAM
  INFO    — display model metadata
  SMOKE   — quick sanity check
  RUN  →  — generation sub-menu
  SEED    — enter a seed phrase
  DRAW    — redraw the output on the graph screen
  QUIT    — exit
```

### Step-by-step

**1. INIT**
Select `INIT` first. This unarchives all lists and matrices from flash storage into RAM. It must complete before anything else will work. Run it once per session or after a RAM clear.

**2. SEED**
Select `SEED` from the main menu. The program will prompt for the number of words in your seed phrase.

> **Syllable limit:** The context window is 10 syllables. Count syllables, not words — "the moon rose" is 3 syllables, "beautiful remember" is already 5. Do not exceed 10 syllables total or the seed will be truncated. A good starting point is 3–5 short common words.

Enter each word one at a time when prompted. Words must exist in `vocab_words.txt`; unknown words are rejected. After the last word the program returns to the main menu.

**3. RUN → GEN**
Select `RUN` from the main menu, then select `GEN` from the run sub-menu. Generation begins immediately. Output is rendered to the graph screen as syllables accumulate into words.

> **Garbage collection:** The TI-84 will periodically pause mid-run and display `Garbage Collect? (1=No, 2=Yes)`. Always press `2`. This is normal and will happen multiple times during a full run. Pressing `1` risks an out-of-memory crash.

A full generation run takes **2.5–3 hours**. Keep an eye on the calculator for garbage collection prompts — missing one and letting it sit on `2` will abort the run.

---

## Data Files

Static files (committed to the repo):

| File | Purpose |
|---|---|
| `data/cmudict.dict` | CMU Pronouncing Dictionary — required by the syllabifier |
| `data/seeds.txt` | Seed prompts for batch inference, one per line |
| `data/vocab_words.txt` | Known English words for seed validation on the calculator |
| `data/vocab.json` | Vocab metadata for the web visualizer (auto-generated, small) |

Generated files (produced by the pipeline, not committed):

| File | Produced by |
|---|---|
| `data/training_data.json` | `corpus_processor.py` |
| `data/project_lexicon.json` | `build_project_lexicon.py` |
| `data/tilm2_weights.npz` | `tilm2_model.py` |

---

## Script Reference

| Script | Stage | Purpose |
|---|---|---|
| `corpus_gen.py` | 1 | Generate synthetic prose corpus |
| `clean_corpus.py` | 1 | Strip headers, footnotes, metadata from existing text files |
| `corpus_processor.py` | 1 | Tokenize corpus into syllable sequences with tags |
| `syllabifier.py` | 1 | CMU dictionary syllabifier (imported by corpus_processor) |
| `build_project_lexicon.py` | 1 | Build `project_lexicon.json` word→syllable map |
| `tilm2_model.py` | 2 | Train model, save `.npz` checkpoint |
| `inference.py` | 3 | Interactive PC inference |
| `batch_inference.py` | 3 | Batch PC inference from seeds file |
| `analyze_results.py` | 3 | Pattern analysis on results.txt |
| `check_results.py` | 3 | Heuristic quality checks on results.txt |
| `build_ti_runtime.py` | 4 | Master export orchestrator |
| `export_ti84.py` | 4 | Export weights → CSV + manifest |
| `ti_packed_runtime.py` | 4 | PC simulator of packed TI layout (parity validation) |
| `ti_packed_inference.py` | 4 | Inference using packed CSVs (parity reference) |
| `add_ti_h1_precompute.py` | 4 | Add H1 token-context precompute pages |
| `prune_ti_h1_base_weights.py` | 4 | Remove redundant base W_h after precomputation |
| `repack_ti_column_major.py` | 4 | Repack W_h2 tensors in column-major order |
| `convert_ti_exports_tivars.py` | 4 | Convert CSVs → native `.8xl`/`.8xp`/`.8xg` files |
| `generate_ti_basic_runtime.py` | 5 | Generate TI-BASIC programs (T2H1, T2H2, T2OUT, T2GEN, etc.) |
| `generate_ti_english_runtime.py` | 5 | Generate English runtime programs (T2WORD, T2DRAW) |
| `debug_step1.py` | 6 | Print step-1 hidden/logit values for calculator comparison |
| `export_ti_step1_compare_fixture.py` | 6 | Generate step-1 reference data in calculator-readable format |
| `export_ti_forward_fixture.py` | 6 | Multi-step forward pass fixture for full parity testing |
| `ti_layout_analyze.py` | 6 | Verify matrix/list packing fits TI-84 memory constraints |
| `infer_ti_context_ids.py` | 6 | Map context window token indices for debugging |

---

## Further Reading

- `docs/TI_EXPORT_PIPELINE.md` — detailed build flags, output folders, H1/H2 optimization internals
- `docs/TI_TARGET.md` — TI-84 Plus CE hardware constraints and storage strategy
- `docs/DEBUG_PARITY_STATUS.md` — parity verification status and known issues
