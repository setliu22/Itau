# Hit-Zone Aware Embedding Models for Fine-Grained Visual Homoglyph Detection

## Project structure

The pipeline runs in this order:

```
data/raw/  →  training/make_splits.py  →  data/splits/
                                                 ↓
                          ┌──────────────────────┴──────────────────────┐
                          │                                             │
              training/train.py                          training/train_glyphnet.py
           (or strip_design_sweep.py)                                   │
                          │                                  outputs/runs/glyphnet/
                  outputs/runs/                                         │
                          │                          ┌──────────────────┘
                          ↓                          ↓
                  evaluation/summarize.py    evaluation/baselines.py
```

---

### `rendering/`

| File | Purpose |
|------|---------|
| `renderer.py` | Renders a string to a fixed-height grayscale NumPy array using DejaVu Sans. Width scales with the string; height defaults to 32 px. Returns a `float32` array in `[0, 1]`. |
| `slicer.py` | Slices a rendered image into overlapping or non-overlapping column strips. Returns a `float32` array of shape `(num_slices, height, slice_width)` normalised to `[0, 1]`. Default: `slice_width=4`, `stride=4`. |

---

### `data/`

| File | Purpose |
|------|---------|
| `raw/domains_spoof.pkl` | Raw dataset of `(name_a, name_b, label)` pairs with train/validate/test keys. Source of truth — do not modify. |
| `raw/process_spoof.pkl` | Alternate processed version of the raw pairs. |
| `splits/train.pkl` | Training split produced by `make_splits.py`. |
| `splits/val.pkl` | Validation split. |
| `splits/test.pkl` | Test split. |

---

### `training/`

| File | Purpose |
|------|---------|
| `make_splits.py` | **Run first.** Reads `data/raw/domains_spoof.pkl` and writes the three split pkl files under `data/splits/`. Only needs to be run once. |
| `dataset.py` | `NamePairDataset` — renders and slices name pairs on-the-fly; `collate_fn` pads variable-length slice sequences into `(B, max_slices, height, slice_width)` tensors. |
| `train.py` | **Single training run.** Reads hyperparameters from `configs/default.yaml` (or `--config`). Saves `best.pt`, `config.yaml`, and `log.csv` to `outputs/runs/<run_name>/`. |
| `strip_design_sweep.py` | **Hyperparameter sweep.** Trains across all combinations of pooling, background, slice width, stride, and padding removal. Appends each result to `outputs/results.csv`. Supports `--resume` to continue an interrupted sweep. |
| `train_glyphnet.py` | **GlyphNet training.** Trains the GlyphNet CNN on individual rendered domain-name images (pairs are expanded into two per-name samples). Saves `best.pt` and `log.csv` to `outputs/runs/glyphnet/`. Required before running the `glyphnet` baseline in `evaluation/baselines.py`. |

---

### `models/`

| File | Purpose |
|------|---------|
| `encoder.py` | `VisualEncoder` — takes `(batch, num_slices, height, slice_width)`, flattens each slice, runs a two-layer Conv1D stem, then optionally applies a sequence encoder, and pools into a `(batch, embed_dim)` embedding. Controlled by two parameters: `encoder_type` selects the sequence encoder (`'conv1d'` — no additional encoder; `'bilstm'` — single-layer bidirectional LSTM, forward+backward final hidden states concatenated; `'transformer'` — 2-layer `TransformerEncoder` with `nhead=4`, `dim_feedforward=256`), and `pooling` selects how the sequence is reduced (`'mean'`, `'max'`, `'attention'`, or `'cls'`). Pooling is skipped for `'bilstm'` since the LSTM already reduces to a single vector. `'cls'` is only valid with `encoder_type='transformer'`; it prepends a learnable classification token to the slice sequence and uses its output embedding as the sequence representation. Also exposes `encode_slices()` which returns the full `(batch, num_slices, embed_dim)` sequence for use by downstream modules such as `HitZoneModule`. |
| `cross_attention.py` | `HitZoneModule` — cross-attention between a fraudulent and a real slice sequence. Fraudulent slices are the queries; real slices are the keys/values. Returns a global cosine similarity scalar `(batch,)` and a per-slice attention weight matrix `(batch, num_slices_a, num_slices_b)` whose rows sum to 1. High-attention real-name columns are the "hit zones". Uses a single `nn.MultiheadAttention` layer (`batch_first=True`). |
| `similarity.py` | `SimilarityHead` — computes cosine similarity between two embeddings, passes the scalar through a linear layer to produce a binary logit. Use with `BCEWithLogitsLoss`. |
| `glyphnet.py` | `GlyphNet` — binary spoof/non-spoof CNN classifier. Three Conv2D blocks each followed by a CBAM attention module (Gupta et al., 2023 style), global average pooling, and a two-layer linear head producing a single logit. Input: `(B, 1, H, W)` grayscale image. |

---

### `configs/`

| File | Purpose |
|------|---------|
| `default.yaml` | Base config for all training runs. Controls rendering height, background colour, slice width/stride, model embed dim, encoder type (`encoder_type: conv1d \| bilstm \| transformer`), pooling strategy, and training hyperparameters. Copy and modify to create experiment configs. |

---

### `evaluation/`

| File | Purpose |
|------|---------|
| `summarize.py` | **Run after the sweep.** Reads `outputs/results.csv` and prints the top-N configurations by val AUC plus the marginal effect of each design axis (pooling, background, slice width, padding). |
| `baselines.py` | **Baseline evaluation.** Evaluates six baseline methods on the test split and writes ROC-AUC, average precision, and best-threshold F1 to `outputs/results_baselines.csv`. See [Baselines](#baselines) below. |
| `evaluate_run.py` | **Per-run test-set evaluation.** Loads a trained checkpoint (`best.pt` or `latest.pt`) by run name, runs inference on the test split, and reports ROC-AUC, average precision, threshold-dependent metrics (accuracy, precision, recall, F1, specificity, MCC), and an ASCII confusion matrix — all at both the best-F1 threshold and a fixed 0.5 threshold. Optionally appends a result row to a CSV. |

---

## Baselines

`evaluation/baselines.py` measures six competing approaches. Each returns a similarity score in \[0, 1\] (higher = more likely a spoof pair). Metrics reported: ROC-AUC, average precision, and best-threshold F1.

| Method | Category | Description |
|--------|----------|-------------|
| `levenshtein` | String edit distance | Normalized Levenshtein similarity via rapidfuzz. |
| `damerau_levenshtein` | String edit distance | Normalized Damerau-Levenshtein (handles adjacent transpositions) via rapidfuzz. |
| `token_set_ratio` | String similarity | Token Set Ratio via rapidfuzz, scaled to \[0, 1\]. |
| `typo_pegging` | Visual-aware edit distance | Position-weighted edit distance with a hand-crafted visual confusion matrix. Earlier character positions are weighted more heavily; substitutions between visually confusable pairs (0/O, 1/l/I, rn/m, v/w, c/d) are penalized at 25% of the normal substitution cost. |
| `word_embedding` | Semantic embedding | Cosine similarity between mean fastText character n-gram vectors (`fasttext-wiki-news-subwords-300`, loaded via **gensim**). Included as a negative control — semantic embeddings encode meaning, not visual form, so they are expected to perform poorly on homoglyphs. |
| `glyphnet` | CNN classifier | Average spoof probability from a trained `GlyphNet` CNN (3× Conv2D + CBAM, Gupta et al. 2023 style). Each name in the pair is rendered independently; the pair score is the mean sigmoid output. **Requires training first** via `training/train_glyphnet.py`. Checkpoint must exist at `outputs/runs/glyphnet/best.pt`. |

### Running baselines

```bash
# Train GlyphNet first (if not already done):
python training/train_glyphnet.py

# Run all six baselines on the test split:
python evaluation/baselines.py

# Optional overrides:
python evaluation/baselines.py --test data/splits/test.csv --out outputs/results_baselines.csv
```

> **Note:** The `word_embedding` baseline requires `gensim`, which is not in `requirements.txt`. Install it separately: `pip install gensim`. The ~1 GB fastText model is downloaded automatically on the first run via `gensim.downloader`.

---

### `notebooks/`

| File | Purpose |
|------|---------|
| `explore.ipynb` | Visual sanity-checks for the rendering and slicing pipeline. Shows rendered images, per-slice 2-D heatmaps, a difference map between a genuine and spoofed pair, and the effect of varying `slice_width`/`stride` configs. |

---

### `outputs/`

| Path | Purpose |
|------|---------|
| `runs/<run_name>/best.pt` | Best model checkpoint (highest val AUC) for that run. |
| `runs/<run_name>/log.csv` | Epoch-level metrics: `train_loss`, `val_loss`, `val_auc`. |
| `runs/<run_name>/config.yaml` | Exact config used for that run. |
| `results.csv` | Aggregated sweep results — one row per sweep combination. |

---

## Running on an SSH / cluster machine

### 1. What to upload

Upload the project directory excluding `.venv/` and any `__pycache__/` folders:

```
fine-grained-homoglyph-detection/
├── configs/
├── data/
│   └── raw/            # must include domains_spoof.pkl (source of truth)
├── evaluation/
│   ├── baselines.py
│   └── summarize.py
├── models/
│   ├── cross_attention.py
│   ├── encoder.py
│   ├── glyphnet.py     # required by train_glyphnet.py and baselines.py
│   └── similarity.py
├── notebooks/          # optional — only needed for exploration
├── rendering/
├── training/
│   ├── dataset.py
│   ├── make_splits.py
│   ├── strip_design_sweep.py
│   ├── train.py
│   └── train_glyphnet.py
├── outputs/            # can start empty; created automatically
└── requirements.txt
```

Copy to the cluster using `scp` (do not upload `.venv` — it will be built on the cluster):

```bash
scp -r fine-grained-homoglyph-detection/ <user>@<host>:~/
```

Or use a GUI SFTP client such as **WinSCP** (Windows) — drag and drop the folder, excluding `.venv/`.

---

### 2. Set up the environment (on the cluster)

```bash
cd ~/fine-grained-homoglyph-detection

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Extra dependency for the word_embedding baseline (downloads ~1 GB fastText model on first run):
pip install gensim
```

> If the cluster provides a PyTorch module (e.g. via `module load pytorch`), load it before creating the venv to avoid re-downloading large binaries.

---

### 3. Run order

**Step 1 — Make splits** (only needed once; skip if `data/splits/` already contains the pkl files)

```bash
python training/make_splits.py
```

**Step 2a — Single training run** (to verify everything works)

```bash
python training/train.py
# or with a custom config:
python training/train.py --config configs/my_experiment.yaml
```

Before running the sweep, open `configs/default.yaml` and set `num_workers` to a value greater than 0 (e.g. 4) — the default of 0 is safe on Windows but slow on Linux.

**Step 2a (alt) — Train with BiLSTM or Transformer encoder**

Create a config file that overrides `model.encoder_type`. For example, `configs/bilstm.yaml`:

```yaml
# configs/bilstm.yaml
run_name: bilstm_mean

model:
  encoder_type: bilstm
  embed_dim: 128
  pooling: mean       # pooling is ignored for bilstm but kept for consistency
```

Or `configs/transformer.yaml`:

```yaml
# configs/transformer.yaml
run_name: transformer_attention

model:
  encoder_type: transformer
  embed_dim: 128
  pooling: attention   # 'mean' | 'max' | 'attention' | 'cls'
```

Both files only need to include keys you want to override — unset keys fall back to `configs/default.yaml`.  Then train:

```bash
python training/train.py --config configs/bilstm.yaml
python training/train.py --config configs/transformer.yaml
```

> **Note:** `embed_dim` must be even when using `encoder_type: bilstm` (the bidirectional LSTM splits it as `hidden_size = embed_dim // 2` per direction).  The transformer encoder requires `embed_dim` to be divisible by `nhead` (default 4).

**Step 2b — Full hyperparameter sweep** (the main experiment)

```bash
python training/strip_design_sweep.py

# Resume an interrupted sweep without re-running finished combos:
python training/strip_design_sweep.py --resume

# Quick smoke test (5 epochs, 3 combos):
python training/strip_design_sweep.py --sweep-epochs 5 --max-runs 3
```

The sweep trains 72 combinations (3 pooling × 2 padding × 2 background × 3 slice widths × 2 strides). Each result is appended to `outputs/results.csv` as it completes, so the sweep is safe to interrupt and resume.

**Step 2c — Evaluate a trained run on the test set**

```bash
# Evaluate the best checkpoint of a run:
python evaluation/evaluate_run.py --run transformer_attn

# Evaluate the last-epoch checkpoint instead:
python evaluation/evaluate_run.py --run transformer_attn --checkpoint latest

# Append results to a CSV for comparison across runs:
python evaluation/evaluate_run.py --run transformer_attn --out outputs/test_results.csv
python evaluation/evaluate_run.py --run bilstm_mean      --out outputs/test_results.csv
```

Reports ROC-AUC, average precision, threshold-dependent metrics (accuracy, precision, recall, F1, specificity, MCC), and an ASCII confusion matrix at both the best-F1 threshold and a fixed 0.5 threshold. The run config is read automatically from `outputs/runs/<run_name>/config.yaml`.

---

**Step 3 — Summarise sweep results**

```bash
python evaluation/summarize.py
# or point at a specific results file:
python evaluation/summarize.py --results outputs/results.csv --top 10
```

Prints the top-N configs by val AUC and the marginal effect of each design axis.

**Step 4 — Train GlyphNet** (required before running baselines)

```bash
python training/train_glyphnet.py
# Checkpoint saved to outputs/runs/glyphnet/best.pt
```

**Step 5 — Evaluate baselines**

```bash
python evaluation/baselines.py
# Results saved to outputs/results_baselines.csv
```

---

### 4. Retrieve results

```bash
# From your local machine:
scp -r <user>@<host>:~/fine-grained-homoglyph-detection/outputs/ ./outputs/
```

Or use WinSCP to drag the `outputs/` folder back to your machine.

---

## Encoder Architecture Reference

This section documents every encoder type and pooling strategy implemented in `models/encoder.py` at the level of detail needed to describe them in a methods section.  All dimension values refer to the defaults in `configs/default.yaml` (`embed_dim=128`, `height=32`, `slice_width=6`, so `slice_dim=192`).

---

### Shared input preprocessing

Every encoder type goes through the same two steps before the encoder-specific stage.

**1. Slicing and flattening**

A domain name is rendered to a grayscale image of fixed height `H` and variable width, then cut into `N` column strips of width `W` with stride `S` (see `rendering/`).  Each strip is flattened to a 1-D vector of dimension `slice_dim = H × W`.  The sequence of `N` flattened strips forms the raw input to the encoder.  Because different names produce different numbers of strips, sequences within a batch are zero-padded to the longest length `N_max`.

Input tensor shape: `(B, N_max, H, W)`  →  reshaped to `(B, N_max, slice_dim)`

**2. Conv1D stem**

The stem projects each strip into the model's embedding space and captures local visual context across adjacent strips.  It treats the strip sequence as a 1-D signal and applies two Conv1D layers in sequence:

```
Conv1d(slice_dim → embed_dim, kernel=3, padding=1) → ReLU
Conv1d(embed_dim → embed_dim, kernel=3, padding=1) → ReLU
```

Both layers use same-padding (`padding=1`, `kernel_size=3`) so the sequence length is preserved.  The kernel spans three consecutive strips, giving each output position access to its immediate left and right neighbours before any encoder or pooling stage.

Output shape: `(B, embed_dim, N_max)` — one `embed_dim`-dimensional vector per strip position.

With defaults, the stem has `(192×128×3 + 128) + (128×128×3 + 128) = 73,984 + 49,280 = 123,264` trainable parameters.

---

### Encoder types

#### `conv1d` — Conv1D stem only

The Conv1D stem is the entire sequence model.  No additional temporal encoder is applied; the per-strip representations from the stem are passed directly to the pooling stage.

This is the simplest configuration.  It captures local (±1 strip) context via the two convolutional layers but has no mechanism to model long-range dependencies across the sequence.  It is the baseline against which `bilstm` and `transformer` are compared in the sweep.

Additional parameters beyond the stem: none (or 129 for attention pooling scorer).

---

#### `bilstm` — Bidirectional LSTM

A single-layer bidirectional LSTM is applied to the Conv1D stem output.  The stem output is transposed to `(B, N_max, embed_dim)` to satisfy the LSTM's `batch_first=True` convention.

```
LSTM(
    input_size  = embed_dim,       # 128
    hidden_size = embed_dim // 2,  # 64
    num_layers  = 1,
    bidirectional = True,
    batch_first   = True,
)
```

The LSTM processes the strip sequence in both directions.  At the end of the sequence, the final hidden states of the forward pass `h_fwd ∈ R^(embed_dim/2)` and the backward pass `h_bwd ∈ R^(embed_dim/2)` are concatenated:

```
embedding = concat(h_fwd, h_bwd)  ∈ R^embed_dim
```

Because this reduces the entire sequence to a single vector via the recurrent state, no pooling step is needed or applied — the `pooling` config key is ignored for this encoder type.

`embed_dim` must be even.  With defaults (`embed_dim=128`, `hidden_size=64`), LSTM parameter count per direction: `4 × (input_size × hidden_size + hidden_size² + 2 × hidden_size) = 4 × (8192 + 4096 + 128) = 49,664`.  Total for both directions: `99,328`.

The BiLSTM captures global sequential context and is sensitive to the order of strips.  Unlike the transformer, it processes the sequence causally (forward direction) and anti-causally (backward direction) and its recurrent inductive bias may generalise better on shorter sequences.

---

#### `transformer` — Transformer encoder with Rotary Position Embeddings

Two stacked `_RoPEEncoderLayer` layers are applied to the Conv1D stem output (transposed to `(B, N_max, embed_dim)`).

**Architecture (per layer)**

Each layer follows the pre-norm convention (LayerNorm applied *before* each sub-layer rather than after), which is more stable for training from scratch than the post-norm default:

```
x ← x + Dropout(RoPE-MHA(LayerNorm(x), key_padding_mask))
x ← x + Dropout(FFN(LayerNorm(x)))
```

The feed-forward network is:

```
FFN(x) = Linear(dim_feedforward → embed_dim)(ReLU(Linear(embed_dim → dim_feedforward)(x)))
```

with `dim_feedforward=256`.

Parameters per layer (defaults): attention `4 × (128² + 128) = 66,048`, FFN `(128×256+256) + (256×128+128) = 65,792`, LayerNorms `2 × 256 = 512` → `~132,352` per layer, `~264,704` for 2 layers.

**Rotary Position Embeddings (RoPE)**

Standard sinusoidal or learned absolute position embeddings assign a fixed representation to each index position.  This is problematic for variable-length strip sequences: position 5 in a 6-strip name is near the end of a short word, while position 5 in a 30-strip name is near the beginning of a long one — the same absolute index carries very different semantic meaning.

RoPE (Su et al., 2021) instead encodes position by rotating the query and key vectors in the attention computation.  For a head of dimension `d_h`, define frequency terms:

```
θ_i = 1 / 10000^(2i / d_h),   i = 0, 1, ..., d_h/2 − 1
```

For a vector `x` at position `m`, the rotary embedding applies a block-diagonal rotation matrix `R_m` whose 2×2 blocks are:

```
[cos(m·θ_i)  -sin(m·θ_i)]
[sin(m·θ_i)   cos(m·θ_i)]
```

In practice this is computed without materialising the rotation matrix:

```
RoPE(x, m) = x ⊙ cos(m·θ) + rotate_half(x) ⊙ sin(m·θ)
```

where `rotate_half(x)` splits `x` into two halves `[x₁, x₂]` and returns `[-x₂, x₁]`, and `θ` is the vector of all `θ_i` repeated twice (to match `d_h`).

RoPE is applied to Q and K (but not V) before computing the attention scores.  Because the dot product `Q_m · K_n` then depends only on the relative offset `m − n` and not on absolute positions, the model learns position-relative attention patterns.  This is important here because what matters for homoglyph detection is the local visual neighbourhood around each strip, not where in the sequence the strip happens to fall.

RoPE has **no trainable parameters** — the cos/sin tables are pre-computed and cached as non-persistent buffers.

**Padding mask**

Since sequences within a batch are zero-padded, a boolean `key_padding_mask` of shape `(B, N_max)` is constructed from the true sequence lengths: entry `[b, i]` is `True` if position `i` is beyond the actual length of sample `b`.  This mask is passed into every attention layer so that real strip positions never attend to padding.  The same mask is also used during pooling to exclude padded positions from the aggregate (see pooling section below).

**Multi-head attention configuration**

```
nhead    = 4
head_dim = embed_dim // nhead = 32   (with embed_dim=128)
scale    = head_dim^(-0.5) ≈ 0.177
```

---

### Pooling strategies

After the encoder stage (or directly after the stem for `conv1d`), the per-strip sequence `(B, embed_dim, N)` is reduced to a single vector `(B, embed_dim)`.  For the transformer encoder, all pooling operations use the true sequence lengths to exclude padded positions.

#### `mean` — Global average pooling

Each dimension of the embedding is averaged across all valid strip positions:

```
e = (1 / |valid|) Σ_{i ∈ valid} x_i
```

No additional parameters.  Treats every strip as equally important.

#### `max` — Global max pooling

The maximum value is taken element-wise across all valid strip positions:

```
e_d = max_{i ∈ valid} x_{i,d}   for each dimension d
```

No additional parameters.  The output captures the most activated feature across the entire name regardless of where it occurs.  Padded positions are set to `-inf` before the max so they cannot win.

#### `attention` — Learned attention pooling

A single linear layer scores each strip, and the final embedding is the softmax-weighted sum:

```
s_i = w^T x_i + b           (scalar score per strip)
α   = softmax({s_i}_{valid}) (weights over valid positions only)
e   = Σ_i α_i x_i
```

where `w ∈ R^embed_dim` and `b ∈ R` are learnable.  This gives the model the ability to focus on the most discriminative strips (e.g., the character positions where the homoglyph substitution occurs) rather than weighting all positions equally.  Padded positions are masked to `-inf` before the softmax so they receive zero weight.

Additional parameters: `embed_dim + 1 = 129` (with defaults).

#### `cls` — Classification token (transformer only)

A learnable `[CLS]` token `c ∈ R^embed_dim` is prepended to the strip sequence *before* the transformer encoder, extending the input from `(B, N, embed_dim)` to `(B, N+1, embed_dim)` with the token at position 0.  The padding mask is extended accordingly so the CLS position is always treated as valid.  After the transformer, the output at position 0 is taken directly as the sequence embedding — no explicit pooling step is needed:

```
[c, x_1, ..., x_N]  →  Transformer  →  [e_cls, e_1, ..., e_N]
embedding = e_cls
```

The token is initialised with `trunc_normal_(std=0.02)`.  Because RoPE assigns position 0 to the CLS token and positions 1..N to the strips, the rotary embeddings remain meaningful for the strip positions (they see relative offsets 1..N rather than 0..N−1, which shifts absolute values but preserves relative structure).

This is the aggregation strategy used by BERT and ViT.  The CLS token can in principle learn to attend to any subset of strip positions via self-attention, giving it more flexibility than the attention-pooling scorer (which scores strips independently with a single linear layer).

Additional parameters: `embed_dim = 128` (the CLS token itself; with defaults).

Only valid with `encoder_type: transformer`.

---

### Valid encoder × pooling combinations

| `encoder_type` | `pooling: mean` | `pooling: max` | `pooling: attention` | `pooling: cls` |
|---|---|---|---|---|
| `conv1d` | ✓ | ✓ | ✓ | — |
| `bilstm` | (ignored — LSTM output used directly) | — | — | — |
| `transformer` | ✓ | ✓ | ✓ | ✓ |

For `bilstm`, setting `pooling` in the config has no effect on the model; the key `pooling: mean` is used by convention to keep the config valid.

---

### Default hyperparameters

| Parameter | Value | Config key |
|---|---|---|
| `embed_dim` | 128 | `model.embed_dim` |
| `slice_dim` | 192 (= 32 × 6) | derived from `rendering.height` × `slicing.slice_width` |
| Transformer layers | 2 | hard-coded in `encoder.py` |
| Transformer heads (`nhead`) | 4 | hard-coded in `encoder.py` |
| Transformer head dim | 32 (= 128 / 4) | derived |
| FFN width (`dim_feedforward`) | 256 | hard-coded in `encoder.py` |
| BiLSTM hidden size | 64 (= 128 / 2) | derived from `embed_dim` |
| BiLSTM layers | 1 | hard-coded in `encoder.py` |
| Conv1D stem kernel | 3, same padding | hard-coded in `encoder.py` |