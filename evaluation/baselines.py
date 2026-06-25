"""String-similarity baselines for homoglyph detection.

Six methods are implemented, each returning a similarity score in [0, 1]
(higher = more similar = more likely to be a spoof):

  levenshtein_sim          — normalized Levenshtein via rapidfuzz
  damerau_levenshtein_sim  — normalized Damerau-Levenshtein via rapidfuzz
  token_set_ratio_sim      — Token Set Ratio via rapidfuzz, scaled to [0, 1]
  typoPegging_sim          — position-weighted edit distance with a visual
                             confusion matrix (Liu et al., 2016 style)
  WordEmbeddingSimilarity  — fastText subword-vector cosine similarity
  GlyphNetBaseline         — average spoof probability from a trained GlyphNet
                             CNN (Gupta et al., 2023) for the two rendered images

Usage:
    python evaluation/baselines.py
    python evaluation/baselines.py --test data/splits/test.csv
    python evaluation/baselines.py --out  baseline_results.csv
    python evaluation/baselines.py --resume          # skip already-completed methods
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import distance as rf_distance, fuzz as rf_fuzz
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Visual confusion matrix for typoPegging
# ---------------------------------------------------------------------------
# Each pair (a, b) means character a is often visually confused with character b.
# The set is symmetric — both orders are listed explicitly.
# Coverage: 0/O, 1/l/I, rn/m (approximated as r↔n, n↔m), vv/w (v↔w), cl/d (c↔l, c↔d).

_CONFUSION_PAIRS: frozenset[tuple[str, str]] = frozenset({
    # 0 / O / o
    ("0", "o"), ("o", "0"),
    ("0", "O"), ("O", "0"),
    ("o", "O"), ("O", "o"),
    # 1 / l / I
    ("1", "l"), ("l", "1"),
    ("1", "I"), ("I", "1"),
    ("l", "I"), ("I", "l"),
    # rn / m  — individual character approximations
    ("r", "n"), ("n", "r"),
    ("n", "m"), ("m", "n"),
    # vv / w
    ("v", "w"), ("w", "v"),
    ("v", "u"), ("u", "v"),
    # cl / d
    ("c", "l"), ("l", "c"),
    ("c", "d"), ("d", "c"),
})

# Cost (in [0, 1]) of substituting a visually confused character pair.
# Regular substitution costs 1.0; confused pairs cost this much less.
_CONFUSION_SUB_COST: float = 0.25


def _pos_weight(i: int, max_len: int) -> float:
    """Linear position decay: 1.0 at position 0, 0.5 at the last position."""
    if max_len <= 1:
        return 1.0
    return 1.0 - 0.5 * i / (max_len - 1)


# ---------------------------------------------------------------------------
# Individual similarity functions
# ---------------------------------------------------------------------------

def levenshtein_sim(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0, 1] (1 = identical)."""
    return rf_distance.Levenshtein.normalized_similarity(a, b)


def damerau_levenshtein_sim(a: str, b: str) -> float:
    """Normalized Damerau-Levenshtein similarity in [0, 1] (1 = identical)."""
    return rf_distance.DamerauLevenshtein.normalized_similarity(a, b)


def token_set_ratio_sim(a: str, b: str) -> float:
    """Token Set Ratio similarity in [0, 1] (1 = identical token sets)."""
    return rf_fuzz.token_set_ratio(a, b) / 100.0


def typoPegging_sim(a: str, b: str) -> float:
    """Position-weighted visual-confusion edit distance, normalized to [0, 1].

    Earlier character positions are weighted more heavily (position decay).
    Substitutions between visually confusable character pairs are penalized
    less than arbitrary substitutions.

    Normalization: divide weighted distance by the unweighted max string length
    so that fully-identical strings → 1.0 and the score is clipped to [0, 1].
    """
    a, b = a.lower(), b.lower()
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 1.0

    max_len = max(n, m)

    # Standard DP table; costs are position-weighted.
    dp: list[list[float]] = [[0.0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _pos_weight(i - 1, max_len)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _pos_weight(j - 1, max_len)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Use the position along the diagonal as the effective position.
            eff_pos = int(round((i + j) / 2.0)) - 1
            w = _pos_weight(eff_pos, max_len)

            if a[i - 1] == b[j - 1]:
                sub_cost = 0.0
            elif (a[i - 1], b[j - 1]) in _CONFUSION_PAIRS:
                sub_cost = w * _CONFUSION_SUB_COST
            else:
                sub_cost = w

            dp[i][j] = min(
                dp[i - 1][j] + w,           # deletion from a
                dp[i][j - 1] + w,           # insertion into a
                dp[i - 1][j - 1] + sub_cost,  # substitution / match
            )

    # Normalize by unweighted string length as specified.
    sim = 1.0 - dp[n][m] / max_len
    return float(np.clip(sim, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Word-embedding similarity (fastText subword vectors)
# ---------------------------------------------------------------------------
# NOTE ON FIT FOR THIS TASK:
# Word-level semantic embeddings (fastText, GloVe, word2vec) encode *meaning*:
# "apple" and "fruit" are nearby because they co-occur in similar contexts.
# Homoglyph spoofing operates on *visual form*, not meaning — "paypa1" and
# "paypal" share no semantic context but are visually nearly identical.
# Cosine similarity between their mean embedding vectors will therefore be
# low even for obvious spoofs, and the method cannot distinguish homoglyphs
# from genuinely unrelated company names with similar semantic fields.
# This baseline is included to demonstrate that gap empirically.
#
# To approximate character-level shape via fastText, we avoid word-level
# lookup entirely and instead decompose each name into character n-grams
# (fastText-style: boundary markers "<"/">" + n=3..6), look each n-gram up
# in the pretrained vocabulary, and average the found subword vectors.
# This partially captures surface form but is still limited by which n-grams
# appear in the model's vocabulary.

class WordEmbeddingSimilarity:
    """Cosine similarity between mean fastText subword-vector embeddings.

    The model is loaded lazily on the first call to ``sim``.  A single
    module-level instance (``word_embedding``) is reused for all calls.
    """

    MODEL_NAME = "fasttext-wiki-news-subwords-300"
    MIN_N = 3
    MAX_N = 6

    def __init__(self) -> None:
        self._model = None  # loaded lazily

    def _load(self):
        if self._model is None:
            import gensim.downloader as api
            print(f"  [WordEmbeddingSimilarity] loading {self.MODEL_NAME} ...")
            self._model = api.load(self.MODEL_NAME)
        return self._model

    @staticmethod
    def _ngrams(s: str, min_n: int, max_n: int) -> list[str]:
        """Generate fastText-style character n-grams with boundary markers."""
        # Wrap in boundary markers as fastText does during training.
        s = f"<{s}>"
        grams = []
        for n in range(min_n, max_n + 1):
            grams.extend(s[i: i + n] for i in range(len(s) - n + 1))
        return grams

    def _embed(self, s: str) -> np.ndarray:
        """Return the mean fastText subword vector for raw string *s*."""
        model = self._load()
        grams = self._ngrams(s.lower(), self.MIN_N, self.MAX_N)
        vecs = [model[g] for g in grams if g in model]
        if not vecs:
            return np.zeros(model.vector_size, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def sim(self, a: str, b: str) -> float:
        """Cosine similarity between the mean subword embeddings of *a* and *b*."""
        va = self._embed(a)
        vb = self._embed(b)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        cosine = float(np.dot(va, vb) / (norm_a * norm_b))
        # Clip to [0, 1]: cosine similarity can be negative, but for name
        # pairs derived from the same language the practical range is [0, 1].
        return float(np.clip(cosine, 0.0, 1.0))


# Module-level singleton — avoids reloading the 1 GB model across calls.
word_embedding = WordEmbeddingSimilarity()


# ---------------------------------------------------------------------------
# GlyphNet baseline
# ---------------------------------------------------------------------------

class GlyphNetBaseline:
    """Binary spoof-probability baseline using a trained GlyphNet CNN.

    Each name in a pair is rendered independently with render_name(), passed
    through the trained model, and converted to a spoof probability via
    sigmoid.  The pair score is the mean of the two probabilities.

    The model is loaded lazily on the first call to ``sim``.

    Args:
        checkpoint_path: Path to best.pt saved by training/train_glyphnet.py.
                         Defaults to outputs/runs/glyphnet/best.pt relative
                         to the repository root.
    """

    DEFAULT_CKPT = ROOT / "outputs" / "runs" / "glyphnet" / "best.pt"

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self._ckpt_path = Path(checkpoint_path) if checkpoint_path else self.DEFAULT_CKPT
        self._model = None
        self._device = None
        self._render_cfg: dict = {}

    def _load(self) -> None:
        if self._model is not None:
            return

        import sys
        sys.path.insert(0, str(ROOT))
        import torch
        from models.glyphnet import GlyphNet

        if not self._ckpt_path.exists():
            raise FileNotFoundError(
                f"GlyphNet checkpoint not found at {self._ckpt_path}. "
                "Run training/train_glyphnet.py first."
            )

        print(f"  [GlyphNetBaseline] loading checkpoint from {self._ckpt_path} ...")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self._ckpt_path, map_location=self._device)

        cfg = ckpt.get("config", {})
        self._render_cfg = {
            "height":     cfg.get("height",     32),
            "background": cfg.get("background", "black"),
        }

        self._model = GlyphNet(
            in_channels=1,
            base_channels=cfg.get("base_channels", 32),
            embed_dim=cfg.get("embed_dim",     128),
        ).to(self._device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()
        print(
            f"  [GlyphNetBaseline] loaded (epoch={ckpt.get('epoch')}, "
            f"val_auc={ckpt.get('val_auc', '?'):.4f})"
        )

    def _spoof_prob(self, name: str) -> float:
        """Return the spoof probability for a single rendered domain name."""
        self._load()  # must come first — adds ROOT to sys.path before imports below
        import torch
        from rendering.renderer import render_name as _render_name

        img = _render_name(name, **self._render_cfg)         # (H, W) float32
        # (1, 1, H, W) — batch=1, channel=1
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logit = self._model(x)                           # (1,)
            prob = torch.sigmoid(logit).item()
        return float(prob)

    def sim(self, a: str, b: str) -> float:
        """Average spoof probability for the two rendered images.

        Higher score → pair is more likely a spoof.

        Args:
            a: fraudulent_name
            b: real_name

        Returns:
            Mean of sigmoid(GlyphNet(render(a))) and sigmoid(GlyphNet(render(b))).
        """
        return (self._spoof_prob(a) + self._spoof_prob(b)) / 2.0


# Module-level singleton — model is loaded once on first call.
glyphnet_baseline = GlyphNetBaseline()


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

METHODS: dict[str, callable] = {
    "levenshtein":         levenshtein_sim,
    "damerau_levenshtein": damerau_levenshtein_sim,
    "token_set_ratio":     token_set_ratio_sim,
    "typo_pegging":        typoPegging_sim,
    "word_embedding":      word_embedding.sim,
    "glyphnet":            glyphnet_baseline.sim,
}


def _best_threshold_metrics(
    labels: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    """Return classification metrics at the threshold that maximises F1.

    Returns a dict with keys: ``threshold``, ``f1``, ``accuracy``,
    ``precision``, ``recall``, ``specificity``, ``mcc``.
    """
    thresholds = np.unique(scores)
    best_f1 = -1.0
    best_t = thresholds[0]
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t

    preds = (scores >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold":   float(best_t),
        "f1":          float(best_f1),
        "accuracy":    float(accuracy_score(labels, preds)),
        "precision":   float(precision_score(labels, preds, zero_division=0)),
        "recall":      float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(specificity),
        "mcc":         float(matthews_corrcoef(labels, preds)),
    }


def evaluate_all_baselines(
    df: pd.DataFrame,
    out_path: Path | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    """Evaluate every baseline method on *df*, saving results incrementally.

    Args:
        df: DataFrame with columns ``fraudulent_name``, ``real_name``, ``label``.
        out_path: If provided, results are written to this CSV after each method
                  completes so progress is never lost.
        resume:   If True and *out_path* exists, skip methods already present in
                  that file and append new results to it.

    Returns:
        DataFrame with one row per method and columns:
        ``method``, ``roc_auc``, ``avg_precision``,
        ``f1``, ``accuracy``, ``precision``, ``recall``,
        ``specificity``, ``mcc``, ``best_threshold``.

        Threshold-dependent metrics are computed at the threshold that
        maximises F1.
    """
    labels = df["label"].astype(int).to_numpy()

    # Load previously completed results when resuming.
    completed: set[str] = set()
    results: list[dict] = []
    if resume and out_path is not None and out_path.exists():
        existing = pd.read_csv(out_path)
        results = existing.to_dict("records")
        completed = set(existing["method"].tolist())
        if completed:
            print(f"  Resuming — skipping already-completed methods: {sorted(completed)}")

    for name, fn in METHODS.items():
        if name in completed:
            print(f"  {name:<24}  [skipped — already in {out_path.name}]")
            continue

        scores = np.array([
            fn(row["fraudulent_name"], row["real_name"])
            for _, row in df.iterrows()
        ])

        roc_auc  = roc_auc_score(labels, scores)
        avg_prec = average_precision_score(labels, scores)
        tm       = _best_threshold_metrics(labels, scores)

        results.append({
            "method":         name,
            "roc_auc":        round(roc_auc,        6),
            "avg_precision":  round(avg_prec,        6),
            "f1":             round(tm["f1"],         6),
            "accuracy":       round(tm["accuracy"],   6),
            "precision":      round(tm["precision"],  6),
            "recall":         round(tm["recall"],     6),
            "specificity":    round(tm["specificity"],6),
            "mcc":            round(tm["mcc"],        6),
            "best_threshold": round(tm["threshold"],  6),
        })
        print(
            f"  {name:<24}  roc_auc={roc_auc:.4f}  avg_prec={avg_prec:.4f}  "
            f"f1={tm['f1']:.4f}  acc={tm['accuracy']:.4f}  "
            f"prec={tm['precision']:.4f}  rec={tm['recall']:.4f}  "
            f"spec={tm['specificity']:.4f}  mcc={tm['mcc']:.4f}  "
            f"thr={tm['threshold']:.4f}"
        )

        # Save incrementally so a crash doesn't lose completed work.
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(out_path, index=False)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _load_or_create_test_csv(csv_path: Path, pkl_path: Path) -> pd.DataFrame:
    """Load test.csv; generate it from test.pkl if it doesn't exist yet."""
    if csv_path.exists():
        return pd.read_csv(csv_path)

    print(f"test.csv not found — generating from {pkl_path} ...")
    with pkl_path.open("rb") as f:
        rows: list[tuple[str, str, float]] = pickle.load(f)

    df = pd.DataFrame(rows, columns=["fraudulent_name", "real_name", "label"])
    df["label"] = df["label"].astype(int)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df):,} rows to {csv_path}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        default="data/splits/test.csv",
        help="Path to the test split CSV (generated from test.pkl if absent).",
    )
    parser.add_argument(
        "--out",
        default="baseline_results.csv",
        help="Where to write the results CSV (default: baseline_results.csv).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip baselines already present in the output CSV and append new results.",
    )
    args = parser.parse_args()

    test_csv = ROOT / args.test
    test_pkl = test_csv.with_suffix(".pkl")
    out_path = ROOT / args.out

    df = _load_or_create_test_csv(test_csv, test_pkl)
    print(f"Loaded {len(df):,} test pairs  (positive rate: {df['label'].mean():.3f})\n")

    print("Running baselines ...")
    results_df = evaluate_all_baselines(df, out_path=out_path, resume=args.resume)

    print(f"\nResults saved to {out_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
