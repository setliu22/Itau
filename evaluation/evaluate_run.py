"""Evaluate a trained run on the test set and report classification metrics.

Loads the encoder + similarity head from a saved checkpoint, runs inference
over the test split, and reports ROC-AUC, average precision, and
threshold-dependent metrics (accuracy, precision, recall, F1, specificity,
MCC) at two thresholds:
  - Best-F1 threshold: the score cut-off that maximises F1 on the test set.
  - Fixed 0.5 threshold: useful as a quick sanity check.

Usage:
    # Evaluate best checkpoint of a run:
    python evaluation/evaluate_run.py --run transformer_attn

    # Evaluate the latest (last-epoch) checkpoint instead:
    python evaluation/evaluate_run.py --run transformer_attn --checkpoint latest

    # Override the test split path:
    python evaluation/evaluate_run.py --run transformer_attn --test data/splits/test.pkl

    # Save results to a CSV:
    python evaluation/evaluate_run.py --run transformer_attn --out outputs/eval_transformer_attn.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.encoder import VisualEncoder
from models.similarity import SimilarityHead
from training.dataset import NamePairDataset, collate_fn


# ---------------------------------------------------------------------------
# Config / model loading
# ---------------------------------------------------------------------------

def load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml found in {run_dir}")
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device) -> tuple[VisualEncoder, SimilarityHead]:
    r = cfg["rendering"]
    s = cfg["slicing"]
    m = cfg["model"]
    slice_dim = r["height"] * s["slice_width"]
    encoder_type = m.get("encoder_type", "conv1d")
    if encoder_type == "visual":
        encoder_type = "conv1d"
    encoder = VisualEncoder(
        slice_dim=slice_dim,
        embed_dim=m["embed_dim"],
        pooling=m["pooling"],
        encoder_type=encoder_type,
    ).to(device)
    head = SimilarityHead(embed_dim=m["embed_dim"]).to(device)
    return encoder, head


def load_checkpoint(
    run_dir: Path,
    which: str,
    encoder: VisualEncoder,
    head: SimilarityHead,
    device: torch.device,
) -> dict:
    """Load encoder and head weights from best.pt or latest.pt.

    Returns the raw checkpoint dict (useful for printing epoch / val_auc).
    """
    fname = "best.pt" if which == "best" else "latest.pt"
    ckpt_path = run_dir / fname
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])
    return ckpt


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    encoder: VisualEncoder,
    head: SimilarityHead,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over *loader* and return (scores, labels) as numpy arrays.

    scores: sigmoid probabilities in [0, 1] — higher means more likely a spoof.
    labels: ground-truth binary labels (0 or 1).
    """
    encoder.eval()
    head.eval()

    all_scores: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for slices_a, lengths_a, slices_b, lengths_b, labels in loader:
            slices_a  = slices_a.to(device)
            lengths_a = lengths_a.to(device)
            slices_b  = slices_b.to(device)
            lengths_b = lengths_b.to(device)

            emb_a  = encoder(slices_a, lengths_a)
            emb_b  = encoder(slices_b, lengths_b)
            logits = head(emb_a, emb_b)
            probs  = torch.sigmoid(logits).cpu().tolist()

            if isinstance(probs, float):
                probs = [probs]
            all_scores.extend(probs)
            all_labels.extend(labels.tolist())

    return np.array(all_scores, dtype=np.float32), np.array(all_labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "threshold":   threshold,
        "accuracy":    float(accuracy_score(labels, preds)),
        "precision":   float(precision_score(labels, preds, zero_division=0)),
        "recall":      float(recall_score(labels, preds, zero_division=0)),
        "f1":          float(f1_score(labels, preds, zero_division=0)),
        "specificity": specificity,
        "mcc":         float(matthews_corrcoef(labels, preds)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return the threshold that maximises F1, computed in O(n log n) via
    precision_recall_curve rather than an O(n²) scan over unique scores."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # precision/recall have one extra element at the end (boundary condition);
    # thresholds has len(precision) - 1 entries.
    denom = precision[:-1] + recall[:-1]
    f1s   = np.where(denom > 0, 2 * precision[:-1] * recall[:-1] / denom, 0.0)
    return float(thresholds[np.argmax(f1s)])


def compute_all_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    roc_auc  = float(roc_auc_score(labels, scores))
    avg_prec = float(average_precision_score(labels, scores))
    best_t   = best_f1_threshold(labels, scores)

    best_tm  = threshold_metrics(labels, scores, best_t)
    fixed_tm = threshold_metrics(labels, scores, 0.5)

    return {
        "roc_auc":     roc_auc,
        "avg_precision": avg_prec,
        "best_threshold": best_tm,
        "fixed_threshold": fixed_tm,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _hline(char: str = "─", width: int = 60) -> str:
    return char * width


def _print_confusion_matrix(tp: int, tn: int, fp: int, fn: int) -> None:
    """Print a labelled ASCII confusion matrix.

    Rows = actual class, columns = predicted class.

         Predicted
         Neg   Pos
    Actual Neg │ TN  │ FP │
    Actual Pos │ FN  │ TP │
    """
    # Width of each cell — wide enough for the largest count.
    w = max(len(str(n)) for n in (tp, tn, fp, fn))
    w = max(w, 4)   # at least 4 chars so labels fit

    div = f"    ├{'─'*(w+2)}┼{'─'*(w+2)}┤"
    top = f"    ┌{'─'*(w+2)}┬{'─'*(w+2)}┐"
    bot = f"    └{'─'*(w+2)}┴{'─'*(w+2)}┘"

    col_head  = f"    {'':8}  {'Pred Neg':^{w+2}}  {'Pred Pos':^{w+2}}"
    row_neg   = f"    {'Actual Neg':8}│ {tn:>{w}} │ {fp:>{w}} │"
    row_pos   = f"    {'Actual Pos':8}│ {fn:>{w}} │ {tp:>{w}} │"

    print(col_head)
    print(top)
    print(row_neg)
    print(div)
    print(row_pos)
    print(bot)


def print_metrics(run_name: str, ckpt_info: str, metrics: dict) -> None:
    print()
    print(_hline("═"))
    print(f"  Run:        {run_name}")
    print(f"  Checkpoint: {ckpt_info}")
    print(_hline("═"))

    print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")
    print(f"  Avg precision:   {metrics['avg_precision']:.4f}")

    for label, key in [("Best-F1 threshold", "best_threshold"),
                       ("Fixed 0.5 threshold", "fixed_threshold")]:
        tm = metrics[key]
        print()
        print(f"  {label}  (thr={tm['threshold']:.4f})")
        print(_hline("─", 50))
        print(f"    Accuracy:    {tm['accuracy']:.4f}")
        print(f"    Precision:   {tm['precision']:.4f}")
        print(f"    Recall:      {tm['recall']:.4f}")
        print(f"    F1:          {tm['f1']:.4f}")
        print(f"    Specificity: {tm['specificity']:.4f}")
        print(f"    MCC:         {tm['mcc']:.4f}")
        print()
        _print_confusion_matrix(tm["tp"], tm["tn"], tm["fp"], tm["fn"])

    print()
    print(_hline("═"))


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_results(out_path: Path, run_name: str, ckpt_info: str, metrics: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bt  = metrics["best_threshold"]
    ft  = metrics["fixed_threshold"]
    row = {
        "run_name":           run_name,
        "checkpoint":         ckpt_info,
        "roc_auc":            round(metrics["roc_auc"],        6),
        "avg_precision":      round(metrics["avg_precision"],  6),
        # best-F1 threshold
        "best_threshold":     round(bt["threshold"],  6),
        "best_accuracy":      round(bt["accuracy"],   6),
        "best_precision":     round(bt["precision"],  6),
        "best_recall":        round(bt["recall"],     6),
        "best_f1":            round(bt["f1"],         6),
        "best_specificity":   round(bt["specificity"],6),
        "best_mcc":           round(bt["mcc"],        6),
        "best_tp":            bt["tp"],
        "best_tn":            bt["tn"],
        "best_fp":            bt["fp"],
        "best_fn":            bt["fn"],
        # fixed 0.5 threshold
        "fixed_accuracy":     round(ft["accuracy"],   6),
        "fixed_precision":    round(ft["precision"],  6),
        "fixed_recall":       round(ft["recall"],     6),
        "fixed_f1":           round(ft["f1"],         6),
        "fixed_specificity":  round(ft["specificity"],6),
        "fixed_mcc":          round(ft["mcc"],        6),
        "fixed_tp":           ft["tp"],
        "fixed_tn":           ft["tn"],
        "fixed_fp":           ft["fp"],
        "fixed_fn":           ft["fn"],
    }
    write_header = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  Results appended to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained run on the test set."
    )
    parser.add_argument(
        "--run", required=True,
        help="Run name as it appears under outputs/runs/ (e.g. transformer_attn).",
    )
    parser.add_argument(
        "--checkpoint", choices=["best", "latest"], default="best",
        help="Which checkpoint to load: 'best' (highest val AUC) or 'latest' "
             "(last epoch). Default: best.",
    )
    parser.add_argument(
        "--test", default=None,
        help="Path to the test split pkl file. Defaults to the path stored in "
             "the run's config.yaml (data.test_pkl).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Inference batch size (default: 64).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader num_workers (default: 0; set >0 on Linux/Mac).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Optional path to append results to as a CSV row.",
    )
    args = parser.parse_args()

    run_dir = ROOT / "outputs" / "runs" / args.run
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    cfg    = load_config(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder, head = build_model(cfg, device)
    ckpt = load_checkpoint(run_dir, args.checkpoint, encoder, head, device)

    ckpt_auc  = ckpt.get("val_auc") or ckpt.get("best_auc")
    auc_str   = f"{ckpt_auc:.4f}" if ckpt_auc is not None else "?"
    ckpt_info = (
        f"{args.checkpoint}.pt  "
        f"(epoch={ckpt.get('epoch', '?')}, val_auc={auc_str})"
    )
    print(f"Loaded checkpoint: {ckpt_info}")

    # Resolve test split path
    if args.test is not None:
        test_pkl = Path(args.test)
    else:
        test_pkl = ROOT / cfg["data"]["test_pkl"]

    if not test_pkl.exists():
        raise SystemExit(f"Test split not found: {test_pkl}")

    r = cfg["rendering"]
    s = cfg["slicing"]
    test_ds = NamePairDataset(
        pkl_path=test_pkl,
        height=r["height"],
        background=r["background"],
        slice_width=s["slice_width"],
        stride=s["stride"],
        remove_padding=s["remove_padding"],
        pad_to_width=s.get("pad_to_width"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Test set: {len(test_ds):,} pairs  ({test_pkl})")

    scores, labels = run_inference(encoder, head, test_loader, device)
    pos_rate = labels.mean()
    print(f"Inference complete — positive rate: {pos_rate:.3f}\n")

    metrics = compute_all_metrics(labels, scores)
    print_metrics(args.run, ckpt_info, metrics)

    if args.out is not None:
        save_results(Path(args.out), args.run, ckpt_info, metrics)


if __name__ == "__main__":
    main()
