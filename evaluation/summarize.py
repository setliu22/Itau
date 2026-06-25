"""Summarise sweep results from outputs/results.csv and compare against baselines.

Prints:
  1. Top-5 configurations by val AUC.
  2. Marginal effect of each design axis (pooling, remove_padding, background,
     slice_width): for each axis, groups all runs by that axis's value and
     reports mean ± std val AUC, averaged over all other axes.
  3. Full comparison table: every Conv1D sweep run merged with all baselines,
     sorted by ROC-AUC descending.
  4. Best-vs-baselines table: the single best Conv1D config alongside every
     baseline, sorted by ROC-AUC descending.

Both comparison tables are saved to outputs/final_comparison.csv.

NOTE: Conv1D ROC-AUC values come from the validation set; baseline values come
from the test set.  Direct numeric comparisons should account for this split
difference.

Usage:
    python evaluation/summarize.py
    python evaluation/summarize.py --results outputs/results.csv
    python evaluation/summarize.py --baselines outputs/results_baselines.csv
    python evaluation/summarize.py --out outputs/final_comparison.csv
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Axes whose marginal effect we report.  stride is intentionally excluded
# because it is coupled to slice_width by construction.
# ---------------------------------------------------------------------------
MARGINAL_AXES = ["pooling", "remove_padding", "background", "slice_width"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No data in {path}")
    for row in rows:
        row["best_val_auc"] = float(row["best_val_auc"])
        row["slice_width"]  = int(row["slice_width"])
        row["stride"]       = int(row["stride"])
    return rows


def is_whole_image(row: dict) -> bool:
    """True for the padded single-strip whole-image runs."""
    ptw = row.get("pad_to_width", "")
    return ptw not in ("", "None", None)


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def std(values: list[float]) -> float:
    return statistics.pstdev(values)   # population std (no Bessel correction)


def hline(char: str = "─", width: int = 72) -> str:
    return char * width


def fmt_auc(v: float) -> str:
    return f"{v:.4f}"


def fmt_mean_std(m: float, s: float) -> str:
    return f"{m:.4f} ± {s:.4f}"


# ---------------------------------------------------------------------------
# Top-N table
# ---------------------------------------------------------------------------

def top_n(rows: list[dict], n: int = 5) -> None:
    sorted_rows = sorted(rows, key=lambda r: r["best_val_auc"], reverse=True)
    top = sorted_rows[:n]

    print(hline("═"))
    print(f"  TOP {n} CONFIGURATIONS BY VAL AUC  ({len(rows)} total runs)")
    print(hline("═"))

    col_widths = {
        "rank":      4,
        "auc":       8,
        "pooling":   11,
        "padding":   8,
        "bg":        6,
        "sw":        3,
        "st":        3,
    }

    header = (
        f"  {'#':>{col_widths['rank']}}  "
        f"{'AUC':>{col_widths['auc']}}  "
        f"{'pooling':<{col_widths['pooling']}}  "
        f"{'pad':>{col_widths['padding']}}  "
        f"{'bg':<{col_widths['bg']}}  "
        f"{'sw':>{col_widths['sw']}}  "
        f"{'st':>{col_widths['st']}}"
    )
    print(header)
    print(hline())

    for rank, row in enumerate(top, 1):
        pad_str = "yes" if str(row["remove_padding"]).lower() in ("true", "1") else "no"
        print(
            f"  {rank:>{col_widths['rank']}}  "
            f"{fmt_auc(row['best_val_auc']):>{col_widths['auc']}}  "
            f"{row['pooling']:<{col_widths['pooling']}}  "
            f"{pad_str:>{col_widths['padding']}}  "
            f"{row['background']:<{col_widths['bg']}}  "
            f"{row['slice_width']:>{col_widths['sw']}}  "
            f"{row['stride']:>{col_widths['st']}}"
        )

    print(hline())


# ---------------------------------------------------------------------------
# Marginal effects
# ---------------------------------------------------------------------------

def marginal_effects(rows: list[dict]) -> None:
    print()
    print(hline("═"))
    print("  MARGINAL EFFECT OF EACH DESIGN AXIS  (mean ± std val AUC)")
    print(hline("═"))

    for axis in MARGINAL_AXES:
        # Group AUC values by this axis's value
        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            key = row[axis]
            groups[key].append(row["best_val_auc"])

        print(f"\n  {axis}")
        print(f"  {'value':<20}  {'n':>4}  {'mean ± std':>16}  {'min':>8}  {'max':>8}")
        print("  " + hline("─", 64))

        # Sort values sensibly: numeric if possible, else lexicographic
        def sort_key(k):
            try:
                return (0, float(k))
            except (ValueError, TypeError):
                return (1, str(k).lower())

        for value in sorted(groups.keys(), key=sort_key):
            aucs = groups[value]
            print(
                f"  {str(value):<20}  "
                f"{len(aucs):>4}  "
                f"{fmt_mean_std(mean(aucs), std(aucs)):>16}  "
                f"{fmt_auc(min(aucs)):>8}  "
                f"{fmt_auc(max(aucs)):>8}"
            )

    print()
    print(hline("═"))


# ---------------------------------------------------------------------------
# Stride analysis
# ---------------------------------------------------------------------------

def stride_analysis(rows: list[dict]) -> None:
    """Compare overlapping vs non-overlapping strides, overall and per slice_width."""
    strip_rows = [r for r in rows if not is_whole_image(r)]
    if not strip_rows:
        return

    print()
    print(hline("═"))
    print("  STRIDE ANALYSIS  (regular strip runs only, mean ± std val AUC)")
    print(hline("═"))

    # --- overall: overlapping vs non-overlapping ---
    overlap_groups: dict[str, list[float]] = defaultdict(list)
    for row in strip_rows:
        key = "overlapping" if row["stride"] < row["slice_width"] else "non-overlapping"
        overlap_groups[key].append(row["best_val_auc"])

    print("\n  Overall: overlapping (stride < sw) vs non-overlapping (stride == sw)")
    print(f"  {'type':<18}  {'n':>4}  {'mean ± std':>16}  {'min':>8}  {'max':>8}")
    print("  " + hline("─", 60))
    for key in ["overlapping", "non-overlapping"]:
        if key not in overlap_groups:
            continue
        aucs = overlap_groups[key]
        print(
            f"  {key:<18}  {len(aucs):>4}  "
            f"{fmt_mean_std(mean(aucs), std(aucs)):>16}  "
            f"{fmt_auc(min(aucs)):>8}  {fmt_auc(max(aucs)):>8}"
        )

    # --- per slice_width: each (sw, stride) pair ---
    print("\n  Per slice_width: each (sw, stride) pair")
    print(f"  {'sw':>4}  {'stride':>6}  {'overlap%':>8}  {'n':>4}  "
          f"{'mean ± std':>16}  {'min':>8}  {'max':>8}")
    print("  " + hline("─", 66))

    sw_st_groups: dict[tuple, list[float]] = defaultdict(list)
    for row in strip_rows:
        sw_st_groups[(row["slice_width"], row["stride"])].append(row["best_val_auc"])

    for (sw, st) in sorted(sw_st_groups.keys()):
        aucs = sw_st_groups[(sw, st)]
        overlap_pct = round((1 - st / sw) * 100)
        print(
            f"  {sw:>4}  {st:>6}  {overlap_pct:>7}%  {len(aucs):>4}  "
            f"{fmt_mean_std(mean(aucs), std(aucs)):>16}  "
            f"{fmt_auc(min(aucs)):>8}  {fmt_auc(max(aucs)):>8}"
        )

    print()
    print(hline("═"))


# ---------------------------------------------------------------------------
# Baseline loader
# ---------------------------------------------------------------------------

def load_baselines(path: Path) -> list[dict]:
    """Load results_baselines.csv into a list of dicts with typed numeric fields."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("method", "").strip()]
    if not rows:
        raise SystemExit(f"No data in {path}")
    for row in rows:
        row["roc_auc"]              = float(row["roc_auc"])
        row["avg_precision"]        = float(row["avg_precision"])
        row["f1_at_best_threshold"] = float(row["f1_at_best_threshold"])
    return rows


# ---------------------------------------------------------------------------
# Unified-row schema
# ---------------------------------------------------------------------------
# Both sweep runs and baselines are normalised to this common dict shape so
# they can be sorted and printed in the same table.
#
# Columns present for Conv1D sweep rows only (empty string for baselines):
#   pooling, remove_padding, background, slice_width, stride
# Columns present for baseline rows only (empty string for sweep rows):
#   avg_precision, f1_at_best_threshold

_UNIFIED_FIELDS = [
    "name", "type", "roc_auc",
    "avg_precision", "f1_at_best_threshold",
    "pooling", "remove_padding", "background", "slice_width", "stride",
]


def _sweep_to_unified(row: dict) -> dict:
    pad_str = "yes" if str(row["remove_padding"]).lower() in ("true", "1") else "no"
    return {
        "name":                  row["run_name"],
        "type":                  "conv1d",
        "roc_auc":               row["best_val_auc"],   # val split
        "avg_precision":         "",
        "f1_at_best_threshold":  "",
        "pooling":               row["pooling"],
        "remove_padding":        pad_str,
        "background":            row["background"],
        "slice_width":           str(row["slice_width"]),
        "stride":                str(row["stride"]),
    }


def _baseline_to_unified(row: dict) -> dict:
    return {
        "name":                  row["method"],
        "type":                  "baseline",
        "roc_auc":               row["roc_auc"],        # test split
        "avg_precision":         row["avg_precision"],
        "f1_at_best_threshold":  row["f1_at_best_threshold"],
        "pooling":               "",
        "remove_padding":        "",
        "background":            "",
        "slice_width":           "",
        "stride":                "",
    }


def build_unified(sweep_rows: list[dict], baseline_rows: list[dict]) -> list[dict]:
    """Merge and sort both result sets by roc_auc descending."""
    unified = (
        [_sweep_to_unified(r) for r in sweep_rows]
        + [_baseline_to_unified(r) for r in baseline_rows]
    )
    return sorted(unified, key=lambda r: r["roc_auc"], reverse=True)


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------

def _fmt_opt(v, fmt=".4f") -> str:
    """Format a numeric value or return '–' for empty/missing."""
    if v == "" or v is None:
        return "–"
    return format(float(v), fmt)


def print_comparison_table(rows: list[dict], title: str, subtitle: str = "") -> None:
    name_w = max(len("name"), max(len(r["name"]) for r in rows))
    # Fixed widths for other columns
    W = dict(type=8, roc_auc=8, avg_prec=8, f1=8,
             pooling=11, rm_pad=6, bg=5, sw=3, st=3)
    # Total row width: leading "  " + rank(4) + "  " + name + rest
    rest_w = sum(W.values()) + 2 * len(W) + 6  # approx separators
    total_w = 4 + 2 + name_w + rest_w + 4

    print()
    print(hline("═", total_w))
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(hline("═", total_w))

    header = (
        f"  {'#':>4}  "
        f"{'name':<{name_w}}  "
        f"{'type':<{W['type']}}  "
        f"{'roc_auc':>{W['roc_auc']}}  "
        f"{'avg_prec':>{W['avg_prec']}}  "
        f"{'f1':>{W['f1']}}  "
        f"{'pooling':<{W['pooling']}}  "
        f"{'rm_pad':>{W['rm_pad']}}  "
        f"{'bg':<{W['bg']}}  "
        f"{'sw':>{W['sw']}}  "
        f"{'st':>{W['st']}}"
    )
    print(header)
    print(hline("─", total_w))

    for rank, row in enumerate(rows, 1):
        # Highlight baselines rows so they stand out visually
        marker = " *" if row["type"] == "baseline" else "  "
        sw_str = row["slice_width"] if row["slice_width"] != "" else "–"
        st_str = row["stride"]      if row["stride"]      != "" else "–"
        bg_str = row["background"]  if row["background"]  != "" else "–"
        pl_str = row["pooling"]     if row["pooling"]     != "" else "–"
        rp_str = row["remove_padding"] if row["remove_padding"] != "" else "–"

        print(
            f"{marker}{rank:>4}  "
            f"{row['name']:<{name_w}}  "
            f"{row['type']:<{W['type']}}  "
            f"{row['roc_auc']:.4f}  "
            f"{_fmt_opt(row['avg_precision']):>{W['avg_prec']}}  "
            f"{_fmt_opt(row['f1_at_best_threshold']):>{W['f1']}}  "
            f"{pl_str:<{W['pooling']}}  "
            f"{rp_str:>{W['rm_pad']}}  "
            f"{bg_str:<{W['bg']}}  "
            f"{sw_str:>{W['sw']}}  "
            f"{st_str:>{W['st']}}"
        )

    print(hline("─", total_w))
    n_sweep    = sum(1 for r in rows if r["type"] == "conv1d")
    n_baseline = sum(1 for r in rows if r["type"] == "baseline")
    print(
        f"  {len(rows)} rows  "
        f"({n_sweep} conv1d [val AUC] + {n_baseline} baseline [test AUC])"
        f"  * = baseline"
    )


# ---------------------------------------------------------------------------
# CSV writer for both tables in one file
# ---------------------------------------------------------------------------

def save_both_tables(
    full: list[dict],
    best_vs: list[dict],
    out_path: Path,
) -> None:
    """Write both comparison tables to *out_path*, separated by a blank line.

    Each section has its own header row so either half can be read with
    pandas using ``skiprows`` or ``comment='#'``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=_UNIFIED_FIELDS, extrasaction="ignore"
        )

        f.write(
            "# FULL COMPARISON TABLE"
            " (all Conv1D sweep runs + baselines, sorted by ROC-AUC desc)\n"
            "# Conv1D roc_auc = val split; baseline roc_auc = test split\n"
        )
        writer.writeheader()
        writer.writerows(full)

        f.write("\n")   # blank line between sections

        f.write(
            "# BEST CONV1D CONFIG VS BASELINES"
            " (sorted by ROC-AUC desc)\n"
        )
        writer.writeheader()
        writer.writerows(best_vs)

    print(f"\n  Saved both tables → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="outputs/results.csv",
        help="Path to the results CSV produced by strip_design_sweep.py",
    )
    parser.add_argument(
        "--baselines",
        default="outputs/results_baselines.csv",
        help="Path to the baselines results CSV produced by evaluation/baselines.py",
    )
    parser.add_argument(
        "--out",
        default="outputs/final_comparison.csv",
        help="Where to write the merged comparison tables",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of top configurations to show (default: 5)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")

    rows = load(results_path)

    # ------------------------------------------------------------------
    # Existing sweep analyses (unchanged)
    # ------------------------------------------------------------------
    top_n(rows, n=args.top)
    marginal_effects([r for r in rows if not is_whole_image(r)])
    stride_analysis(rows)

    # ------------------------------------------------------------------
    # Merged comparison tables
    # ------------------------------------------------------------------
    baselines_path = Path(args.baselines)
    if not baselines_path.exists():
        print(f"\nBaselines file not found at {baselines_path} — skipping comparison.")
        print("Run `python evaluation/baselines.py` first to generate it.")
        return

    baseline_rows = load_baselines(baselines_path)

    # Full table: all sweep runs + all baselines
    full_unified = build_unified(rows, baseline_rows)

    # Best-vs-baselines: single best Conv1D run + all baselines
    best_sweep_row = max(rows, key=lambda r: r["best_val_auc"])
    best_unified   = build_unified([best_sweep_row], baseline_rows)

    n_sweep = len(rows)
    n_base  = len(baseline_rows)

    print_comparison_table(
        full_unified,
        title=f"FULL COMPARISON — {n_sweep} Conv1D runs + {n_base} baselines",
        subtitle=(
            "Conv1D roc_auc = validation split  |  "
            "baseline roc_auc = test split"
        ),
    )
    print_comparison_table(
        best_unified,
        title="BEST CONV1D CONFIG vs BASELINES",
        subtitle=(
            f"Best run: {best_sweep_row['run_name']}  "
            f"(val_auc={best_sweep_row['best_val_auc']:.4f})"
        ),
    )

    save_both_tables(full_unified, best_unified, Path(args.out))


if __name__ == "__main__":
    main()
