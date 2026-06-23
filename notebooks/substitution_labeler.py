#!/usr/bin/env python3
"""Notebook-friendly reviewer for OCR-confusable substitutions.

Open this file in JupyterLab or VS Code's notebook mode. It loads the ranked
DejaVu Sans substitution table, keeps the top two candidates per source
character by LEGIT score, renders them, and lets you mark each row keep or
discard.

Default input:
    .cache/exhaustive_character_substitutions/contextual_top5/ranked_substitutions.parquet

Default output:
    data/substitutions/ocr_confusable_legit_reviewed.parquet
    data/substitutions/ocr_confusable_legit_reviewed.csv
"""

from __future__ import annotations

import base64
import html
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

try:  # Optional runtime dependency for the notebook UI.
    import ipywidgets as widgets
    from IPython.display import HTML, display
except Exception as exc:  # pragma: no cover - import-time guidance only
    widgets = None
    HTML = None
    display = None
    _WIDGET_IMPORT_ERROR = exc
else:  # pragma: no cover - import-time guidance only
    _WIDGET_IMPORT_ERROR = None


def repo_root() -> Path:
    try:
        here = Path(__file__).resolve()
    except NameError:  # Running in a notebook cell.
        here = Path.cwd().resolve()
    if (here / "scripts").exists():
        return here
    if (here.parent / "scripts").exists():
        return here.parent
    return Path.cwd().resolve()


ROOT = repo_root()
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from ocr_common import TrOCRTextRenderer, default_dejavu_sans_path  # noqa: E402


DEFAULT_INPUT = (
    ROOT / ".cache/exhaustive_character_substitutions/contextual_top5/ranked_substitutions.parquet"
)
DEFAULT_OUTPUT_PARQUET = ROOT / "data/substitutions/ocr_confusable_legit_reviewed.parquet"
DEFAULT_OUTPUT_CSV = ROOT / "data/substitutions/ocr_confusable_legit_reviewed.csv"
DEFAULT_TOP_PER_SOURCE = 2

REQUIRED_COLUMNS = {
    "source_character": ["source_character", "real_span"],
    "replacement_character": ["replacement_character", "candidate_span"],
    "legit_q25": ["legit_q25"],
    "legit_median": ["legit_median"],
    "legit_positive_rate": ["legit_positive_rate"],
    "ocr_attack_contexts": ["ocr_attack_contexts"],
    "visual_similarity_score": ["visual_similarity_score"],
    "example_original_text": ["example_original_text", "target"],
    "example_substituted_text": ["example_substituted_text", "candidate"],
    "example_official_legit_score": ["example_official_legit_score", "official_legit_score"],
    "proxy_rank": ["proxy_rank"],
}


def _first_existing(frame: pd.DataFrame, names: list[str], default: Any = pd.NA) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def normalize_table(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    rename_map: dict[str, str] = {}
    for canonical, aliases in REQUIRED_COLUMNS.items():
        for alias in aliases:
            if alias in frame.columns:
                rename_map[alias] = canonical
                break
    frame = frame.rename(columns=rename_map)
    for canonical in REQUIRED_COLUMNS:
        if canonical not in frame.columns:
            frame[canonical] = pd.NA
    if "substitution_family" in frame.columns:
        frame = frame[frame["substitution_family"].eq("ocr_confusable") | frame["substitution_family"].isna()]
    if "meets_min_support" in frame.columns:
        frame = frame[frame["meets_min_support"].fillna(True)]
    if "legit_q25" in frame.columns:
        frame = frame[frame["legit_q25"].astype(float).gt(0.0)]
    sort_columns = [
        "source_character",
        "legit_q25",
        "legit_median",
        "legit_positive_rate",
        "ocr_attack_contexts",
        "visual_similarity_score",
        "example_official_legit_score",
        "proxy_rank",
    ]
    ascending = [True, False, False, False, False, False, False, True]
    existing_sort_columns = [c for c in sort_columns if c in frame.columns]
    existing_ascending = [ascending[sort_columns.index(c)] for c in existing_sort_columns]
    frame = frame.sort_values(existing_sort_columns, ascending=existing_ascending, na_position="last")
    frame["source_rank"] = frame.groupby("source_character", sort=False).cumcount() + 1
    frame = frame[frame["source_rank"].le(DEFAULT_TOP_PER_SOURCE)].reset_index(drop=True)
    frame["review_label"] = "keep"
    frame["review_state"] = "auto"
    frame["reviewed_at"] = pd.NA
    return frame


def render_to_data_uri(renderer: TrOCRTextRenderer, text: str) -> str:
    image = renderer.render_text(text)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def unicode_label(text: str) -> str:
    parts = []
    for char in text:
        parts.append(f"U+{ord(char):04X} {unicodedata.name(char, 'UNNAMED')}")
    return " | ".join(parts)


def make_html_image(renderer: TrOCRTextRenderer, text: str, label: str) -> str:
    uri = render_to_data_uri(renderer, text)
    return (
        "<div style='display:flex; flex-direction:column; gap:6px;'>"
        f"<div style='font: 600 12px/1.2 sans-serif; color:#444;'>{html.escape(label)}</div>"
        f"<img src='{uri}' style='max-width:100%; border:1px solid #ddd; background:#000; padding:4px;'/>"
        f"<div style='font: 12px/1.35 monospace; color:#555; white-space:pre-wrap;'>{html.escape(text)}</div>"
        "</div>"
    )


@dataclass
class ReviewPaths:
    input_path: Path = DEFAULT_INPUT
    output_parquet: Path = DEFAULT_OUTPUT_PARQUET
    output_csv: Path = DEFAULT_OUTPUT_CSV


class SubstitutionLabeler:
    def __init__(self, frame: pd.DataFrame, *, output: ReviewPaths, renderer: TrOCRTextRenderer) -> None:
        self.frame = frame.copy()
        self.output = output
        self.renderer = renderer
        self.frame["review_label"] = self.frame["review_label"].fillna("keep")
        self.frame["review_state"] = self.frame["review_state"].fillna("auto")
        self.frame["reviewed_at"] = self.frame["reviewed_at"].where(self.frame["reviewed_at"].notna(), pd.NA)
        self.frame["keep_threshold"] = float(self.frame["legit_q25"].astype(float).min())

        self.sources = list(dict.fromkeys(self.frame["source_character"].astype(str)))
        self.source_widget = widgets.Dropdown(
            options=[(src, src) for src in self.sources],
            description="Source",
            layout=widgets.Layout(width="260px"),
        )
        self.row_widget = widgets.Select(
            options=[],
            description="Top 2",
            layout=widgets.Layout(width="360px", height="150px"),
        )
        self.threshold_widget = widgets.FloatText(
            value=0.0,
            description="Keep >= ",
            step=0.1,
            layout=widgets.Layout(width="220px"),
        )
        self.label_widget = widgets.ToggleButtons(
            options=[("Keep", "keep"), ("Discard", "discard"), ("Skip", "skip")],
            description="Label",
            button_style="",
        )
        self.keep_button = widgets.Button(description="Keep", button_style="success", icon="check")
        self.discard_button = widgets.Button(description="Discard", button_style="danger", icon="times")
        self.auto_button = widgets.Button(description="Auto-fill", button_style="info", icon="magic")
        self.prev_button = widgets.Button(description="Prev", icon="arrow-left")
        self.next_button = widgets.Button(description="Next", icon="arrow-right")
        self.save_button = widgets.Button(description="Save", button_style="primary", icon="save")
        self.status = widgets.HTML()
        self.preview = widgets.Output()

        self.source_widget.observe(self._on_source_change, names="value")
        self.row_widget.observe(self._on_row_change, names="value")
        self.label_widget.observe(self._on_label_change, names="value")
        self.keep_button.on_click(lambda _: self._set_label("keep"))
        self.discard_button.on_click(lambda _: self._set_label("discard"))
        self.auto_button.on_click(lambda _: self._auto_label_current_source())
        self.prev_button.on_click(lambda _: self._step(-1))
        self.next_button.on_click(lambda _: self._step(+1))
        self.save_button.on_click(lambda _: self.save())

        self._populate_rows(self.sources[0] if self.sources else None)

    def _source_frame(self, source: str) -> pd.DataFrame:
        return self.frame[self.frame["source_character"].astype(str).eq(source)].copy()

    def _populate_rows(self, source: str | None) -> None:
        if source is None:
            self.row_widget.options = []
            return
        source_df = self._source_frame(source)
        options = []
        for idx, row in source_df.iterrows():
            title = (
                f"{int(row['source_rank'])}. {row['source_character']} -> {row['replacement_character']} "
                f"(q25 {float(row['legit_q25']):.3f})"
            )
            options.append((title, idx))
        self.row_widget.options = options
        if options:
            self.row_widget.value = options[0][1]

    def _current_index(self) -> int | None:
        return self.row_widget.value if self.row_widget.value is not None else None

    def _current_row(self) -> pd.Series | None:
        idx = self._current_index()
        if idx is None or idx not in self.frame.index:
            return None
        return self.frame.loc[idx]

    def _refresh(self) -> None:
        row = self._current_row()
        if row is None:
            return
        self.label_widget.unobserve(self._on_label_change, names="value")
        self.label_widget.value = row["review_label"]
        self.label_widget.observe(self._on_label_change, names="value")
        self.status.value = (
            f"<b>{html.escape(str(row['source_character']))}</b> -> "
            f"<b>{html.escape(str(row['replacement_character']))}</b>"
            f" | source rank {int(row['source_rank'])} | q25 {float(row['legit_q25']):.3f}"
            f" | median {float(row['legit_median']):.3f}"
            f" | attacks {int(row['ocr_attack_contexts'])}"
            f" | current label <b>{html.escape(str(row['review_label']))}</b>"
        )
        with self.preview:
            from IPython.display import clear_output

            clear_output(wait=True)
            source_text = str(row.get("example_original_text", row["source_character"]))
            replacement_text = str(row.get("example_substituted_text", row["replacement_character"]))
            legit_score = row.get("example_official_legit_score", pd.NA)
            summary = (
                "<div style='display:grid; grid-template-columns: 1fr 1fr; gap:18px;'>"
                f"{make_html_image(self.renderer, str(row['source_character']), 'Source character')}"
                f"{make_html_image(self.renderer, str(row['replacement_character']), 'Replacement character')}"
                "</div>"
                "<div style='height:12px;'></div>"
                "<div style='display:grid; grid-template-columns: 1fr 1fr; gap:18px;'>"
                f"{make_html_image(self.renderer, source_text, 'Example original text')}"
                f"{make_html_image(self.renderer, replacement_text, 'Example substituted text')}"
                "</div>"
                "<div style='margin-top:14px; font: 12px/1.45 monospace; color:#444; white-space:pre-wrap;'>"
                f"Source label: {html.escape(unicode_label(str(row['source_character'])))}\n"
                f"Replacement label: {html.escape(unicode_label(str(row['replacement_character'])))}\n"
                f"Example official LEGIT score: {'' if pd.isna(legit_score) else f'{float(legit_score):.3f}'}\n"
                f"Suggested keep threshold: {self.threshold_widget.value:.3f}\n"
                "</div>"
            )
            display(HTML(summary))

    def _set_label(self, label: str) -> None:
        row = self._current_row()
        if row is None:
            return
        self.frame.loc[self._current_index(), "review_label"] = label
        self.frame.loc[self._current_index(), "review_state"] = "manual" if label != "skip" else "skipped"
        self.frame.loc[self._current_index(), "reviewed_at"] = datetime.now(timezone.utc).isoformat()
        self._refresh()

    def _auto_label_current_source(self) -> None:
        source = str(self.source_widget.value)
        threshold = float(self.threshold_widget.value)
        mask = self.frame["source_character"].astype(str).eq(source)
        keep_mask = self.frame.loc[mask, "legit_q25"].astype(float).ge(threshold)
        self.frame.loc[mask, "review_label"] = keep_mask.map({True: "keep", False: "discard"}).to_numpy()
        self.frame.loc[mask, "review_state"] = "auto"
        self.frame.loc[mask, "reviewed_at"] = datetime.now(timezone.utc).isoformat()
        self._refresh()

    def _on_source_change(self, change: dict[str, Any]) -> None:
        self._populate_rows(str(change["new"]))

    def _on_row_change(self, change: dict[str, Any]) -> None:
        self._refresh()

    def _on_label_change(self, change: dict[str, Any]) -> None:
        if change.get("new") is None:
            return
        self.frame.loc[self._current_index(), "review_label"] = change["new"]
        self.frame.loc[self._current_index(), "review_state"] = "manual"
        self.frame.loc[self._current_index(), "reviewed_at"] = datetime.now(timezone.utc).isoformat()
        self._refresh()

    def _step(self, delta: int) -> None:
        options = list(self.row_widget.options)
        if not options:
            return
        values = [value for _, value in options]
        try:
            pos = values.index(self.row_widget.value)
        except ValueError:
            pos = 0
        pos = max(0, min(len(values) - 1, pos + delta))
        self.row_widget.value = values[pos]

    def save(self) -> None:
        self.output.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        self.output.output_csv.parent.mkdir(parents=True, exist_ok=True)
        out = self.frame.copy()
        out.to_parquet(self.output.output_parquet, index=False)
        out.to_csv(self.output.output_csv, index=False)
        self.status.value = self.status.value + f" | saved {self.output.output_parquet.name}"

    def widget(self):
        controls = widgets.VBox(
            [
                widgets.HBox([self.source_widget, self.row_widget]),
                widgets.HBox([self.threshold_widget, self.label_widget]),
                widgets.HBox([
                    self.keep_button,
                    self.discard_button,
                    self.auto_button,
                    self.prev_button,
                    self.next_button,
                    self.save_button,
                ]),
                self.status,
            ]
        )
        return widgets.VBox([controls, self.preview])


def load_review_table(
    path: Path = DEFAULT_INPUT,
    *,
    top_per_source: int = DEFAULT_TOP_PER_SOURCE,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = normalize_table(frame)
    if top_per_source <= 0:
        raise ValueError("top_per_source must be positive")
    return frame.groupby("source_character", sort=False).head(top_per_source).reset_index(drop=True)


def build_labeler(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_parquet: Path = DEFAULT_OUTPUT_PARQUET,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    top_per_source: int = DEFAULT_TOP_PER_SOURCE,
) -> SubstitutionLabeler:
    frame = load_review_table(input_path, top_per_source=top_per_source)
    renderer = TrOCRTextRenderer(font_path=default_dejavu_sans_path(), font_size=56, image_height=96)
    labeler = SubstitutionLabeler(
        frame,
        output=ReviewPaths(
            input_path=input_path,
            output_parquet=output_parquet,
            output_csv=output_csv,
        ),
        renderer=renderer,
    )
    return labeler


if widgets is None:  # pragma: no cover - import-time guidance only
    raise RuntimeError(
        "ipywidgets is required for the interactive labeler. "
        "Install it with `pip install -r requirements.txt` or `pip install ipywidgets`."
        f" Original import error: {_WIDGET_IMPORT_ERROR}"
    )


labeler = build_labeler()
display(labeler.widget())
