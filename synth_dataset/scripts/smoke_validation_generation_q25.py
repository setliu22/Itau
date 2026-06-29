#!/usr/bin/env python3
"""Smoke checks for the Q25 validation-generation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SYNTH_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SYNTH_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rf_evaluation  # noqa: E402
import validation_generator as generator  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(7)
    exact_rule_low = generator.Rule("exact", "single_character", "a", "α", 1.0, 100, "a_to_alpha")
    exact_rule_high = generator.Rule("exact", "single_character", "a", "а", 5.0, 100, "a_to_cyrillic_a")
    ocr_rule = generator.Rule("ocr", "single_character", "b", "ƅ", 4.0, 100, "b_to_bhook")
    forward_rule = generator.Rule("multichar_forward", "forward", "m", "rn", 4.5, 100, "m_to_rn")
    reverse_rule = generator.Rule("multichar_reverse", "reverse", "rn", "m", 4.2, 100, "rn_to_m")
    adjacent_index = {"abcd": [generator.AdjacentRule("abcd", "abdc", 3.0, 2, 3)]}
    rule_lookups = {
        "multichar_forward": [forward_rule],
        "multichar_reverse": [reverse_rule],
        "ocr": [ocr_rule],
        "exact": [exact_rule_low, exact_rule_high],
    }

    units = generator.make_units("abcd")
    units, adjacent_ops = generator.apply_adjacent_family(
        units,
        adjacent_index=adjacent_index,
        max_count=1,
        probability=1.0,
        rng=rng,
    )
    assert generator.units_to_text(units) == "abdc"
    assert len(adjacent_ops) == 1

    units = generator.make_units("mrn")
    units, forward_ops = generator.apply_family(
        units,
        family="multichar_forward",
        rules=[forward_rule],
        max_count=1,
        probability=1.0,
        temperature=0.0,
        rng=rng,
    )
    units, reverse_ops = generator.apply_family(
        units,
        family="multichar_reverse",
        rules=[reverse_rule],
        max_count=1,
        probability=1.0,
        temperature=0.0,
        rng=rng,
    )
    assert len(forward_ops) == 1 and len(reverse_ops) == 1
    assert reverse_ops[0]["position"] != 0, "reverse multichar undid newly created forward span"
    assert generator.units_to_text(units) == "rnm"

    units = generator.make_units("aa")
    units, exact_ops = generator.apply_family(
        units,
        family="exact",
        rules=[exact_rule_high],
        max_count=2,
        probability=1.0,
        temperature=0.0,
        rng=rng,
    )
    assert len(exact_ops) == 2
    assert len({op["position"] for op in exact_ops}) == 2

    units = generator.make_units("a")
    units, ops = generator.apply_family(
        units,
        family="exact",
        rules=[exact_rule_low, exact_rule_high],
        max_count=1,
        probability=1.0,
        temperature=0.0,
        rng=rng,
    )
    assert ops[0]["replacement"] == "а"

    weights = generator.q25_weights([1.0, 5.0], 1.0)
    assert weights[1] > weights[0]
    assert not np.isclose(weights[0], weights[1])

    assert generator.sample_attempt_count(3, 0.0, rng) == 0
    assert generator.sample_attempt_count(3, 1.0, rng) == 3

    params = {
        "max_adjacent_swaps": 0,
        "adjacent_apply_probability": 0.0,
        "max_multichar_forward": 0,
        "multichar_forward_apply_probability": 0.0,
        "multichar_forward_temperature": 0.0,
        "max_multichar_reverse": 0,
        "multichar_reverse_apply_probability": 0.0,
        "multichar_reverse_temperature": 0.0,
        "max_ocr_substitutions": 0,
        "ocr_apply_probability": 0.0,
        "ocr_selection_temperature": 0.0,
        "max_exact_lookalikes": 0,
        "exact_apply_probability": 0.0,
        "exact_selection_temperature": 0.0,
        "max_total_modifications": 1,
    }
    generated, fallback_ops, counts = generator.generate_spoof(
        "a",
        params=params,
        adjacent_index={},
        rule_lookups=rule_lookups,
        rng=rng,
    )
    assert generated != "a"
    assert counts["total"] == 1
    assert fallback_ops[0].get("fallback") is True

    toy = pd.DataFrame(
        [
            {"real_name": "alpha", "fraudulent_name": "alpha", "label": 0.0},
            {"real_name": "bravo", "fraudulent_name": "bravo", "label": 0.0},
            {"real_name": "charlie", "fraudulent_name": "charlie", "label": 0.0},
            {"real_name": "delta", "fraudulent_name": "delta", "label": 0.0},
            {"real_name": "echo", "fraudulent_name": "еcho", "label": 1.0},
            {"real_name": "foxtrot", "fraudulent_name": "foxtrοt", "label": 1.0},
            {"real_name": "golf", "fraudulent_name": "goƚf", "label": 1.0},
            {"real_name": "hotel", "fraudulent_name": "hoteƚ", "label": 1.0},
        ]
    )
    train_idx, holdout_idx = rf_evaluation.grouped_train_holdout_split(
        toy["label"].to_numpy(),
        toy["real_name"].to_numpy(),
        train_fraction=0.5,
        seed=7,
    )
    assert not (set(toy.iloc[train_idx]["real_name"]) & set(toy.iloc[holdout_idx]["real_name"]))
    assert rf_evaluation.rf_predictability_from_ba(0.4) == 0.6
    assert rf_evaluation.rf_predictability_from_ba(0.5) == 0.5
    assert rf_evaluation.rf_predictability_from_ba(0.9) == 0.9

    try:
        import optuna

        study = optuna.create_study(directions=["maximize", "minimize", "minimize"])
        assert [direction.name for direction in study.directions] == ["MAXIMIZE", "MINIMIZE", "MINIMIZE"]
    except ModuleNotFoundError:
        pass

    print("Q25 validation-generation smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
