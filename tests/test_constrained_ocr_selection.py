from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from glyph_identity import score_source_identity  # noqa: E402
from ocr_common import TrOCRTextReader, default_dejavu_sans_path  # noqa: E402
from evaluate_holdout_ocr_transfer import summarize_strategy  # noqa: E402
from transform_pairs_with_ocr_atlas import (  # noqa: E402
    AtlasSpoofGenerator,
    apply_operations,
    constrained_candidate_rank,
    contextual_operation_weight,
    exact_output_rate,
    generate_model_aware_operation_combinations,
    merge_non_overlapping_operations,
    ocr_goal_pass,
)
from rank_contextual_ocr_substitutions import (  # noqa: E402
    aggregate_rankings,
    apply_substitution,
    build_substitution_table,
    select_top_replacements_by_source,
)


class GlyphIdentityTests(unittest.TestCase):
    def test_komi_sje_is_not_admissible_as_latin_s(self) -> None:
        from PIL import ImageFont

        font = ImageFont.truetype(str(default_dejavu_sans_path()), 144)
        score = score_source_identity(
            "s",
            "\u050d",
            font=font,
            canonical_spans="abcdefghijklmnopqrstuvwxyz0123456789",
        )

        self.assertLess(score.source_margin, 0.0)


class ConstrainedRankTests(unittest.TestCase):
    def rank(self, **overrides: object) -> tuple[float, ...]:
        values = {
            "feasible": True,
            "legit_pass": True,
            "ocr_pass": True,
            "text_pass": True,
            "family_pass": True,
            "legit_score": 1.0,
            "substitutions": 1,
            "worst_ocr_score": 0.2,
            "visual_floor": 0.9,
            "raw_score": 0.5,
            "is_current": False,
        }
        values.update(overrides)
        return constrained_candidate_rank(**values)

    def test_feasible_candidate_beats_stronger_infeasible_attack(self) -> None:
        feasible = self.rank(worst_ocr_score=0.8)
        infeasible = self.rank(
            feasible=False,
            ocr_pass=False,
            legit_score=10.0,
            worst_ocr_score=0.0,
        )
        self.assertLess(feasible, infeasible)

    def test_legibility_precedes_ocr_strength_inside_feasible_set(self) -> None:
        more_legible = self.rank(legit_score=2.0, worst_ocr_score=0.8)
        more_ocr_adversarial = self.rank(legit_score=1.0, worst_ocr_score=0.0)
        self.assertLess(more_legible, more_ocr_adversarial)

    def test_fewer_substitutions_break_legibility_tie(self) -> None:
        one_edit = self.rank(substitutions=1, worst_ocr_score=0.8)
        two_edits = self.rank(substitutions=2, worst_ocr_score=0.0)
        self.assertLess(one_edit, two_edits)


class ContextualSubstitutionRankTests(unittest.TestCase):
    def test_applies_only_the_selected_occurrence(self) -> None:
        self.assertEqual(
            apply_substitution("letter", start=3, end=4, candidate_span="7"),
            "let7er",
        )

    def test_legit_lower_quartile_ranks_ocr_feasible_substitutions(self) -> None:
        contexts = pd.DataFrame(
            [
                self.context(0, "a", "@", 1.0),
                self.context(0, "a", "@", 2.0),
                self.context(1, "b", "8", 3.0),
                self.context(1, "b", "8", 4.0),
            ]
        )

        ranked = aggregate_rankings(contexts, min_support=2)

        self.assertEqual(list(ranked["candidate_span"]), ["8", "@"])
        self.assertEqual(list(ranked["proxy_rank"]), [1, 2])

    def test_contextual_weight_prefers_supported_legible_attack(self) -> None:
        weak = contextual_operation_weight(
            {"legit_q25": -2.0, "ocr_attack_rate": 0.25, "ocr_attack_contexts": 1}
        )
        strong = contextual_operation_weight(
            {"legit_q25": 3.0, "ocr_attack_rate": 0.75, "ocr_attack_contexts": 8}
        )

        self.assertGreater(strong, weak)

    def test_keeps_at_most_five_good_replacements_per_source(self) -> None:
        contexts = pd.DataFrame(
            [
                self.context(index, "a", chr(ord("k") + index), float(10 - index))
                for index in range(6)
                for _ in range(2)
            ]
        )
        ranked = aggregate_rankings(contexts, min_support=2)

        top = select_top_replacements_by_source(
            ranked,
            expected_sources="ab-",
            limit=5,
        )

        self.assertEqual(len(top), 5)
        self.assertEqual(list(top["source_rank"]), [1, 2, 3, 4, 5])

    def test_persistent_table_records_ocr_and_legit_provenance(self) -> None:
        contexts = pd.DataFrame([self.context(0, "a", "@", 2.0) for _ in range(2)])
        ranked = aggregate_rankings(contexts, min_support=2)
        top = select_top_replacements_by_source(ranked, expected_sources="a", limit=5)
        args = Namespace(
            render_variants="robust",
            model_names=["printed", "handwritten"],
            legit_model_name="official-legit",
            legit_processor_name="official-processor",
        )

        table = build_substitution_table(top, contexts=contexts, args=args)

        self.assertEqual(table.iloc[0]["source_character"], "a")
        self.assertEqual(table.iloc[0]["replacement_character"], "@")
        self.assertIn("white text on black", table.iloc[0]["ocr_renderer"])
        self.assertIn("word-pair", table.iloc[0]["legit_score_scope"])
        self.assertEqual(table.iloc[0]["example_original_text"], "sample")
        self.assertEqual(table.iloc[0]["example_substituted_text"], "s@mple")

    @staticmethod
    def context(
        substitution_id: int,
        real_span: str,
        candidate_span: str,
        legit_score: float,
    ) -> dict[str, object]:
        return {
            "substitution_id": substitution_id,
            "real_span": real_span,
            "candidate_span": candidate_span,
            "substitution_family": "ocr_confusable",
            "operation": "single_homoglyph",
            "visual_similarity_score": 0.9,
            "ocr_real_rate": 0.0,
            "ocr_wrong_rate": 1.0,
            "bucket": "safe_hard",
            "clean_ocr_eligible": True,
            "character_ocr_attack": True,
            "whole_word_ocr_attack": True,
            "ocr_feasible": True,
            "official_legit_score": legit_score,
            "target": "sample",
            "candidate": "s@mple",
            "development_ocr_results_json": '{"printed":{"candidate_whole_outputs":["wrong"]}}',
            "character_ocr_models_json": '{"printed":{"candidate_outputs":["x"]}}',
        }


class MultiModelConstraintTests(unittest.TestCase):
    def test_attack_must_pass_for_every_development_model(self) -> None:
        rates = [
            exact_output_rate(["wrong", "wrong"], "target"),
            exact_output_rate(["target", "wrong"], "target"),
        ]
        passes = [
            ocr_goal_pass(
                rate,
                selection_goal="attack-both",
                max_exact_match_rate=0.0,
                min_exact_match_rate=1.0,
            )
            for rate in rates
        ]

        self.assertEqual(passes, [True, False])
        self.assertFalse(all(passes))

    def test_preservation_requires_every_variant(self) -> None:
        rate = exact_output_rate(["target", "target"], "target")

        self.assertTrue(
            ocr_goal_pass(
                rate,
                selection_goal="preserve-both",
                max_exact_match_rate=0.0,
                min_exact_match_rate=1.0,
            )
        )


class RandomOcrConfusableGenerationTests(unittest.TestCase):
    def test_ocr_only_mode_never_draws_from_identity_atlas(self) -> None:
        attack = pd.DataFrame(
            [
                {
                    "real_span": "a",
                    "candidate_span": "@",
                    "operation": "single_homoglyph",
                    "visual_similarity_score": 0.9,
                    "ocr_real_rate": 0.0,
                    "ocr_wrong_rate": 1.0,
                    "bucket": "safe_hard",
                    "legit_q25": 2.0,
                    "ocr_attack_rate": 1.0,
                    "ocr_attack_contexts": 4,
                }
            ]
        )
        identity = pd.DataFrame(
            [
                {
                    "real_span": "a",
                    "candidate_span": "α",
                    "operation": "visual_identity_homoglyph",
                    "visual_similarity_score": 0.99,
                    "ocr_real_rate": 1.0,
                    "ocr_wrong_rate": 0.0,
                    "bucket": "visual_identity",
                }
            ]
        )
        args = SimpleNamespace(
            generation_mode="ocr-confusable-only",
            min_substitutions=1,
            max_substitutions=1,
            min_identity_substitutions=0,
            max_identity_substitutions=0,
            min_ocr_confusable_substitutions=1,
        )
        generator = AtlasSpoofGenerator(attack, args, identity_atlas=identity)

        candidate, operations = generator._generate_once(
            "area",
            np.random.default_rng(7),
        )

        self.assertIn("@", candidate)
        self.assertEqual(
            {operation["substitution_family"] for operation in operations},
            {"ocr_confusable"},
        )

class ModelAwareCombinationTests(unittest.TestCase):
    def operation(
        self,
        *,
        start: int,
        real_span: str,
        candidate_span: str,
        family: str,
    ) -> dict[str, object]:
        return {
            "start": start,
            "end": start + len(real_span),
            "real_span": real_span,
            "candidate_span": candidate_span,
            "operation": "test",
            "visual_similarity_score": 0.9,
            "ocr_real_rate": 0.0,
            "ocr_wrong_rate": 1.0,
            "bucket": "safe_hard",
            "substitution_family": family,
        }

    def record(
        self,
        candidate: str,
        operations: list[dict[str, object]],
        *,
        printed_pass: bool,
        handwritten_pass: bool,
    ) -> dict[str, object]:
        return {
            "candidate": candidate,
            "operations": operations,
            "_selection_analysis": {
                "raw_score": 0.5,
                "models": {
                    "printed": {
                        "character_ocr_pass": printed_pass,
                        "character_exact_match_rate": 0.0 if printed_pass else 1.0,
                    },
                    "handwritten": {
                        "character_ocr_pass": handwritten_pass,
                        "character_exact_match_rate": 0.0 if handwritten_pass else 1.0,
                    },
                },
            },
        }

    def test_combines_complementary_non_overlapping_proposals(self) -> None:
        real_name = "abcdef"
        left_operations = [
            self.operation(start=0, real_span="a", candidate_span="@", family="visual_identity"),
            self.operation(start=1, real_span="b", candidate_span="8", family="ocr_confusable"),
            self.operation(start=2, real_span="c", candidate_span="(", family="ocr_confusable"),
        ]
        right_operations = [
            self.operation(start=3, real_span="d", candidate_span="|)", family="visual_identity"),
            self.operation(start=4, real_span="e", candidate_span="3", family="ocr_confusable"),
            self.operation(start=5, real_span="f", candidate_span="ph", family="ocr_confusable"),
        ]
        records = [
            self.record("@8(def", left_operations, printed_pass=True, handwritten_pass=False),
            self.record("abc|)3ph", right_operations, printed_pass=False, handwritten_pass=True),
        ]

        combined = generate_model_aware_operation_combinations(
            real_name=real_name,
            records=records,
            model_names=["printed", "handwritten"],
            limit=4,
            min_substitutions=3,
            max_substitutions=6,
            min_identity_substitutions=1,
            max_identity_substitutions=3,
            min_ocr_confusable_substitutions=1,
            max_text_ensemble_score=1.0,
            disallowed=set(),
        )

        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]["candidate"], "@8(|)3ph")
        self.assertEqual(
            combined[0]["model_aware_expected_model_coverage"],
            ["handwritten", "printed"],
        )

    def test_rejects_conflicting_operations_at_the_same_span(self) -> None:
        operations = [
            self.operation(start=0, real_span="a", candidate_span="@", family="visual_identity"),
            self.operation(start=0, real_span="a", candidate_span="4", family="ocr_confusable"),
        ]

        self.assertIsNone(merge_non_overlapping_operations("abc", operations))
        self.assertEqual(
            apply_operations("abc", [operations[0]]),
            "@bc",
        )


class CharacterwiseOcrTests(unittest.TestCase):
    def test_reuses_encoder_classification_for_repeated_characters(self) -> None:
        reader = object.__new__(TrOCRTextReader)
        rendered: list[tuple[str, int]] = []

        def render_text(char: str, **variation: int) -> tuple[str, int]:
            image = (char, int(variation.get("y_shift", 0)))
            rendered.append(image)
            return image

        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-"
        alphabet_index = {char: index for index, char in enumerate(alphabet)}

        def embed_images(
            images: list[tuple[str, int]], *, batch_size: int
        ) -> np.ndarray:
            self.assertEqual(batch_size, 8)
            matrix = np.zeros((len(images), len(alphabet)), dtype=np.float32)
            for row, (char, _shift) in enumerate(images):
                target = "a" if char == "α" else char
                matrix[row, alphabet_index[target]] = 1.0
            return matrix

        reader.render_text = render_text  # type: ignore[method-assign]
        reader.embed_images = embed_images  # type: ignore[method-assign]
        result = reader.recognize_characterwise(
            ["aαa", "αaa"],
            batch_size=8,
            variations=[{"y_shift": 0}, {"y_shift": 1}],
        )

        self.assertEqual(len(rendered), 2 * (len(alphabet) + 2))
        self.assertEqual(result["aαa"], ["aaa", "aaa"])
        self.assertEqual(result["αaa"], ["aaa", "aaa"])


class HoldoutTransferSummaryTests(unittest.TestCase):
    def test_conditions_attack_success_on_robust_clean_recovery(self) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "fraudulent_name": ["abc-candidate", "def-candidate"],
                "real_name": ["abc", "def"],
                "label": [1.0, 1.0],
            }
        )
        outputs = {
            "abc": ["abc", "abc"],
            "abc-candidate": ["wrong", "wrong"],
            "def": ["def", "wrong"],
            "def-candidate": ["wrong", "wrong"],
        }

        summary = summarize_strategy(frame, outputs, ["v0", "v1"])
        aggregate = summary["aggregate"]

        self.assertEqual(aggregate["eligible_clean_recovered_all_variants_rows"], 1)
        self.assertEqual(aggregate["candidate_failed_all_variants_conditioned_rate"], 1.0)
        self.assertEqual(aggregate["candidate_preserved_all_variants_conditioned_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
