"""CPU-only synthetic promotion-gate tests; no GPU or real model data."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from Erratum.v14a14_blind_panel import build_holdout
from Erratum.v14a14_promotion_gate import prepare, assess


class PromotionGateContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        self.review = self.root / "blind_human_review"
        cases, _ = build_holdout()
        self.cases = cases
        for arm, high_rep, eos in (("original_80", 3, 0.05), ("causal_80", 1, 0.90)):
            d = self.root / arm / "explicit_eos"
            d.mkdir(parents=True)
            generations = [
                {
                    "id": case["id"],
                    "semantic_family": case["semantic_family"],
                    "generation": f"Synthetic {arm} {case['id']} answer.",
                }
                for case in cases
            ]
            (d / "gradient_clipping_outputs_u0040.json").write_text(
                json.dumps(generations), encoding="utf-8"
            )
            (d / "generalization_diagnostic_u0040.json").write_text(
                json.dumps({
                    "update": 40,
                    "first_sentence_and_generation_repetition": {
                        "high_repetition_count": high_rep,
                    },
                    "eos_eot_probability_at_reference_answer_boundary": {
                        "mean_p_either_termination_after_reference": eos,
                    },
                    "holdout_reference_teacher_forcing": {
                        "reference_continuation_nll": 3.0,
                        "reference_continuation_top1": 0.25,
                    },
                }), encoding="utf-8"
            )
        self.candidate = self.root / "causal_80/candidate/v14a14_causal_u0040_candidate.pt"
        self.candidate.parent.mkdir(parents=True)
        self.candidate.write_bytes(b"SYNTHETIC_NOT_A_REAL_MODEL")
        size = self.candidate.stat().st_size
        sha = hashlib.sha256(self.candidate.read_bytes()).hexdigest()
        (self.root / "v14a14_posttraining_diagnostic.json").write_text(
            json.dumps({
                "matched_batch_order": True,
                "candidate_retained": True,
                "promotion_attempted": False,
                "arms": {
                    "original_80": {},
                    "causal_80": {
                        "candidate_checkpoint": {
                            "path": str(self.candidate),
                            "sha256": sha,
                            "size_bytes": size,
                        }
                    },
                },
            }), encoding="utf-8"
        )

    def _scores(self, passing: bool) -> Path:
        prepare(self.root, self.review)
        mapping = json.loads((self.review / "sealed_arm_mapping.json").read_text())["mapping"]
        grades = [
            {
                "blind_id": p["blind_id"],
                "verdict": ("adequate" if passing and p["arm"] == "causal_80" else "inadequate"),
                "reason": "" if passing and p["arm"] == "causal_80" else "Lacks causal mechanism",
            }
            for p in mapping
        ]
        path = self.root / "adjudication.json"
        path.write_text(json.dumps({"reviews": grades}), encoding="utf-8")
        return path

    def test_passing_factual_and_eos_gates_permit_research_promotion(self):
        score_path = self._scores(True)
        report = assess(self.root, self.review, score_path, apply=True)
        self.assertEqual(report["status"], "GATES_PASS_PROMOTION_ELIGIBLE")
        self.assertTrue(all(report["gate_results"].values()))
        promoted = self.root / "promoted/v14a14_causal_u0040_promoted.pt"
        self.assertEqual(promoted.read_bytes(), self.candidate.read_bytes())
        self.assertEqual(promoted.with_suffix(".decision.json").is_file(), True)
        self.assertEqual(report["promotion_scope"],
                         "research_checkpoint_not_canonical_parent_replacement")

    def test_unsuccessful_human_judgments_block_promotion(self):
        score_path = self._scores(False)
        with self.assertRaises(RuntimeError):
            assess(self.root, self.review, score_path, apply=True)
        self.assertFalse((self.root / "promoted").exists())

    def test_changed_generation_invalidates_prior_blind_packet(self):
        score_path = self._scores(True)
        p = self.root / "causal_80/explicit_eos/gradient_clipping_outputs_u0040.json"
        items = json.loads(p.read_text())
        items[0]["generation"] = "TAMPERED"
        p.write_text(json.dumps(items))
        with self.assertRaises(RuntimeError):
            assess(self.root, self.review, score_path, apply=True)


if __name__ == "__main__":
    unittest.main()
