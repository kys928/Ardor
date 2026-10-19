"""CPU-only contract tests for v14a13 factual audit and adjudication."""
import unittest

from Erratum.v14a13_supervision_audit import (
    CONTROL_DIGEST, CORRECTED_DIGEST, build_report, rebuild, digest,
)
from Erratum.v14a13_proposition_evaluator import applicable, evaluate


class SupervisionAuditTest(unittest.TestCase):
    def test_exact_frozen_inputs_and_targets(self):
        control, corrected = rebuild()
        self.assertEqual(len(control), 28)
        self.assertEqual(len(corrected), 28)
        self.assertEqual(digest(control), CONTROL_DIGEST)
        self.assertEqual(digest(corrected), CORRECTED_DIGEST)
        self.assertEqual([r["id"] for r in control], [r["id"] for r in corrected])
        self.assertEqual(
            [r["prompt"] for r in control],
            [r["prompt"] for r in corrected],
        )

    def test_duplicate_and_mixed_learning_rate_prompts_identified(self):
        report = build_report()
        self.assertEqual(report["case_count"], 28)
        cases = {c["id"][-4:]: c for c in report["cases"]}
        for identifier in ("0152", "0376"):
            self.assertIn("mixed_learning_rate_and_collision_task",
                          cases[identifier]["corrected_flags"])
            self.assertTrue(any(x.startswith("duplicate_prompt:")
                                for x in cases[identifier]["corrected_flags"]))
        self.assertTrue(report["automatic_flags_are_not_factual_scores"])

    def test_false_premise_and_lr_propositions_are_required(self):
        self.assertIn("explicitly_reject_false_premise",
                      applicable({"family": "false_premise_correction"}) )
        self.assertIn("mechanisms_distinguished",
                      applicable({"family": "learning_rate_comparison"}) )
        self.assertIn("contrasted_collision", applicable({
            "family": "learning_rate_comparison",
            "question_class": "collision_comparison",
        }))


class PropositionScoringTest(unittest.TestCase):
    def setUp(self):
        self.case = {"id": "holdout_001", "family": "learning_rate_comparison",
                     "question_class": "normal"}
        self.response = (
            "Clipping rescales large gradients above a threshold. "
            "Learning rate scales the optimizer step generally. "
            "These controls are different."
        )
        self.labels = {
            key: {"status": "supported", "evidence": self.response}
            for key in applicable(self.case)
        }

    def annotation(self):
        return {"case_id": self.case["id"], "reviewer": "reviewer-A",
                "response": self.response, "propositions": dict(self.labels),
                "contradictions": []}

    def test_supported_annotated_propositions(self):
        # The scores reflect an explicit reviewer annotation, not keyword matches.
        self.assertTrue(evaluate(self.case, self.annotation())["adequate"])

    def test_missing_proposition_blocks_pass(self):
        a = self.annotation()
        a["propositions"]["mechanisms_distinguished"] = {
            "status": "missing", "evidence": "",
        }
        self.assertFalse(evaluate(self.case, a)["adequate"])

    def test_contradiction_overrides_complete_supported_list(self):
        a = self.annotation()
        a["contradictions"] = [{
            "evidence": "Learning rate scales the optimizer step generally.",
            "description": "Test reviewer records a contradiction",
        }]
        self.assertFalse(evaluate(self.case, a)["adequate"])

    def test_fabricated_evidence_rejected(self):
        a = self.annotation()
        a["propositions"]["mechanisms_distinguished"] = {
            "status": "supported", "evidence": "This exact sentence is absent",
        }
        with self.assertRaises(ValueError):
            evaluate(self.case, a)

    def test_false_premise_requires_explicit_rejection(self):
        case = {"id": "false_01", "family": "false_premise_correction"}
        response = "Large gradients harm training. Gradient clipping can bound them."
        claims = {k: {"status": "supported", "evidence": response}
                  for k in applicable(case)}
        claims["explicitly_reject_false_premise"] = {
            "status": "missing", "evidence": "",
        }
        result = evaluate(case, {
            "case_id": case["id"], "reviewer": "reviewer-B",
            "response": response, "propositions": claims, "contradictions": [],
        })
        self.assertFalse(result["adequate"])


if __name__ == "__main__":
    unittest.main()
