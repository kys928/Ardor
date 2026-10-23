"""v14a14 CPU-only contract: no model, checkpoint, tokenizer or GPU loaded."""
import unittest
from collections import Counter

from Erratum.v14a14_signal_dataset import (
    build_pairs, FAMILIES_TARGET, normalize, select_rows, digest,
)
from Erratum.v14a14_blind_panel import build_holdout
from Erratum.v14a13_proposition_evaluator import PROPOSITIONS, applicable


class NewSupervisionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original, cls.revised, cls.audit = build_pairs()
        cls.holdout, cls.holdout_manifest = build_holdout()

    def test_expanded_counts_and_targeted_emphasis(self):
        self.assertEqual(len(self.revised), 80)
        self.assertEqual(dict(Counter(r["semantic_family"] for r in self.revised)), FAMILIES_TARGET)
        for family in ("backprop_stability", "learning_rate_comparison",
                       "false_premise_correction"):
            self.assertEqual(FAMILIES_TARGET[family], 16)

    def test_matching_inputs_differ_only_in_chosen(self):
        self.assertEqual(len(self.original), len(self.revised))
        for old, new in zip(self.original, self.revised):
            self.assertEqual({k: v for k, v in old.items() if k != "chosen"},
                             {k: v for k, v in new.items() if k != "chosen"})
            self.assertNotEqual(old["chosen"], new["chosen"])
            self.assertTrue(new["prompt"].endswith("\n-"))
        self.assertEqual(self.audit["manifest"]["original_control_paired_prompt_sha256"],
                         self.audit["manifest"]["revised_paired_prompt_sha256"])

    def test_all_prompts_unique_and_not_frozen_old28(self):
        self.assertEqual(len({normalize(r["prompt"]) for r in self.revised}), 80)
        self.assertEqual(len(set(r["id"] for r in self.revised)), 80)
        self.assertEqual(self.audit["manifest"]["distinct_prompts"], 80)

    def test_proposition_evidence_for_every_revised_row(self):
        for row, ev in zip(self.revised, self.audit["evidence"]):
            self.assertEqual(row["id"], ev["id"])
            self.assertEqual(row["semantic_family"], ev["family"])
            self.assertEqual(set(PROPOSITIONS[row["semantic_family"]]),
                             {p["proposition_id"] for p in ev["requirements"]})
            for fact in ev["requirements"]:
                self.assertEqual(fact["status"], "supported")
                self.assertIn(fact["evidence"], row["chosen"])
                self.assertGreater(len(fact["evidence"]), 30)
            self.assertIn(ev["contextual_application"], row["chosen"])
            self.assertEqual(ev["approval"], "PENDING_HUMAN_REVIEW")
        self.assertFalse(self.audit["manifest"]["training_eligible"])

    def test_holdout_is_independent_in_prompt_text(self):
        self.assertEqual(len(self.holdout), 28)
        p = {normalize(r["prompt"]) for r in self.revised}
        h = {normalize(r["prompt"]) for r in self.holdout}
        self.assertEqual(len(h), 28)
        self.assertTrue(p.isdisjoint(h))
        self.assertEqual(len(set(r["id"] for r in self.holdout)), 28)
        self.assertEqual(self.holdout_manifest["holdout_rows_and_references_sha256"],
                         digest(self.holdout))
        self.assertEqual(dict(Counter(r["semantic_family"] for r in self.holdout)),
                         {family: 4 for family in FAMILIES_TARGET})

    def test_holdout_has_required_prop_keys_and_distinct_refs(self):
        control_answers = {normalize(r["chosen"]) for r in self.original}
        intervention_answers = {normalize(r["chosen"]) for r in self.revised}
        for case in self.holdout:
            self.assertTrue(case["prompt"].endswith("\n-"))
            self.assertTrue(case["chosen"].strip())
            self.assertEqual(set(case["required_propositions"]),
                             set(applicable({"family": case["semantic_family"],
                                             "question_class": case["question_class"]})))
            self.assertNotIn(normalize(case["chosen"]), control_answers | intervention_answers)
        self.assertEqual(sum(r["question_class"] == "collision_comparison"
                             for r in self.holdout), 1)

    def test_branches_are_diagnostics_only(self):
        self.assertEqual(self.audit["manifest"]["status"], "DRAFT_NOT_TRAINING_APPROVED")
        self.assertTrue(self.holdout_manifest["not_seen_by_model_or_optimizer"])
        self.assertEqual(self.holdout_manifest["train_prompt_sha256"],
                         self.audit["manifest"]["revised_paired_prompt_sha256"])


if __name__ == "__main__":
    unittest.main()
