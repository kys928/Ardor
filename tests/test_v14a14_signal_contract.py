"""v14a14 CPU-only contract: no model, checkpoint, tokenizer or GPU loaded."""
import unittest
import tempfile
import json
from pathlib import Path
from collections import Counter

from Erratum.v14a14_signal_dataset import (
    build_pairs, FAMILIES_TARGET, normalize, select_rows, digest,
    REVIEW_ROUND_1_INADEQUATE, REVIEW_ROUND_1_INADEQUATE_CASES,
    review_only_amended_ten, review_round_3_seven, ROUND_3_TARGETED_REWRITES,
)
from Erratum.v14a14_blind_panel import build_holdout
from Erratum.v14a14_controlled_eos_train import PIN, preflight, verify_data_review
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

    def test_pinned_preflight_requires_exact_input_hashes(self):
        old, new, panel, pins = preflight()
        self.assertEqual(pins, PIN)
        self.assertEqual(len(old), 80)
        self.assertEqual(len(new), 80)
        self.assertEqual(len(panel), 28)

    def test_unsigned_attestation_blocks_training(self):
        ids = [r["id"] for r in self.revised]
        doc = {
            "proposition_audit_sha256": PIN["proposition_audit_sha256"],
            "train_prompt_sha256": PIN["train_prompt_sha256"],
            "reviewers": [
                {"reviewer_id": "A", "status": "pending", "approved_row_ids": ids},
                {"reviewer_id": "B", "status": "approved_after_review",
                 "approved_row_ids": ids}
            ],
            "resolved_disagreements": True,
        }
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "review.json"
            p.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_data_review(p, ids)
            doc["reviewers"][0]["status"] = "approved_after_review"
            doc["reviewers"][1]["contradictions_found_unresolved"] = True
            p.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_data_review(p, ids)
            doc["reviewers"][1]["contradictions_found_unresolved"] = False
            doc["reviewers"][1]["reviewer_id"] = "A"
            p.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_data_review(p, ids)

    def test_round1_repair_keeps_all_70_accepted_unchanged(self):
        self.assertEqual(len(REVIEW_ROUND_1_INADEQUATE), 10)
        self.assertEqual(tuple(sorted(REVIEW_ROUND_1_INADEQUATE_CASES)),
                         tuple(sorted([9, 14, 15, 24, 33, 35, 36, 44, 45, 47])))
        manifest = self.audit["manifest"]
        status = manifest["review_round_1_summary"]
        self.assertEqual(status["original_cases"], 80)
        self.assertEqual(status["adequate_original"], 70)
        self.assertEqual(status["inadequate_original"], 10)
        self.assertEqual(status["untouched_adequate_text_count"], 70)
        self.assertEqual(status["amended_case_count"], 10)
        self.assertEqual(status["previous_80_causal_sha256"],
                         "4b3afb700aa1cd7f880010949860a33aa127842db2148b2b1d6060accb70e3a9")
        self.assertEqual(sum(r["review_round_1_repair"] is not None
                             for r in self.audit["evidence"]), 10)
        for r, e in zip(self.revised, self.audit["evidence"]):
            self.assertEqual(e["review_round_1_case_number"],
                             self.revised.index(r) + 1)
            self.assertEqual(e["review_round_2_status"], "PENDING_REVIEW")
        # Exact pre-adjudication hash is rebuilt and verified by build_pairs().
        packet = review_only_amended_ten(self.revised, self.audit["evidence"])
        self.assertEqual(packet.count("**Original round-one inadequate answer**"), 10)
        self.assertEqual(packet.count("**Round-two submitted answer (historical, frozen)**"), 10)
        self.assertIn("Case 45", packet)
        self.assertIn("gradient accumulation", packet.lower())
        round3 = review_round_3_seven(self.revised, self.audit["evidence"])
        self.assertEqual(round3.count("**Newly proposed answer (round three)**"), 7)
        self.assertEqual(round3.count("**Previously reviewed answer (round two)**"), 7)
        self.assertEqual(set(ROUND_3_TARGETED_REWRITES),
                         {e["id"] for e in self.audit["evidence"]
                          if e.get("review_round_3_status") == "HUMAN_ROUND_3_REVIEWED"})
        self.assertEqual(self.audit["manifest"]["review_round_2_summary"]["adequate"], 9)
        self.assertEqual(self.audit["manifest"]["review_round_2_summary"]["inadequate"], 1)
        self.assertEqual(self.audit["manifest"]["review_round_2_summary"]
                         ["round_2_causal_sha256"],
                         "80d7716b29d06469d7bf4dc3ef91cbb4a0ff69490d93bb9845888c1d21451ab3")
        self.assertEqual(self.audit["manifest"]["review_round_2_summary"]
                         ["round_2_evidence_sha256"],
                         "9682546efc3b561845a71d858455434d7f8389081221a6e38dfb123aa7319899")
        by_id = {r["id"]: r for r in self.revised}
        case15 = by_id["v14a4_train_gradient_clipping_0092"]["chosen"]
        self.assertIn("Δθ = −η∇θL", case15)
        self.assertIn("fixed learning rate", case15)
        case47 = by_id["v14a14_train_gradient_clipping_backprop_stability_03"]["chosen"]
        self.assertIn("gradient accumulation", case47.lower())
        self.assertIn("microbatch", case47)
        case45 = by_id["v14a14_train_gradient_clipping_backprop_stability_01"]["chosen"]
        self.assertIn("cannot retroactively repair", case45)
        for case in (33,35,36,44):
            pair = next((r,e) for r,e in zip(self.revised,self.audit["evidence"])
                        if e["review_round_1_case_number"] == case)
            self.assertTrue(pair[0]["chosen"].startswith("Gradient clipping"))


    def test_first_user_review_covers_all_current_answer_hashes(self):
        import hashlib
        source = self.audit["manifest"]["first_user_review_current"]
        self.assertEqual(source["adequate"], 80)
        self.assertEqual(source["inadequate"], 0)
        self.assertEqual(source["uncertain"], 0)
        self.assertEqual(source["reviewer_count"], 1)
        self.assertEqual(source["second_independent_reviewer_status"], "NOT_PERFORMED")
        self.assertFalse(source["training_eligible"])
        self.assertEqual(source["case_35_latest_status"], "USER_ADJUDICATED_ADEQUATE")
        record = source["review_ledger"]
        self.assertEqual(len(record), 80)
        self.assertEqual(len({r["id"] for r in record}), 80)
        self.assertEqual(source["review_ledger_sha256"], digest(record))
        for row, ev, item in zip(self.revised, self.audit["evidence"], record):
            self.assertEqual(item["id"], row["id"])
            self.assertEqual(item["case_number"], ev["review_round_1_case_number"])
            self.assertEqual(item["verdict"], "ADEQUATE")
            self.assertEqual(item["answer_sha256"],
                             hashlib.sha256(row["chosen"].encode("utf-8")).hexdigest())
            self.assertEqual(ev["first_user_review_current_answer_verdict"], "ADEQUATE")
        case35 = next(e for e in self.audit["evidence"]
                      if e["review_round_1_case_number"] == 35)
        self.assertEqual(case35["review_round_4_status"], "USER_ADJUDICATED_ADEQUATE")
        self.assertEqual(case35["first_user_review_source_round"], "round_4_user_verdict")

    def test_branches_are_diagnostics_only(self):
        self.assertEqual(self.audit["manifest"]["status"], "DRAFT_NOT_TRAINING_APPROVED")
        self.assertTrue(self.holdout_manifest["not_seen_by_model_or_optimizer"])
        self.assertEqual(self.holdout_manifest["train_prompt_sha256"],
                         self.audit["manifest"]["revised_paired_prompt_sha256"])


if __name__ == "__main__":
    unittest.main()
