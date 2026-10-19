"""No-GPU v14a13 curated supervision and source-grounded rubric challenges."""
from copy import deepcopy
import unittest

from Erratum.v14a13_supervision_audit import rebuild, digest
from Erratum.v14a13_curated_supervision import build_curated, OVERRIDES
from Erratum.v14a13_proposition_evaluator import (
    evaluate, compare_independent_reviews, applicable,
)
from Erratum.v14a13_rubric_challenges import (
    CHALLENGES, annotations_for_challenge,
)


class CuratedDataTest(unittest.TestCase):
    def test_corrected_candidate_is_distinct_and_frozen_unchanged(self):
        original, prior = rebuild()
        prior_digest = digest(prior)
        rows, report = build_curated()
        self.assertEqual(len(rows), 28)
        self.assertEqual(report["unique_prompts_before"], 24)
        self.assertEqual(report["unique_prompts_after"], 28)
        self.assertEqual(len(report["edited_rows"]), 5)
        self.assertEqual(set(OVERRIDES), {x["id"] for x in report["edited_rows"]})
        self.assertEqual(digest(prior), prior_digest)
        self.assertEqual([x["id"] for x in prior], [x["id"] for x in rows])
        self.assertEqual([x["id"] for x in original], [x["id"] for x in rows])
        for frozen, candidate in zip(prior, rows):
            if candidate["id"] not in OVERRIDES:
                self.assertEqual(frozen, candidate)
            else:
                self.assertEqual(candidate["source"], "behavior")
                self.assertEqual(candidate["route"], "gradient_clipping")
                self.assertEqual(candidate["semantic_family"], frozen["semantic_family"])

    def test_question_and_answer_alignment_in_fixed_pairs(self):
        rows, report = build_curated()
        chosen = {x["id"][-4:]: x for x in rows}
        self.assertEqual(len([r for r in rows if r["collision_route"]]), 1)
        for idx in ("0152", "0376"):
            row = chosen[idx]
            self.assertIsNone(row["collision_route"])
            self.assertEqual(row["question_class"], "normal")
            self.assertNotIn("checkpoint", row["prompt"].lower())
            self.assertNotIn("checkpoint", row["chosen"].lower())
            self.assertIn("learning rate", row["prompt"].lower())
            self.assertIn("learning rate", row["chosen"].lower())
            self.assertIn("clip", row["chosen"].lower())
        self.assertIn("unchanged", chosen["0086"]["chosen"])
        self.assertIn("No.", chosen["0086"]["chosen"])
        self.assertIn("premise is false", chosen["0391"]["chosen"])
        self.assertFalse(any("duplicate_prompt:" in flag
                             for flags in report["triage_flags_after"].values()
                             for flag in flags))
        self.assertFalse(any("mixed_learning_rate_and_collision_task" in flag
                             for flags in report["triage_flags_after"].values()
                             for flag in flags))


class ReferenceGroundedChallengeTest(unittest.TestCase):
    def test_all_hand_authored_golds_match_strict_policy(self):
        self.assertEqual(len(CHALLENGES), 16)
        family_set = {x["family"] for x in CHALLENGES}
        self.assertEqual(len(family_set), 7)
        truths = [x["gold"] for x in CHALLENGES]
        self.assertEqual(truths.count(True), 8)
        self.assertEqual(truths.count(False), 8)
        seen = set()
        for c in CHALLENGES:
            self.assertNotIn(c["id"], seen)
            seen.add(c["id"])
            self.assertTrue(c["reason"])
            case, annotation = annotations_for_challenge(c)
            self.assertEqual(set(applicable(case)), set(annotation["propositions"]))
            result = evaluate(case, annotation)
            self.assertEqual(result["adequate"], c["gold"],
                             msg=f"Gold mismatch for {c['id']}: {c['reason']}")

    def test_contradictory_extra_claim_overrides_positive_propositions(self):
        c = next(x for x in CHALLENGES if x["id"] == "optimizer_overclaim")
        case, annotation = annotations_for_challenge(c)
        self.assertTrue(all(x["status"] == "supported"
                            for x in annotation["propositions"].values()))
        self.assertFalse(evaluate(case, annotation)["adequate"])

    def test_wrong_timing_and_false_lr_equivalence_rejected(self):
        for case_id in ("backprop_wrong_timing", "lr_wrong_equivalence",
                        "norm_false_target", "premise_agrees"):
            item = next(x for x in CHALLENGES if x["id"] == case_id)
            case, annotation = annotations_for_challenge(item)
            self.assertFalse(evaluate(case, annotation)["adequate"])

    def test_two_reviewers_required_and_disagreements_held(self):
        item = next(x for x in CHALLENGES if x["id"] == "lr_good")
        case, annotation = annotations_for_challenge(item)
        second = deepcopy(annotation)
        second["reviewer"] = "separate-reviewer-2"
        paired = compare_independent_reviews(case, [annotation, second])
        self.assertTrue(paired["resolved"])
        self.assertTrue(paired["adequate"])
        self.assertTrue(paired["human_independence_unverified"])
        with self.assertRaises(ValueError):
            compare_independent_reviews(case, [annotation, annotation])
        second["propositions"]["mechanisms_distinguished"] = {
            "status": "unclear", "evidence": "",
        }
        paired = compare_independent_reviews(case, [annotation, second])
        self.assertFalse(paired["resolved"])
        self.assertFalse(paired["adequate"])
        self.assertEqual(paired["proposition_disagreements"],
                         ["mechanisms_distinguished"])
        self.assertTrue(paired["human_adjudication_required"])

    def test_one_reviewer_cannot_certify_truth(self):
        # An explicit limitation: a dishonest or mistaken reviewer can mark
        # incorrect content supported. This evaluator validates annotations,
        # NOT semantic truth or independent reviewer competence.
        bad = next(x for x in CHALLENGES if x["id"] == "lr_wrong_equivalence")
        case, annotation = annotations_for_challenge(bad)
        self.assertFalse(evaluate(case, annotation)["adequate"])
        for name in annotation["propositions"]:
            annotation["propositions"][name] = {
                "status": "supported", "evidence": bad["response"],
            }
        self.assertTrue(evaluate(case, annotation)["adequate"])


if __name__ == "__main__":
    unittest.main()
