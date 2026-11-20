"""Small synthetic tests; no CUB data, downloaded weights, or training required."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "codebook_train"))
from src.models.proto_quantnet import ProtoQuantNet
from physical_pruning_lib import DeletionEvaluator, compact_model, p1_ranking, prune, spatial_logits
from pilot_physical_pruning import balanced_indices, compare_cached_features, load_cache, run_pilots


torch.set_num_threads(1)
DEVICE = torch.device(os.environ.get("PILOT_TEST_DEVICE", "cpu"))


def fixture(codes=12):
    torch.manual_seed(123)
    model = ProtoQuantNet(nn.Identity(), torch.randn(codes, 8), 3, train_codebook=True).to(DEVICE).eval()
    with torch.no_grad():
        model.classifier.weight.copy_(torch.randn(3, codes) * 0.5 + 0.7)
    features = torch.randn(12, 8, 2, 2)
    labels = torch.arange(12) % 3
    return model, features, labels


class PhysicalPruningTests(unittest.TestCase):
    def test_single_candidates_match_actual_compact_forward(self):
        model, features, labels = fixture()
        engine = DeletionEvaluator(spatial_logits(model, features, 4), model.classifier.weight, labels,
                                   image_batch_size=3, candidate_chunk_size=2)
        losses = engine.candidate_losses(list(range(model.num_prototypes)))
        for removed in range(model.num_prototypes):
            retained = torch.tensor([i for i in range(model.num_prototypes) if i != removed])
            actual = compact_model(model, retained)(features.to(DEVICE)).logits
            expected = F.cross_entropy(actual, labels.to(DEVICE))
            self.assertAlmostEqual(float(losses[removed]), float(expected.detach()), places=5)

    def test_multi_deletion_and_sequential_updates_match(self):
        model, features, labels = fixture()
        engine = DeletionEvaluator(spatial_logits(model, features, 4), model.classifier.weight, labels)
        joint = engine.loss_after([1, 3, 6])
        engine.remove([1, 3, 6])
        actual = compact_model(model, engine.ids)(features.to(DEVICE)).logits
        torch.testing.assert_close(actual, engine.predictions, rtol=1e-5, atol=2e-6)
        self.assertAlmostEqual(joint, engine.loss, places=5)
        engine.remove([0, 2])
        actual = compact_model(model, engine.ids)(features.to(DEVICE)).logits
        torch.testing.assert_close(actual, engine.predictions, rtol=1e-5, atol=2e-6)

    def test_spatial_maximum_can_move(self):
        probabilities = torch.tensor([[[0.40, 0.01, 0.59], [0.35, 0.60, 0.05]]], device=DEVICE)
        engine = DeletionEvaluator(probabilities.log(), torch.tensor([[1., 0., 0.], [0., 0., 1.]], device=DEVICE),
                                   torch.tensor([0]))
        self.assertAlmostEqual(float(engine.probabilities.amax(1)[0, 0]), 0.4, places=5)
        engine.remove([1])
        self.assertAlmostEqual(float(engine.probabilities.amax(1)[0, 0]), 0.875, places=5)

    def test_near_one_probability_uses_stable_path(self):
        logits = torch.tensor([[[80., 0., -2.]]], device=DEVICE)
        engine = DeletionEvaluator(logits, torch.tensor([[0., 1., 0.], [0., 0., 1.]], device=DEVICE), torch.tensor([0]))
        candidate = float(engine.candidate_losses([0])[0])
        self.assertAlmostEqual(candidate, engine.loss_after([0]), places=6)

    def test_zero_classifier_column_still_affects_normalization(self):
        logits = torch.tensor([[[0., 1., 0.]]], device=DEVICE)
        engine = DeletionEvaluator(logits, torch.tensor([[1., 0., 0.], [0., 0., 2.]], device=DEVICE), torch.tensor([0]))
        before = engine.predictions.clone()
        engine.remove([1])
        self.assertFalse(torch.allclose(before, engine.predictions))

    def test_compaction_preserves_raw_weights_buffer_bias_and_source(self):
        model, features, _ = fixture()
        model = ProtoQuantNet(nn.Identity(), model.codes.detach().clone(), 3, bias=True).to(DEVICE).eval()
        with torch.no_grad():
            model.classifier.weight[0, 0] = -2
            model.classifier.bias.copy_(torch.tensor([0.1, -0.2, 0.3], device=DEVICE))
        state = copy.deepcopy(model.state_dict())
        retained = torch.tensor([0, 2, 5], device=DEVICE)
        compact = compact_model(model, retained)
        self.assertNotIsInstance(compact.codes, nn.Parameter)
        torch.testing.assert_close(compact.classifier.weight, model.classifier.weight[:, retained])
        torch.testing.assert_close(compact.classifier.bias, model.classifier.bias)
        self.assertEqual(float(compact.classifier.weight[0, 0].detach()), -2)
        for key, value in state.items():
            torch.testing.assert_close(model.state_dict()[key], value)
        self.assertEqual(model.num_prototypes, 12)
        self.assertTrue(torch.isfinite(compact(features.to(DEVICE)).logits).all())

    def test_exact_counts_and_permutation_checks_for_all_methods(self):
        model, features, labels = fixture()
        _, ranking = p1_ranking(model, features, labels, 4)
        for method in ("p1", "greedy", "batched"):
            engine = DeletionEvaluator(spatial_logits(model, features, 4), model.classifier.weight, labels)
            events = []
            result = prune(engine, ranking, 5, method, 6, 3, record=events.append)
            self.assertEqual(len(engine.ids), 7)
            self.assertEqual(sum(result["accepted_batch_sizes"]), 5)
            self.assertTrue(all(event["remaining_codes"] >= 7 for event in events if event["event"] == "accepted"))
        engine = DeletionEvaluator(spatial_logits(model, features, 4), model.classifier.weight, labels)
        with self.assertRaises(ValueError):
            prune(engine, ranking[:-1], 2, "greedy")
        with self.assertRaises(ValueError):
            prune(engine, ranking, 2, "batched", extra_loss_tolerance=-1)

    def test_interaction_fallback_is_logged(self):
        model, features, labels = fixture()
        engine = DeletionEvaluator(spatial_logits(model, features, 4), model.classifier.weight, labels)
        original = engine.loss_after
        def expensive_joint(indices):
            return original(indices) + (2.0 if len(indices) > 1 else 0.0)
        engine.loss_after = expensive_joint
        events = []
        result = prune(engine, torch.arange(12), 4, "batched", 6, 4, 0.005, 0.25, events.append)
        fallbacks = [event for event in events if event.get("reason") == "interaction_damage_exceeds_tolerance"]
        self.assertGreater(result["fallback_count"], 0)
        self.assertTrue(all(event["fallback_to"] < event["proposed_size"] for event in fallbacks))
        self.assertTrue(all("allowed_additional_damage" in event for event in fallbacks))

    def test_meaningful_tolerance_can_accept_multiple_deletions(self):
        model, features, labels = fixture()
        engine = DeletionEvaluator(spatial_logits(model, features, 4), model.classifier.weight, labels)
        result = prune(engine, torch.arange(12), 4, "batched", 6, 4, extra_loss_tolerance=10.0)
        self.assertEqual(result["accepted_batch_sizes"], [4])
        self.assertEqual(result["fallback_count"], 0)

    def test_cache_check_accepts_two_small_tail_errors(self):
        expected = torch.full((1, 768, 7, 7), 0.25, device=DEVICE)
        actual = expected + 0.0002
        actual.flatten()[:2] += 0.004
        result = compare_cached_features(actual, expected)
        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "bounded_numerical_drift")
        self.assertEqual(result["failing_values"], 2)
        self.assertEqual(result["total_values"], 37632)
        self.assertGreater(result["cosine_similarity"], 0.99999)

    def test_cache_check_preserves_strict_fp16_and_zero_matches(self):
        torch.manual_seed(2)
        actual = torch.randn(1, 768, 7, 7, device=DEVICE)
        for fresh, cached in ((actual, actual.half().float()), (torch.zeros_like(actual), torch.zeros_like(actual))):
            result = compare_cached_features(fresh, cached)
            self.assertTrue(result["passed"])
            self.assertEqual(result["reason"], "strict_match")

    def test_cache_check_rejects_large_isolated_errors(self):
        expected = torch.full((1, 768, 7, 7), 0.25, device=DEVICE)
        actual = expected.clone()
        actual.flatten()[0] += 0.05
        result = compare_cached_features(actual, expected)
        self.assertFalse(result["passed"])
        self.assertGreater(result["max_scaled_error"], 2)

    def test_cache_check_rejects_widespread_small_errors(self):
        expected = torch.full((1, 768, 7, 7), 0.25, device=DEVICE)
        result = compare_cached_features(expected + 0.004, expected)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failing_values"], 37632)

    def test_cache_check_rejects_direction_change_even_with_few_outliers(self):
        expected = torch.zeros(1, 768, 7, 7, device=DEVICE)
        expected.flatten()[0] = 0.001
        actual = expected.clone()
        actual.flatten()[1] = 0.004
        result = compare_cached_features(actual, expected)
        self.assertEqual(result["failing_values"], 1)
        self.assertLess(result["max_scaled_error"], 2)
        self.assertLess(result["cosine_similarity"], 0.99999)
        self.assertFalse(result["passed"])

    def test_cache_check_rejects_nonfinite_and_malformed_features(self):
        expected = torch.ones(1, 768, 7, 7, device=DEVICE)
        for value in (float("nan"), float("inf")):
            actual = expected.clone()
            actual.flatten()[0] = value
            result = compare_cached_features(actual, expected)
            self.assertFalse(result["passed"])
            self.assertEqual(result["reason"], "nonfinite_features")
        for actual in (expected.flatten(), expected[:, :0]):
            with self.assertRaises(ValueError):
                compare_cached_features(actual, expected)

    def test_balanced_training_subset(self):
        labels = torch.arange(12) % 3
        first = balanced_indices(labels, 3, 2, 40)
        self.assertTrue(torch.equal(first, balanced_indices(labels, 3, 2, 40)))
        self.assertEqual(torch.bincount(labels[first]).tolist(), [2, 2, 2])
        with self.assertRaises(ValueError):
            balanced_indices(labels, 3, 5, 40)

    def test_missing_cache_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "Missing train feature cache"):
            load_cache(Path("missing_physical_pilot_cache.pt"), None, "train", "unused")

    def test_wrong_cache_split_hash_and_image_ids_are_rejected(self):
        class Dataset:
            img_ids = [7]
            labels = [0]
            def __len__(self):
                return 1
        valid = {"features": torch.zeros(1, 768, 7, 7).half(), "labels": torch.tensor([0]),
                 "img_ids": torch.tensor([7]), "split": "train", "backbone_sha256": "matching"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.pt"
            torch.save(valid, path)
            self.assertEqual(len(load_cache(path, Dataset(), "train", "matching")["labels"]), 1)
            for changed in ({"split": "val"}, {"backbone_sha256": "other"}, {"img_ids": torch.tensor([8])}):
                torch.save({**valid, **changed}, path)
                with self.assertRaises(ValueError):
                    load_cache(path, Dataset(), "train", "matching")

    def test_candidate_chunking_preserves_deletion_losses(self):
        model, features, labels = fixture()
        logits = spatial_logits(model, features, 4)
        small = DeletionEvaluator(logits, model.classifier.weight, labels, image_batch_size=1, candidate_chunk_size=1)
        large = DeletionEvaluator(logits, model.classifier.weight, labels, image_batch_size=5, candidate_chunk_size=4)
        candidates = [0, 2, 5, 8, 11]
        torch.testing.assert_close(small.candidate_losses(candidates), large.candidate_losses(candidates),
                                   rtol=1e-5, atol=1e-6)

    def test_two_pilots_fresh_ranking_outputs_and_validation_independence(self):
        model, features, labels = fixture()
        train = {"features": features, "labels": labels, "img_ids": torch.arange(12)}
        validation = {"features": features[:6] * 1.1, "labels": labels[:6], "img_ids": torch.arange(100, 106)}
        args = argparse.Namespace(calibration_per_class=1, seed=40, second_start=8, removals=2,
                                  image_batch_size=4, candidate_chunk_size=2, shortlist_size=4, batch_size=2,
                                  batch_extra_loss_tol=0.005, batch_extra_loss_rel=0.25, smoke=True)
        call_sizes = []
        def ranked(current, *other):
            call_sizes.append(current.num_prototypes)
            return p1_ranking(current, *other)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "first"
            out.mkdir()
            with mock.patch("pilot_physical_pruning.p1_ranking", side_effect=ranked):
                summary = run_pilots(model, train, validation, args, out)
            self.assertEqual(call_sizes, [12, 8])
            self.assertEqual(len(summary["results"]), 6)
            self.assertTrue(summary["starts"][1]["ranking_recomputed_from_compact_start"])
            self.assertTrue((out / "results.csv").is_file())
            for row in summary["results"]:
                self.assertEqual(row["remaining_codes"], 10 if row["pilot"] == 1 else 6)
                self.assertTrue(row["checkpoint_reload_verification"]["passed"])
            changed = {**validation, "labels": (validation["labels"] + 1) % 3}
            second = Path(temporary) / "second"
            second.mkdir()
            repeated = run_pilots(model, train, changed, args, second)
            self.assertEqual([r["retained_dense_ids"] for r in summary["results"]],
                             [r["retained_dense_ids"] for r in repeated["results"]])
            ranking = torch.load(out / "pilot2_8_to_6" / "fixed_p1_ranking.pt", weights_only=True)
            shared = compact_model(model, p1_ranking(model, features, labels, 4)[1][-8:].sort().values)
            fresh, _ = p1_ranking(shared, features, labels, 4)
            torch.testing.assert_close(ranking["importance"], fresh)
        self.assertEqual(model.num_prototypes, 12)


if __name__ == "__main__":
    unittest.main()
