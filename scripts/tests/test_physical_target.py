"""Small synthetic target-runner tests; no CUB images, pretrained weights or full experiments."""
from __future__ import annotations

import argparse
import copy
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from test_physical_pruning import fixture
import physical_pruning_target as T


def options(target=5, smoke=False, reuse=None):
    return argparse.Namespace(target_count=target, smoke=smoke, seed=40, calibration_per_class=1,
                              shortlist_size=4, batch_size=3, image_batch_size=4, candidate_chunk_size=2,
                              batch_extra_loss_tol=0.005, batch_extra_loss_rel=0.25,
                              reuse_p1_dir=reuse, pq_ckpt=Path("existing_dense_seed40.pth"),
                              backbone_ckpt=Path(T.SELECTED_BACKBONE), device="auto")


def data(features, labels):
    train = {"features": features, "labels": labels, "img_ids": torch.arange(len(labels))}
    val = {"features": features[:6] * 1.1, "labels": labels[:6], "img_ids": torch.arange(100, 106)}
    return train, val


class PhysicalTargetTests(unittest.TestCase):
    def run_small(self, output, args=None, codes=12, validation_labels=None):
        model, features, labels = fixture(codes)
        train, val = data(features, labels)
        if validation_labels is not None:
            val["labels"] = validation_labels
        with contextlib.redirect_stdout(io.StringIO()):
            result = T.run_comparison(model, train, val, args or options(), output,
                                      {"fingerprint": "synthetic-test-identity"})
        return model, result

    def test_three_configurations_exact_physical_count_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            model, result = self.run_small(out)
            self.assertEqual([r["method"] for r in result["results"]], ["dense", "p1", "batched"])
            self.assertEqual([r["remaining_codes"] for r in result["results"]], [12, 5, 5])
            self.assertEqual(model.num_prototypes, 12)
            self.assertFalse(result["test_read"])
            self.assertEqual(result["retraining_epochs"], 0)
            self.assertTrue(result["results"][0]["checkpoint_reused"])
            self.assertEqual(result["results"][0]["pruning_seconds"], 0)
            self.assertFalse((out / "dense/pruned.pth").exists())
            ranking = torch.load(out / "fixed_p1_ranking.pt", weights_only=True)["ranking"]
            self.assertEqual(result["results"][1]["retained_dense_ids"], ranking[-5:].sort().values.tolist())
            for row in result["results"]:
                self.assertTrue(row["physical_inference_verification"]["passed"])
                if row["method"] != "dense":
                    state = torch.load(out / row["method"] / "pruned.pth", weights_only=True)["model"]
                    self.assertEqual(state["codes"].shape[0], 5)
                    self.assertEqual(state["classifier.weight"].shape[1], 5)
                    self.assertTrue(row["checkpoint_reload_verification"]["passed"])
                predictions = out / row["method"] / "validation_predictions.npz"
                self.assertTrue(predictions.is_file())
            self.assertTrue((out / "results.csv").is_file())
            self.assertTrue((out / "results.md").is_file())

    def test_long_sequence_3000_to_1300_matches_compact_inference(self):
        # Real target dimensions, but only 12 synthetic 8-channel images, 3 classes and 4 positions.
        args = options(target=1300)
        args.batch_size = 64
        with tempfile.TemporaryDirectory() as temporary:
            _, result = self.run_small(Path(temporary), args, codes=3000)
            for row in result["results"][1:]:
                self.assertEqual(row["remaining_codes"], 1300)
                self.assertEqual(row["removed_count"], 1700)
                self.assertEqual(len(set(row["retained_dense_ids"])), 1300)
                self.assertTrue(row["checkpoint_reload_verification"]["passed"])

    def test_validation_labels_do_not_influence_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / "a", Path(temporary) / "b"
            a.mkdir(); b.mkdir()
            _, first = self.run_small(a)
            _, second = self.run_small(b, validation_labels=torch.zeros(6, dtype=torch.long))
            self.assertEqual([r["retained_dense_ids"] for r in first["results"]],
                             [r["retained_dense_ids"] for r in second["results"]])
            calibration = json.loads((a / "calibration.json").read_text())
            self.assertEqual(calibration["source_split"], "train")
            self.assertTrue(set(calibration["img_ids"]) <= set(range(12)))

    def test_reuse_requires_same_identity_and_tensor_identical_p1_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / "a", Path(temporary) / "b"
            a.mkdir(); b.mkdir()
            model, first = self.run_small(a)
            _, second = self.run_small(b, options(reuse=a / "p1"))
            reused = second["results"][1]
            self.assertTrue(reused["result_reused"])
            self.assertEqual(reused["timing_source"], "previous_verified_run")
            self.assertEqual(reused["pruning_seconds"], first["results"][1]["pruning_seconds"])
            retained = torch.tensor(first["results"][1]["retained_dense_ids"], device=model.codes.device)
            with self.assertRaisesRegex(ValueError, "not a verified matching"):
                T.reusable_p1(a / "p1", model, retained, "another-identity")
            path = a / "p1/pruned.pth"
            bundle = torch.load(path, weights_only=True)
            bundle["model"]["codes"][0, 0] += 0.1
            torch.save(bundle, path)
            with self.assertRaisesRegex(ValueError, "differs from exact"):
                T.reusable_p1(a / "p1", model, retained, "synthetic-test-identity")

    def test_smoke_removes_only_two_prototypes(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, result = self.run_small(Path(temporary), options(target=1300, smoke=True))
            self.assertEqual([r["remaining_codes"] for r in result["results"]], [12, 10, 10])
            self.assertTrue(result["smoke"])
            self.assertEqual(result["requested_target_count"], 1300)

    def test_nonfinite_features_and_split_overlap_are_rejected(self):
        model, features, labels = fixture()
        for nonfinite in (True, False):
            train, val = data(features.clone(), labels)
            if nonfinite:
                train["features"][0, 0, 0, 0] = torch.nan
            else:
                val["img_ids"][0] = train["img_ids"][0]
            with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "Nonfinite|overlap"):
                T.run_comparison(model, train, val, options(), Path(temporary), {"fingerprint": "test"})

    def test_original_backbone_other_targets_and_seeds_are_rejected(self):
        valid = options(target=1300)
        T.validate_args(valid)
        for changes in ({"seed": 41}, {"target_count": 1500},
                        {"backbone_ckpt": Path("cub200_convnext_tiny_full_20261116_135156.pth")},
                        {"batch_extra_loss_tol": float("nan")}, {"candidate_chunk_size": 0},
                        {"smoke": True, "reuse_p1_dir": Path("previous")}):
            args = argparse.Namespace(**{**vars(valid), **changes})
            with self.assertRaises(ValueError):
                T.validate_args(args)

    def test_fresh_output_does_not_overwrite_previous_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "run.log").touch()
            T.fresh_output(path)
            (path / "results.json").write_text("old results")
            with self.assertRaisesRegex(ValueError, "not fresh"):
                T.fresh_output(path)
            self.assertEqual((path / "results.json").read_text(), "old results")

    def test_main_records_completion_and_failure_without_real_artifacts(self):
        model, features, labels = fixture()
        train, val = data(features, labels)
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                argv = ["physical_pruning_target.py", "--backbone_ckpt", T.SELECTED_BACKBONE,
                        "--pq_ckpt", "dense.pth", "--cache_dir", "cache", "--cub_path", "cub",
                        "--output_dir", temporary, "--smoke"]
                with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()), \
                     mock.patch.object(T, "prepare_inputs", return_value=(model, train, val,
                                       {"fingerprint": "test", "identity": {}}, None)):
                    if fail:
                        with mock.patch.object(T, "run_comparison", side_effect=RuntimeError("synthetic failure")), \
                             self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                            T.main()
                    else:
                        T.main()
                state = json.loads((Path(temporary) / "state.json").read_text())
                self.assertEqual(state["status"], "failed" if fail else "completed")
                self.assertEqual((Path(temporary) / "DONE").exists(), not fail)

    def test_missing_inputs_and_non_strict_protocol_fail_before_data_loading(self):
        # These paths never exist: no pretrained checkpoint or dataset is read in this test.
        import types
        args = options(target=1300)
        args.cub_path, args.cache_dir = Path("absent-cub"), Path("absent-cache")
        for protocol in ("paper", "strict"):
            fake = types.SimpleNamespace(protocol_info=lambda: {"protocol": protocol})
            with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(sys.modules, {"pq_lib": fake}), \
                 self.assertRaisesRegex(ValueError, "requires PQ_CUB_PROTOCOL|Missing checkpoint"):
                T.prepare_inputs(args, Path(temporary))

    def test_existing_dense_validation_result_is_checked_and_reused_as_reference(self):
        args = options()
        _, features, labels = fixture()
        _, val = data(features, labels)
        logits = torch.randn(6, 3)
        accuracy = float((logits.argmax(1) == val["labels"]).float().mean() * 100)
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "eval_val"
            metadata = {"split": "val", "protocol": "strict", "split_sha256": "matching", "top1": accuracy,
                        "pq_ckpt": str(args.pq_ckpt.resolve()), "backbone_ckpt": str(args.backbone_ckpt.resolve())}
            prefix.with_suffix(".json").write_text(json.dumps(metadata))
            np.savez(prefix.with_suffix(".npz"), img_ids=val["img_ids"].numpy(),
                     labels=val["labels"].numpy(), logits=logits.numpy())
            with contextlib.redirect_stdout(io.StringIO()):
                result = T.existing_dense_evaluation(prefix, args, val, logits, {"split_sha256": "matching"})
            self.assertEqual(result["status"], "consistent_reference")
            self.assertEqual(result["recorded_accuracy"], accuracy)
            with contextlib.redirect_stdout(io.StringIO()):
                changed = T.existing_dense_evaluation(prefix, args, val, logits + 0.5, {"split_sha256": "matching"})
            self.assertEqual(changed["status"], "reference_differs")
            with self.assertRaisesRegex(ValueError, "different strict split"):
                T.existing_dense_evaluation(prefix, args, val, logits, {"split_sha256": "different"})

    def test_test_evaluation_reference_is_rejected_before_predictions_are_opened(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "eval_test"
            prefix.with_suffix(".json").write_text(json.dumps({"split": "test", "protocol": "strict"}))
            with mock.patch.object(T.np, "load") as loader, self.assertRaisesRegex(ValueError, "never a test result"):
                T.existing_dense_evaluation(prefix, options(), {}, torch.empty(0), {})
            loader.assert_not_called()

    def test_preflight_checks_matched_artifacts_and_never_requests_test_split(self):
        import types
        from src.models.proto_quantnet import ProtoQuantNet

        class Backbone(torch.nn.Module):
            def __init__(self, scale=1.):
                super().__init__()
                self.register_buffer("scale", torch.tensor(scale))
            def forward(self, image):
                return image * self.scale

        class Dataset:
            def __init__(self, split):
                self.img_ids = list(range(5394)) if split == "train" else list(range(10000, 10600))
            def __len__(self):
                return len(self.img_ids)
            def __getitem__(self, index):
                return torch.ones(8, 2, 2), 0

        device = fixture()[0].codes.device
        model = ProtoQuantNet(Backbone(), torch.randn(3000, 8), 200).to(device).eval()
        split_calls = []
        def dataset(root, split):
            split_calls.append(split)
            self.assertIn(split, ("train", "val"))
            return Dataset(split)
        # Expanded views keep this test tiny despite checking the strict dataset lengths.
        train = {"features": torch.ones(1, 8, 2, 2).expand(5394, -1, -1, -1)}
        val = {"features": torch.ones(1, 8, 2, 2).expand(600, -1, -1, -1)}
        fake = types.SimpleNamespace(protocol_info=lambda: {"protocol": "strict", "split_sha256": "test-split"},
                                     dataset=dataset, load_protoquant=lambda *a: copy.deepcopy(model),
                                     load_backbone_classifier=lambda *a: Backbone(), get_backbone=lambda m: m)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            args = options(target=1300, smoke=True)
            args.device = str(device)
            args.backbone_ckpt = out / T.SELECTED_BACKBONE
            args.backbone_ckpt.write_bytes(b"synthetic backbone fingerprint")
            args.pq_ckpt = out / "dense.pth"
            torch.save({"model": model.state_dict()}, args.pq_ckpt)
            args.cub_path, args.cache_dir = out / "cub", out / "cache"
            root = args.cub_path / "CUB_200_2011"
            (root / "images").mkdir(parents=True)
            for name in ("images.txt", "image_class_labels.txt", "train_test_split.txt"):
                (root / name).touch()
            args.cache_dir.mkdir()
            for name in ("train.pt", "val.pt"):
                (args.cache_dir / name).write_bytes(b"synthetic cache fingerprint")
            with mock.patch.dict(sys.modules, {"pq_lib": fake}), \
                 mock.patch.object(T, "load_cache", side_effect=[train, val]), \
                 contextlib.redirect_stdout(io.StringIO()):
                loaded, _, _, provenance, probe = T.prepare_inputs(args, out)
            self.assertEqual(split_calls, ["train", "val"])
            self.assertEqual(loaded.num_prototypes, 3000)
            self.assertEqual(provenance["identity"]["backbone_sha256"], T.sha256(args.backbone_ckpt))
            self.assertTrue(all(c["passed"] for c in provenance["cache_feature_checks"]))
            fake.load_backbone_classifier = lambda *a: Backbone(2.)
            with mock.patch.dict(sys.modules, {"pq_lib": fake}), \
                 mock.patch.object(T, "load_cache", side_effect=[train, val]), \
                 self.assertRaisesRegex(ValueError, "backbone differs"):
                T.prepare_inputs(args, out)


if __name__ == "__main__":
    unittest.main()
