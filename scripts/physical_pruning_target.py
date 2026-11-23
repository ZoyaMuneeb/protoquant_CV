"""Seed-40 improved-backbone dense/P1/batched comparison; validation only, no training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from physical_pruning_lib import DeletionEvaluator, compact_model, forward_features, p1_ranking, prune, spatial_logits, synchronize
from pilot_physical_pruning import (
    balanced_indices, compare_cached_features, evaluate_features, load_cache, reload_check,
    save_checkpoint, verify_compact, verify_deletions, write_json,
)

SELECTED_BACKBONE = "cub200_convnext_tiny_full_20261118_223411.pth"
METHODS = ("dense", "p1", "batched")
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fresh_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        # The launcher creates only run.log before starting Python.
        if {item.name for item in path.iterdir()} != {"run.log"}:
            raise ValueError(f"Output is not fresh: {path}. Keep the old run and choose a new TARGET_OUTPUT_DIR; no resume.")
    path.mkdir(parents=True, exist_ok=True)


def validate_args(args) -> None:
    if args.seed != 40:
        raise ValueError("This experiment is restricted to the improved-backbone seed-40 checkpoint.")
    if args.target_count != 1300:
        raise ValueError("The approved experiment requires --target-count 1300.")
    if min(args.calibration_per_class, args.shortlist_size, args.batch_size,
           args.image_batch_size, args.candidate_chunk_size) < 1:
        raise ValueError("Calibration, shortlist, batch and chunk sizes must be positive.")
    for value in (args.batch_extra_loss_tol, args.batch_extra_loss_rel):
        if not np.isfinite(value) or value < 0:
            raise ValueError("Batch tolerances must be finite and nonnegative.")
    if args.backbone_ckpt.name != SELECTED_BACKBONE:
        raise ValueError(f"Use the selected improved backbone {SELECTED_BACKBONE}, not the original pilot backbone.")
    if args.smoke and args.reuse_p1_dir:
        raise ValueError("Smoke runs cannot reuse a full P1 result; omit P1_REUSE_DIR.")


def peak_memory(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"peak_gpu_allocated_mib": None, "peak_gpu_reserved_mib": None}
    return {"peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}


@torch.no_grad()
def prepare_inputs(args, output: Path):
    """Reuse original cache/feature checks, and enforce exact checkpoint/backbone equality."""
    import pq_lib as L

    if L.protocol_info()["protocol"] != "strict":
        raise ValueError("This experiment requires PQ_CUB_PROTOCOL=strict.")
    for path in (args.backbone_ckpt, args.pq_ckpt):
        if not path.is_file():
            raise ValueError(f"Missing checkpoint: {path}. Configure BACKBONE_CKPT and PQ_CKPT.")
    root = args.cub_path / "CUB_200_2011"
    for name in ("images", "images.txt", "image_class_labels.txt", "train_test_split.txt"):
        if not (root / name).exists():
            raise ValueError(f"Missing dataset path: {root / name}. Configure CUB_PATH.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Check the PyTorch environment and driver.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cpu" and not args.smoke:
        raise ValueError("The complete experiment requires CUDA. Use --smoke or --self-test for a small check.")
    torch.manual_seed(args.seed)
    train_ds = L.dataset(str(args.cub_path), "train")
    val_ds = L.dataset(str(args.cub_path), "val")
    if (len(train_ds), len(val_ds)) != (5394, 600):
        raise ValueError("Expected strict CUB split: 5,394 train / 600 validation images.")
    bb_sha = sha256(args.backbone_ckpt)
    train = load_cache(args.cache_dir / "train.pt", train_ds, "train", bb_sha)
    validation = load_cache(args.cache_dir / "val.pt", val_ds, "val", bb_sha)
    model = L.load_protoquant(str(args.backbone_ckpt), str(args.pq_ckpt), device).eval()
    bundle = torch.load(args.pq_ckpt, map_location="cpu", weights_only=True)
    raw = bundle.get("model", bundle.get("state_dict", bundle))
    normalized = {}
    for key, value in raw.items():
        while key.startswith(("module.", "model_to_wrap.")):
            key = key.split(".", 1)[1]
        if key in normalized:
            raise ValueError(f"Duplicate checkpoint tensor after prefix normalization: {key}")
        normalized[key] = value
    model.load_state_dict(normalized, strict=True)
    del raw, bundle, normalized
    if model.num_prototypes != 3000 or model.num_classes != 200:
        raise ValueError("PQ_CKPT must be the dense seed-40, 3,000-prototype, 200-class Stage 2 model.")
    reference = L.get_backbone(L.load_backbone_classifier(str(args.backbone_ckpt), torch.device("cpu")))
    expected, actual = reference.state_dict(), model.backbone.state_dict()
    if expected.keys() != actual.keys() or any(not torch.equal(value, actual[key].cpu())
                                             for key, value in expected.items()):
        raise ValueError("PQ_CKPT backbone differs from the selected improved backbone/cache. Use runs_bb2/phase1 artifacts.")
    del expected, actual, reference
    checks = []
    for index in (0, len(train_ds) // 2):
        image, label = train_ds[index]
        fresh = model.backbone(image[None].to(device)).float().cpu()
        cached = train["features"][index:index + 1].float()
        check = compare_cached_features(fresh, cached)
        check.update(image_id=train_ds.img_ids[index], training_index=index)
        if check["passed"]:
            a, b = forward_features(model, fresh.to(device)), forward_features(model, cached.to(device))
            target = torch.tensor([label], device=device)
            check.update(max_absolute_logit_difference=float((a - b).abs().max()),
                         prediction_matches=bool((a.argmax(1) == b.argmax(1)).all()),
                         cross_entropy_difference=float(F.cross_entropy(b, target) - F.cross_entropy(a, target)))
        checks.append(check)
        write_json(output / "cache_feature_checks.json", {"backbone_sha256": bb_sha, "checks": checks,
                   "torch": str(torch.__version__), "cudnn_tf32": torch.backends.cudnn.allow_tf32})
        print("[CACHE CHECK] " + json.dumps(check), flush=True)
        if not check["passed"]:
            raise ValueError("Cache features exceed bounded numerical-drift limits; see cache_feature_checks.json.")
        if check["reason"] == "bounded_numerical_drift":
            print("[WARN] accepted bounded numerical cache drift", flush=True)
    image, label = train_ds[0]
    image = image[None].to(device)
    features = model.backbone(image).float()
    probe = DeletionEvaluator(spatial_logits(model, features, 1), model.classifier.weight,
                              torch.tensor([label]), model.classifier.bias)
    probe.remove([0, 1])
    torch.testing.assert_close(compact_model(model, probe.ids)(image).logits, probe.predictions,
                               rtol=2e-4, atol=2e-5)
    source_names = ("scripts/physical_pruning_lib.py", "scripts/pilot_physical_pruning.py",
                    "scripts/physical_pruning_target.py", "scripts/phase2_lib.py", "scripts/pq_lib.py",
                    "codebook_train/src/pipnet_utils.py", "codebook_train/src/construct_model.py")
    source_names += tuple(str(path.relative_to(ROOT)).replace("\\", "/") for base in
                          (ROOT / "codebook_train/src/models", ROOT / "codebook_train/src/datasets")
                          for path in sorted(base.rglob("*.py")))
    identity = {"schema": 1, "backbone_sha256": bb_sha, "pq_sha256": sha256(args.pq_ckpt),
                "cache_sha256": {split: sha256(args.cache_dir / f"{split}.pt") for split in ("train", "val")},
                "protocol": L.protocol_info(), "seed": args.seed, "requested_target": args.target_count,
                "smoke": args.smoke, "torch": str(torch.__version__),
                "code_sha256": {name: sha256(ROOT / name) for name in source_names}}
    # Retain one real training image for final image -> compact-model verification.
    return model, train, validation, {"identity": identity, "fingerprint": fingerprint(identity),
                                     "cache_feature_checks": checks}, (image.cpu(), features.cpu())


@torch.no_grad()
def reusable_p1(directory: Path, source, retained: torch.Tensor, identity_hash: str):
    """Only reuse a verified full target-run result with the same artifacts, code and split."""
    for name in ("results.json", "pruned.pth"):
        if not (directory / name).is_file():
            raise ValueError(f"P1_REUSE_DIR lacks {name}: {directory}")
    result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    if (result.get("experiment_fingerprint") != identity_hash or result.get("method") != "p1"
            or result.get("retained_dense_ids") != retained.cpu().tolist()
            or result.get("remaining_codes") != retained.numel()
            or result.get("smoke") is not False
            or not result.get("checkpoint_reload_verification", {}).get("passed")):
        raise ValueError("P1_REUSE_DIR is not a verified matching full P1-only target result; omit it to recompute.")
    compact = compact_model(source, retained)
    saved = torch.load(directory / "pruned.pth", map_location="cpu", weights_only=True)["model"]
    state = compact.state_dict()
    if state.keys() != saved.keys() or any(not torch.equal(value.cpu(), saved[key]) for key, value in state.items()):
        raise ValueError("Reused P1 checkpoint differs from exact dense-model P1 compaction.")
    return compact, result


def existing_dense_evaluation(prefix: Path, args, validation: dict, logits: torch.Tensor, protocol: dict) -> dict:
    """Reuse recorded validation results as a reference, never opening a test evaluation."""
    metadata = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("split") != "val" or metadata.get("protocol") != "strict":
        raise ValueError("DENSE_EVAL_PREFIX must reference a strict validation evaluation, never a test result.")
    for name in ("pq_ckpt", "backbone_ckpt"):
        if Path(metadata.get(name, "")).resolve() != getattr(args, name).resolve():
            raise ValueError(f"Existing dense evaluation has a different {name}; unset DENSE_EVAL_PREFIX to measure afresh.")
    if metadata.get("split_sha256") != protocol.get("split_sha256"):
        raise ValueError("Existing dense evaluation uses a different strict split.")
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as saved:
        if (not np.array_equal(saved["img_ids"], validation["img_ids"].numpy()) or
                not np.array_equal(saved["labels"], validation["labels"].numpy())):
            raise ValueError("Existing dense validation IDs/labels differ from the current validation cache.")
        previous = torch.from_numpy(saved["logits"].copy()).float()
    if previous.shape != logits.shape or not torch.isfinite(previous).all():
        raise ValueError("Existing dense validation logits are malformed/nonfinite.")
    previous_accuracy = float((previous.argmax(1) == validation["labels"]).float().mean() * 100)
    if abs(previous_accuracy - metadata["top1"]) > 2e-5:
        raise ValueError("Existing dense accuracy does not match its saved validation logits.")
    same_predictions = bool((previous.argmax(1) == logits.argmax(1)).all())
    close = bool(torch.isclose(previous, logits, rtol=0.003, atol=0.003).all())
    reference = {"prefix": str(prefix.resolve()), "recorded_accuracy": metadata["top1"],
                 "cross_entropy_from_saved_logits": float(F.cross_entropy(previous, validation["labels"])),
                 "predictions_match_cached_evaluation": same_predictions,
                 "max_absolute_logit_difference": float((previous - logits).abs().max()),
                 "status": "consistent_reference" if close and same_predictions else "reference_differs"}
    print("[DENSE REFERENCE] " + json.dumps(reference), flush=True)
    if reference["status"] != "consistent_reference":
        print("[WARN] Dense image-evaluation reference differs; fresh cached validation metrics are used for all comparisons.", flush=True)
    return reference


@torch.no_grad()
def run_comparison(model, train: dict, validation: dict, args, output: Path,
                   provenance: dict, image_probe=None) -> dict:
    """Exactly three rows; validation never supplies candidate rankings or deletion losses."""
    device = model.codes.device
    for name, data in (("training", train), ("validation", validation)):
        if not torch.isfinite(data["features"]).all():
            raise ValueError(f"Nonfinite {name} cache features.")
    if set(train["img_ids"].tolist()) & set(validation["img_ids"].tolist()):
        raise ValueError("Training and validation image IDs overlap.")
    count = model.num_prototypes
    target = count - min(2, count - 1) if args.smoke else args.target_count
    if not 0 < target < count:
        raise ValueError("Target must be positive and smaller than the dense prototype count.")
    calibration = balanced_indices(train["labels"], model.num_classes, args.calibration_per_class, args.seed)
    cal_features, cal_labels = train["features"][calibration], train["labels"][calibration]
    write_json(output / "calibration.json", {"source_split": "train", "seed": args.seed,
               "per_class": args.calibration_per_class, "img_ids": train["img_ids"][calibration].tolist(),
               "labels": cal_labels.tolist()})
    synchronize(device)
    t0 = time.perf_counter()
    importance, ranking = p1_ranking(model, train["features"], train["labels"], args.image_batch_size)
    synchronize(device)
    p1_seconds = time.perf_counter() - t0
    torch.save({"importance": importance, "ranking": ranking, "codes": count,
                "scoring_images": len(train["labels"]), "fixed_during_pruning": True}, output / "fixed_p1_ranking.pt")
    t0 = time.perf_counter()
    similarities = spatial_logits(model, cal_features, args.image_batch_size)
    synchronize(device)
    calibration_seconds = time.perf_counter() - t0
    initial_checks = verify_deletions(model, similarities, cal_features, cal_labels, ranking, args)
    write_json(output / "starting_checks.json", {"deletions": initial_checks})
    rows = []
    baseline = None
    for method in METHODS:
        directory = output / method
        directory.mkdir()
        print(f"[TARGET] {method}: {count} -> {count if method == 'dense' else target}", flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        pruning_seconds = 0.0
        compact, engine, reused = model, None, None
        retained = torch.arange(count, device=device)
        result = {"candidate_evaluations": 0, "joint_evaluations": 0, "fallback_count": 0}
        if method != "dense":
            if method == "p1" and args.reuse_p1_dir:
                retained = ranking[count - target:].sort().values.to(device)
                compact, reused = reusable_p1(args.reuse_p1_dir, model, retained, provenance["fingerprint"])
                result.update({key: reused[key] for key in ("candidate_evaluations", "joint_evaluations", "fallback_count")})
                pruning_seconds = reused["pruning_seconds"]
            else:
                engine = DeletionEvaluator(similarities, model.classifier.weight, cal_labels, model.classifier.bias,
                                           args.image_batch_size, args.candidate_chunk_size)
                with (directory / "decisions.jsonl").open("w", encoding="utf-8") as log:
                    def record(event):
                        log.write(json.dumps(event, allow_nan=False) + "\n")
                        log.flush()
                        if event["event"] == "accepted":
                            print(f"[TARGET] {method}: removed {event['removed_total']}/{count - target}; "
                                  f"codes={event['remaining_codes']}; calibration CE={event['loss_after']:.6f}", flush=True)
                    synchronize(device)
                    t0 = time.perf_counter()
                    result = prune(engine, ranking, count - target, method, args.shortlist_size, args.batch_size,
                                   args.batch_extra_loss_tol, args.batch_extra_loss_rel, record)
                    retained = engine.ids
                    compact = compact_model(model, retained)
                    synchronize(device)
                    pruning_seconds = time.perf_counter() - t0
        pruning_memory = peak_memory(device)
        if reused:
            (directory / "decisions.jsonl").write_text(json.dumps({"event": "reused_p1_result",
                "source": str(args.reuse_p1_dir.resolve()), "selection_not_rerun": True}) + "\n", encoding="utf-8")
        if method == "dense":
            reference_logits = forward_features(model, cal_features[:4].to(device))
        elif engine is not None:
            reference_logits = engine.predictions[:4]
        else:
            # Verify reused P1 against the same joint-deletion engine as a new result.
            probe = DeletionEvaluator(similarities[:4], model.classifier.weight, cal_labels[:4], model.classifier.bias)
            retained_set = set(retained.cpu().tolist())
            removed = [i for i in range(count) if i not in retained_set]
            probe.remove(removed)
            reference_logits = probe.predictions
        check = verify_compact(model, retained, cal_features[:4], reference_logits)
        if image_probe is not None:
            image, features = image_probe
            actual = compact(image.to(device)).logits
            expected = forward_features(compact, features.to(device))
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
            check["real_training_image_passed"] = True
        synchronize(device)
        t0 = time.perf_counter()
        metrics, logits = evaluate_features(compact, validation["features"], validation["labels"], args.image_batch_size)
        synchronize(device)
        evaluation_seconds = time.perf_counter() - t0
        if baseline is None:
            baseline = metrics
        expected_count = count if method == "dense" else target
        if compact.codes.shape[0] != expected_count or compact.classifier.weight.shape[1] != expected_count:
            raise RuntimeError("Physical code rows / classifier columns do not match target.")
        result.update(method=method, smoke=args.smoke, starting_codes=count, remaining_codes=compact.num_prototypes,
                      removed_count=count - compact.num_prototypes, validation_accuracy=metrics["accuracy"],
                      validation_cross_entropy=metrics["cross_entropy"], validation_images=metrics["images"],
                      validation_accuracy_change=metrics["accuracy"] - baseline["accuracy"],
                      validation_loss_change=metrics["cross_entropy"] - baseline["cross_entropy"],
                      nonzero_edges=int((torch.relu(compact.classifier.weight) > 0).sum()),
                      active_classifier_codes=int((torch.relu(compact.classifier.weight) > 0).any(0).sum()),
                      physical_code_rows=compact.codes.shape[0], classifier_columns=compact.classifier.weight.shape[1],
                      codebook_mib_fp32=compact.codes.numel() * 4 / 2**20,
                      pruning_seconds=pruning_seconds, evaluation_seconds=evaluation_seconds,
                      pruning_memory=pruning_memory, **peak_memory(device),
                      retained_dense_ids=retained.cpu().tolist(), physical_inference_verification=check,
                      experiment_fingerprint=provenance["fingerprint"], result_reused=reused is not None,
                      timing_source="previous_verified_run" if reused else "this_run")
        if reused:
            result.update(reused_from=str(args.reuse_p1_dir.resolve()),
                          original_pruning_memory=reused.get("pruning_memory"),
                          original_run_peak_gpu_allocated_mib=reused.get("peak_gpu_allocated_mib"))
        if method == "dense":
            result["checkpoint"] = str(args.pq_ckpt)
            result["checkpoint_reused"] = True
            if getattr(args, "dense_eval_prefix", None):
                result["existing_dense_evaluation"] = existing_dense_evaluation(
                    args.dense_eval_prefix, args, validation, logits, provenance["identity"]["protocol"])
        else:
            checkpoint = directory / "pruned.pth"
            save_checkpoint(checkpoint, compact, result)
            result["checkpoint"] = str(checkpoint.resolve())
            result["checkpoint_reload_verification"] = reload_check(checkpoint, model, compact, cal_features[:4])
        # Include reload-verification allocations in the per-method peak, unlike pruning-only peaks.
        result.update(peak_memory(device))
        np.savez_compressed(directory / "validation_predictions.npz", img_ids=validation["img_ids"].numpy(),
                            labels=validation["labels"].numpy(), logits=logits.numpy(), preds=logits.argmax(1).numpy())
        write_json(directory / "results.json", result)
        rows.append(result)
        print(f"[TARGET] {method}: val={metrics['accuracy']:.3f}% CE={metrics['cross_entropy']:.6f}", flush=True)
        del engine, compact
    summary = {"smoke": args.smoke, "target_count": target, "requested_target_count": args.target_count,
               "selection_split": "train", "evaluation_split": "val", "test_read": False,
               "retraining_epochs": 0, "experiment_fingerprint": provenance["fingerprint"],
               "shared_setup_seconds": {"p1_ranking": p1_seconds, "calibration_logits": calibration_seconds},
               "results": rows}
    write_json(output / "results.json", summary)
    fields = ("method", "remaining_codes", "physical_code_rows", "classifier_columns", "nonzero_edges",
              "validation_accuracy", "validation_accuracy_change", "validation_cross_entropy", "validation_loss_change",
              "pruning_seconds", "evaluation_seconds", "peak_gpu_allocated_mib", "peak_gpu_reserved_mib",
              "candidate_evaluations", "joint_evaluations", "fallback_count", "result_reused", "timing_source")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = ["# Improved-backbone physical prototype pruning", "",
              f"Smoke run: {args.smoke}. Seed 40. Validation only. No retraining or test evaluation.", "",
              "| Method | Physical codes | Val top-1 (%) | Val CE | Edges | Pruning seconds | Peak GPU MiB | Reused |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        memory = row["peak_gpu_allocated_mib"]
        memory_text = f"{memory:.1f}" if memory is not None else "n/a"
        report.append(f"| {row['method']} | {row['remaining_codes']} | {row['validation_accuracy']:.3f} | "
                      f"{row['validation_cross_entropy']:.6f} | {row['nonzero_edges']} | {row['pruning_seconds']:.3f} | "
                      f"{memory_text} | {row['result_reused']} |")
    report += ["", "Dense reuses the existing checkpoint; validation is measured again to obtain matching cross-entropy and GPU memory.",
               "A matching existing dense validation evaluation, when available, is reused as a checked reference in dense/results.json.",
               "Pruning time excludes shared P1/calibration setup, validation, correctness checks and checkpoint I/O.",
               "Peak GPU allocation includes resident tensors and per-method verification/evaluation; pruning-only peaks are in JSON.",
               "Reused P1 pruning time comes from its original verified run; current memory is measured again (original memory is in JSON).",
               "CPU smoke runs report no GPU memory. Batch tolerance limits interaction damage, not cumulative loss."]
    (output / "results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("backbone_ckpt", "pq_ckpt", "cache_dir", "cub_path", "output_dir"):
        ap.add_argument("--" + name, required=True, type=Path)
    ap.add_argument("--target-count", dest="target_count", type=int, default=1300)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--calibration_per_class", type=int, default=2)
    ap.add_argument("--shortlist_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--batch_extra_loss_tol", type=float, default=0.005)
    ap.add_argument("--batch_extra_loss_rel", type=float, default=0.25)
    ap.add_argument("--image_batch_size", type=int, default=32)
    ap.add_argument("--candidate_chunk_size", type=int, default=4)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--reuse_p1_dir", type=Path)
    ap.add_argument("--dense_eval_prefix", type=Path)
    ap.add_argument("--smoke", action="store_true")
    return ap


def main():
    args = parser().parse_args()
    validate_args(args)
    output = args.output_dir.resolve()
    fresh_output(output)
    started = time.time()
    state = {"status": "running", "started": started, "target_count": args.target_count, "smoke": args.smoke}
    write_json(output / "state.json", state)
    try:
        t0 = time.perf_counter()
        model, train, validation, provenance, probe = prepare_inputs(args, output)
        requested = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
        if args.smoke:
            args.calibration_per_class = 1
            small = balanced_indices(train["labels"], model.num_classes, 1, args.seed)
            train = {key: value[small] for key, value in train.items()}
            validation = {key: value[:32] for key, value in validation.items()}
            print("[SMOKE] 200 training images, 32 validation images, two removals; final pruned count is 2,998, not 1,300.", flush=True)
        config = {"requested": requested, "effective_calibration_per_class": args.calibration_per_class,
                  "effective_target_count": 2998 if args.smoke else args.target_count, **provenance,
                  "initial_image_inference_verification": "passed", "test_data_loaded": False,
                  "device": str(model.codes.device), "gpu": torch.cuda.get_device_name(model.codes.device)
                  if model.codes.device.type == "cuda" else None,
                  "input_setup_seconds": time.perf_counter() - t0}
        write_json(output / "config.json", config)
        run_comparison(model, train, validation, args, output, provenance, probe)
        write_json(output / "state.json", {**state, "status": "completed", "finished": time.time()})
        (output / "DONE").write_text("completed\n", encoding="utf-8")
        print(f"[DONE] Results: {output / 'results.md'}", flush=True)
    except Exception as error:
        write_json(output / "state.json", {**state, "status": "failed", "error": str(error), "finished": time.time()})
        print(f"[ERROR] {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
