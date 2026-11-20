"""Two no-retraining physical-pruning pilots. Run via scripts/run_physical_pilot.sh."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from physical_pruning_lib import (
    DeletionEvaluator, compact_model, forward_features, p1_ranking, prune,
    spatial_logits, synchronize,
)


METHODS = ("p1", "greedy", "batched")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, model: nn.Module, metadata: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"model": state, "pilot": metadata}, temporary)
    temporary.replace(path)


def compare_cached_features(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Permit a small, bounded numerical tail while retaining the original per-value check.

    FP16 caches generated in large batches can differ slightly from batch-1 inference.
    A relaxed pass allows at most 0.1% of coordinates outside the original tolerance,
    every error must be within twice that tolerance, and cosine must be >= 0.99999.
    This check diagnoses cache drift; compact-deletion equivalence stays strict.
    """
    if actual.shape != expected.shape or actual.numel() == 0:
        raise ValueError("Fresh and cached features must have the same nonempty shape.")
    actual = actual.detach().to(device="cpu", dtype=torch.float64)
    expected = expected.detach().to(device="cpu", dtype=torch.float64)
    policy = {"rtol": 0.003, "atol": 0.003, "max_failing_fraction": 0.001,
              "max_scaled_error": 2.0, "minimum_cosine": 0.99999}
    result = {"policy": policy, "total_values": actual.numel(), "passed": False}
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        return {**result, "reason": "nonfinite_features"}
    difference = (actual - expected).abs()
    allowed = policy["atol"] + policy["rtol"] * expected.abs()
    failing = int((difference > allowed).sum())
    fraction = failing / actual.numel()
    if not bool(actual.any() or expected.any()):
        cosine = 1.0
    else:
        cosine = float(F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0))
    scaled = float((difference / allowed).max())
    bounded = (fraction <= policy["max_failing_fraction"] and
               scaled <= policy["max_scaled_error"] and cosine >= policy["minimum_cosine"])
    passed = failing == 0 or bounded
    reason = "strict_match" if failing == 0 else ("bounded_numerical_drift" if bounded else "cache_mismatch")
    return {**result, "passed": passed, "reason": reason, "failing_values": failing,
            "failing_fraction": fraction, "max_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()), "max_scaled_error": scaled,
            "cosine_similarity": cosine}


def balanced_indices(labels: torch.Tensor, classes: int, per_class: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    chosen = []
    for label in range(classes):
        available = (labels == label).nonzero(as_tuple=False).flatten()
        if len(available) < per_class:
            raise ValueError(f"Class {label} has {len(available)} training images; need {per_class}.")
        chosen.append(available[torch.randperm(len(available), generator=generator)[:per_class]])
    return torch.cat(chosen)


@torch.no_grad()
def evaluate_features(model: nn.Module, features: torch.Tensor, labels: torch.Tensor,
                      batch_size: int) -> tuple[dict, torch.Tensor]:
    predictions = []
    for start in range(0, len(features), batch_size):
        predictions.append(forward_features(model, features[start:start + batch_size].to(model.codes.device)).cpu())
    logits = torch.cat(predictions)
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite validation logits.")
    return {"accuracy": float((logits.argmax(1) == labels).float().mean() * 100),
            "cross_entropy": float(F.cross_entropy(logits, labels)), "images": len(labels)}, logits


@torch.no_grad()
def verify_compact(model: nn.Module, retained: torch.Tensor, features: torch.Tensor,
                   expected_logits: torch.Tensor) -> dict:
    """Run ProtoQuantNet.forward on identical spatial inputs using an identity backbone."""
    reference = compact_model(model, retained)
    reference.backbone = nn.Identity()
    actual = reference(features.to(model.codes.device).float()).logits
    expected = expected_logits.to(actual.device)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    return {"passed": True, "max_absolute_logit_error": float((actual - expected).abs().max()),
            "images": len(features), "codes": reference.num_prototypes,
            "input": "identical cached spatial features through actual compact ProtoQuantNet.forward"}


@torch.no_grad()
def verify_deletions(model: nn.Module, logits: torch.Tensor, features: torch.Tensor,
                     labels: torch.Tensor, ranking: torch.Tensor, args) -> list[dict]:
    count = min(4, len(features))
    checks = []
    for size in sorted({1, min(args.batch_size, model.num_prototypes - 1)}):
        engine = DeletionEvaluator(logits[:count], model.classifier.weight, labels[:count],
                                   model.classifier.bias, args.image_batch_size, args.candidate_chunk_size)
        removed = ranking[:size].tolist()
        predicted_loss = engine.loss_after(removed)
        if size == 1:
            candidate_loss = float(engine.candidate_losses(removed)[0])
            if abs(predicted_loss - candidate_loss) > 1e-4:
                raise RuntimeError("Softmax-renormalized candidate loss disagrees with physical deletion.")
        engine.remove(removed)
        check = verify_compact(model, engine.ids, features[:count], engine.predictions)
        actual_loss = float(F.cross_entropy(engine.predictions, engine.labels))
        if abs(predicted_loss - actual_loss) > 1e-4:
            raise RuntimeError("Physical-removal calibration loss verification failed.")
        check.update(removed_count=size, loss_error=abs(predicted_loss - actual_loss))
        checks.append(check)
    return checks


@torch.no_grad()
def reload_check(path: Path, source: nn.Module, compact: nn.Module, features: torch.Tensor) -> dict:
    from src.pipnet_utils import build_pipnet_model

    reloaded = build_pipnet_model(
        backbone=copy.deepcopy(source.backbone), num_classes=compact.num_classes,
        device=compact.codes.device, pipnet_checkpoint_path=str(path), temperature=compact.temperature,
        train_codebook=isinstance(compact.codes, nn.Parameter),
    ).eval()
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    reloaded.load_state_dict(state, strict=True)
    actual = forward_features(reloaded, features.to(compact.codes.device))
    expected = forward_features(compact, features.to(compact.codes.device))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    return {"passed": True, "codes": reloaded.num_prototypes,
            "max_absolute_logit_error": float((actual - expected).abs().max())}


@torch.no_grad()
def run_pilots(model: nn.Module, train: dict, validation: dict, args, output: Path) -> dict:
    classes = model.classifier.weight.shape[0]
    calibration = balanced_indices(train["labels"], classes, args.calibration_per_class, args.seed)
    calibration_features = train["features"][calibration]
    calibration_labels = train["labels"][calibration]
    write_json(output / "calibration.json", {
        "source_split": "train", "seed": args.seed, "per_class": args.calibration_per_class,
        "img_ids": train["img_ids"][calibration].tolist(), "labels": calibration_labels.tolist(),
    })
    synchronize(model.codes.device)
    started = time.perf_counter()
    dense_importance, dense_ranking = p1_ranking(model, train["features"], train["labels"], args.image_batch_size)
    synchronize(model.codes.device)
    initial_p1_seconds = time.perf_counter() - started
    shared_indices = dense_ranking[-args.second_start:].sort().values
    shared = compact_model(model, shared_indices)
    save_checkpoint(output / "shared_p1_start.pth", shared, {
        "selection": "highest summed P1 from dense training activations", "codes": args.second_start,
        "retained_dense_ids": shared_indices.tolist(), "config": vars(args),
    })
    rows, starts = [], []
    for number, start_model in ((1, model), (2, shared)):
        pilot = output / f"pilot{number}_{start_model.num_prototypes}_to_{start_model.num_prototypes - args.removals}"
        pilot.mkdir()
        print(f"[PILOT {number}] starting with {start_model.num_prototypes} physical prototypes", flush=True)
        synchronize(model.codes.device)
        started = time.perf_counter()
        if number == 1:
            importance, ranking = dense_importance, dense_ranking
            p1_seconds = initial_p1_seconds
        else:
            # Recompute activations and P1 from the actual physically pruned 1,700-code model.
            importance, ranking = p1_ranking(start_model, train["features"], train["labels"], args.image_batch_size)
            synchronize(model.codes.device)
            p1_seconds = time.perf_counter() - started
        torch.save({"importance": importance, "ranking": ranking, "codes": start_model.num_prototypes,
                    "scoring_images": len(train["labels"]), "recomputed_from_compact_start": number == 2},
                   pilot / "fixed_p1_ranking.pt")
        synchronize(model.codes.device)
        started = time.perf_counter()
        similarities = spatial_logits(start_model, calibration_features, args.image_batch_size)
        synchronize(model.codes.device)
        calibration_setup_seconds = time.perf_counter() - started
        base_metrics, _ = evaluate_features(start_model, validation["features"], validation["labels"], args.image_batch_size)
        verification = verify_deletions(start_model, similarities, calibration_features, calibration_labels, ranking, args)
        start_info = {"pilot": number, "codes": start_model.num_prototypes,
                      "validation": base_metrics, "p1_seconds": p1_seconds,
                      "calibration_setup_seconds": calibration_setup_seconds,
                      "verification": verification, "ranking_recomputed_from_compact_start": number == 2}
        write_json(pilot / "starting_model.json", start_info)
        starts.append(start_info)
        for method in METHODS:
            directory = pilot / method
            directory.mkdir()
            print(f"[PILOT {number}] {method}: removing {args.removals} prototypes", flush=True)
            engine = DeletionEvaluator(similarities, start_model.classifier.weight, calibration_labels,
                                       start_model.classifier.bias, args.image_batch_size, args.candidate_chunk_size)
            with (directory / "decisions.jsonl").open("w", encoding="utf-8") as decisions:
                def record(event):
                    decisions.write(json.dumps(event, allow_nan=False) + "\n")
                    decisions.flush()
                    if event["event"] == "accepted":
                        print(f"[PILOT {number}] {method}: removed {event['removed_total']}/{args.removals}; "
                              f"codes={event['remaining_codes']}; calibration CE={event['loss_after']:.6f}", flush=True)
                if model.codes.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(model.codes.device)
                synchronize(model.codes.device)
                started = time.perf_counter()
                result = prune(engine, ranking, args.removals, method, args.shortlist_size, args.batch_size,
                               args.batch_extra_loss_tol, args.batch_extra_loss_rel, record)
                compact = compact_model(start_model, engine.ids)
                synchronize(model.codes.device)
                result["pruning_seconds"] = time.perf_counter() - started
                result["peak_gpu_allocated_mib"] = (torch.cuda.max_memory_allocated(model.codes.device) / 2**20
                                                       if model.codes.device.type == "cuda" else None)
            final_check = verify_compact(start_model, engine.ids, calibration_features[:4], engine.predictions[:4])
            metrics, logits = evaluate_features(compact, validation["features"], validation["labels"], args.image_batch_size)
            dense_ids = engine.ids.cpu() if number == 1 else shared_indices[engine.ids.cpu()]
            result.update(pilot=number, method=method, starting_codes=start_model.num_prototypes,
                          validation_accuracy=metrics["accuracy"], validation_cross_entropy=metrics["cross_entropy"],
                          validation_images=metrics["images"],
                          validation_accuracy_change=metrics["accuracy"] - base_metrics["accuracy"],
                          validation_loss_change=metrics["cross_entropy"] - base_metrics["cross_entropy"],
                          nonzero_edges=int((torch.relu(compact.classifier.weight) > 0).sum()),
                          retained_start_ids=engine.ids.cpu().tolist(), retained_dense_ids=dense_ids.tolist(),
                          physical_inference_verification=final_check, config=vars(args))
            checkpoint = directory / "pruned.pth"
            save_checkpoint(checkpoint, compact, result)
            result["checkpoint_reload_verification"] = reload_check(checkpoint, start_model, compact, calibration_features[:4])
            np.savez_compressed(directory / "validation_predictions.npz", img_ids=validation["img_ids"].numpy(),
                                labels=validation["labels"].numpy(), logits=logits.numpy(), preds=logits.argmax(1).numpy())
            write_json(directory / "results.json", result)
            rows.append(result)
            del engine, compact
        del similarities
    summary = {"smoke": args.smoke, "selection_split": "train", "evaluation_split": "val",
               "test_read": False, "retraining_epochs": 0, "starts": starts, "results": rows}
    write_json(output / "results.json", summary)
    fields = ["pilot", "method", "starting_codes", "remaining_codes", "validation_accuracy",
              "validation_accuracy_change", "validation_cross_entropy", "validation_loss_change",
              "calibration_loss_before", "calibration_loss_after", "pruning_seconds", "nonzero_edges",
              "peak_gpu_allocated_mib", "candidate_evaluations", "joint_evaluations", "fallback_count"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = ["# Physical prototype pruning pilot", "", f"Smoke run: {args.smoke}. No retraining. Validation evaluation.", "",
              "| Pilot | Method | Codes | Val top-1 (%) | Val CE | Delta CE | Pruning seconds | Edges | Fallbacks |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        report.append(f"| {row['pilot']} | {row['method']} | {row['remaining_codes']} | "
                      f"{row['validation_accuracy']:.3f} | {row['validation_cross_entropy']:.6f} | "
                      f"{row['validation_loss_change']:+.6f} | {row['pruning_seconds']:.3f} | "
                      f"{row['nonzero_edges']} | {row['fallback_count']} |")
    report += ["", "Delta CE is relative to that pilot's shared starting model.",
               "Pruning time includes selection, cached-state updates, and model compaction.",
               "Shared P1/cache setup, verification, validation, and checkpoint I/O are reported separately or excluded.",
               "Peak GPU memory includes resident model/cache tensors. CPU runs report no GPU measurement."]
    (output / "results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def load_cache(path: Path, dataset, split: str, backbone_sha: str) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing {split} feature cache: {path}. Run scripts/cache_features.py --splits train val.")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("split") != split or data.get("backbone_sha256") != backbone_sha:
        raise ValueError(f"{path} has the wrong split or backbone hash. Rebuild the train/val cache.")
    for name in ("features", "labels", "img_ids"):
        if not isinstance(data.get(name), torch.Tensor):
            raise ValueError(f"{path} lacks tensor '{name}'. Rebuild the feature cache.")
    if tuple(data["features"].shape) != (len(dataset), 768, 7, 7):
        raise ValueError(f"Unexpected feature shape in {path}; expected [{len(dataset)}, 768, 7, 7].")
    if not data["features"].is_floating_point():
        raise ValueError(f"{path} features must be floating-point tensors.")
    if data["img_ids"].tolist() != dataset.img_ids or data["labels"].tolist() != dataset.labels:
        raise ValueError(f"{path} image IDs or labels do not match the current strict {split} split.")
    return {name: data[name] for name in ("features", "labels", "img_ids")}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone_ckpt", required=True, type=Path)
    ap.add_argument("--pq_ckpt", required=True, type=Path)
    ap.add_argument("--cache_dir", required=True, type=Path)
    ap.add_argument("--cub_path", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--calibration_per_class", type=int, default=2)
    ap.add_argument("--shortlist_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--batch_extra_loss_tol", type=float, default=0.005)
    ap.add_argument("--batch_extra_loss_rel", type=float, default=0.25)
    ap.add_argument("--image_batch_size", type=int, default=32)
    ap.add_argument("--candidate_chunk_size", type=int, default=4)
    ap.add_argument("--removals", type=int, default=160)
    ap.add_argument("--second_start", type=int, default=1700)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--smoke", action="store_true")
    return ap


def main() -> None:
    args = parser().parse_args()
    output = args.output_dir.resolve()
    if (output / "state.json").exists():
        raise SystemExit(f"Pilot output already exists: {output}. Choose a new PILOT_OUTPUT_DIR.")
    output.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "started": time.time()}
    write_json(output / "state.json", state)
    try:
        import pq_lib as L

        if L.protocol_info()["protocol"] != "strict":
            raise ValueError("This pilot requires PQ_CUB_PROTOCOL=strict.")
        for name in ("backbone_ckpt", "pq_ckpt"):
            path = getattr(args, name)
            if not path.is_file():
                raise ValueError(f"Missing {name}: {path}. Configure BACKBONE_CKPT and PQ_CKPT.")
        root = args.cub_path / "CUB_200_2011"
        if not (root / "images").is_dir():
            raise ValueError(f"Missing cropped CUB images at {root / 'images'}. Set CUB_PATH to their parent.")
        for name in ("images.txt", "image_class_labels.txt", "train_test_split.txt"):
            if not (root / name).is_file():
                raise ValueError(f"Missing CUB metadata: {root / name}.")
        if not 0 < args.removals < args.second_start < 3000:
            raise ValueError("Require 0 < removals < second_start < 3000.")
        if min(args.calibration_per_class, args.shortlist_size, args.batch_size,
               args.image_batch_size, args.candidate_chunk_size) < 1:
            raise ValueError("Calibration, shortlist, batch, and chunk sizes must be positive.")
        if any(not math.isfinite(v) or v < 0 for v in (args.batch_extra_loss_tol, args.batch_extra_loss_rel)):
            raise ValueError("Batch loss tolerances must be finite and nonnegative.")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable. Check the PyTorch environment and NVIDIA driver.")
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                              "cpu" if args.device == "auto" else args.device)
        if device.type == "cpu" and not args.smoke:
            raise ValueError("Complete pilot requires CUDA. Use --smoke for a small CPU execution check.")
        torch.manual_seed(args.seed)
        train_ds, val_ds = L.dataset(str(args.cub_path), "train"), L.dataset(str(args.cub_path), "val")
        if (len(train_ds), len(val_ds)) != (5394, 600):
            raise ValueError("Expected the existing strict split of 5,394 training and 600 validation images.")
        sha = L.sha256(args.backbone_ckpt)
        train = load_cache(args.cache_dir / "train.pt", train_ds, "train", sha)
        validation = load_cache(args.cache_dir / "val.pt", val_ds, "val", sha)
        model = L.load_protoquant(str(args.backbone_ckpt), str(args.pq_ckpt), device).eval()
        bundle = torch.load(args.pq_ckpt, map_location="cpu", weights_only=True)
        state_dict = bundle.get("model", bundle.get("state_dict", bundle))
        normalized = {}
        for key, value in state_dict.items():
            while key.startswith(("module.", "model_to_wrap.")):
                key = key.split(".", 1)[1]
            if key in normalized:
                raise ValueError(f"Duplicate checkpoint tensor after removing wrapper prefixes: {key}")
            normalized[key] = value
        # The existing loader is permissive; this pilot requires every model tensor to be present.
        model.load_state_dict(normalized, strict=True)
        del bundle, state_dict, normalized
        if model.num_prototypes != 3000:
            raise ValueError(f"Expected dense 3,000-prototype seed-40 checkpoint; found {model.num_prototypes} codes.")
        reference = L.get_backbone(L.load_backbone_classifier(str(args.backbone_ckpt), torch.device("cpu")))
        source_state, loaded_state = reference.state_dict(), model.backbone.state_dict()
        if source_state.keys() != loaded_state.keys() or any(
                not torch.equal(value, loaded_state[key].cpu()) for key, value in source_state.items()):
            raise ValueError("ProtoQuant checkpoint backbone differs from BACKBONE_CKPT/cache. Use matching artifacts.")
        del reference
        # Hashes, IDs and exact backbone equality have passed. Diagnose numerical cache drift separately.
        cache_checks = []
        with torch.no_grad():
            for index in (0, len(train_ds) // 2):
                image, label = train_ds[index]
                direct_features = model.backbone(image[None].to(device)).float().cpu()
                cached_features = train["features"][index:index + 1].float()
                check = compare_cached_features(direct_features, cached_features)
                check.update(image_id=train_ds.img_ids[index], training_index=index)
                if check["passed"]:
                    direct_logits = forward_features(model, direct_features.to(device))
                    cached_logits = forward_features(model, cached_features.to(device))
                    target = torch.tensor([label], device=device)
                    check.update(max_absolute_logit_difference=float((direct_logits - cached_logits).abs().max()),
                                 prediction_matches=bool((direct_logits.argmax(1) == cached_logits.argmax(1)).all()),
                                 cross_entropy_difference=float(F.cross_entropy(cached_logits, target) -
                                                                F.cross_entropy(direct_logits, target)))
                cache_checks.append(check)
                print("[CACHE CHECK] " + json.dumps(check, allow_nan=False), flush=True)
                write_json(output / "cache_feature_checks.json", {
                    "backbone_ckpt": str(args.backbone_ckpt.resolve()), "pq_ckpt": str(args.pq_ckpt.resolve()),
                    "cache_dir": str(args.cache_dir.resolve()), "torch": str(torch.__version__),
                    "cudnn_tf32": torch.backends.cudnn.allow_tf32, "checks": cache_checks,
                })
                if not check["passed"]:
                    raise ValueError("Training cache feature mismatch exceeds the bounded numerical-drift policy. "
                                     f"See {output / 'cache_feature_checks.json'} and verify inputs/preprocessing.")
                if check["reason"] == "bounded_numerical_drift":
                    print(f"[WARN] Image {check['image_id']}: accepted bounded numerical cache drift; "
                          "statistics saved in cache_feature_checks.json.", flush=True)
            # Physical deletion checked against image -> compact-model logits on a real training image.
            image, label = train_ds[0]
            image = image[None].to(device)
            features = model.backbone(image).float()
            probe = DeletionEvaluator(spatial_logits(model, features, 1), model.classifier.weight,
                                      torch.tensor([label]), model.classifier.bias)
            probe.remove([0, 1])
            actual = compact_model(model, probe.ids)(image).logits
            torch.testing.assert_close(actual, probe.predictions, rtol=2e-4, atol=2e-5)
        effective = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        if args.smoke:
            args.removals = min(args.removals, 2)
            args.calibration_per_class = 1
            small = balanced_indices(train["labels"], L.NUM_CLASSES, 1, args.seed)
            train = {key: value[small] for key, value in train.items()}
            validation = {key: value[:32] for key, value in validation.items()}
            print("[SMOKE] 200 training images, 32 validation images, at most 2 removals per method/pilot.", flush=True)
        for key, value in vars(args).copy().items():
            if isinstance(value, Path):
                setattr(args, key, str(value.resolve()))
        write_json(output / "config.json", {"requested": effective, "effective": vars(args),
                   "backbone_sha256": sha, "pq_sha256": L.sha256(args.pq_ckpt), **L.protocol_info(),
                   "torch": torch.__version__, "device": str(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "initial_image_inference_verification": "passed", "cache_feature_checks": cache_checks,
                   "test_data_loaded": False})
        run_pilots(model, train, validation, args, output)
        write_json(output / "state.json", {**state, "status": "completed", "finished": time.time()})
        (output / "DONE").write_text("completed\n", encoding="utf-8")
        print(f"[DONE] Results: {output / 'results.md'}", flush=True)
    except Exception as error:
        write_json(output / "state.json", {**state, "status": "failed", "error": str(error), "finished": time.time()})
        print(f"[ERROR] {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
