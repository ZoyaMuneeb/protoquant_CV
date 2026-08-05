"""Project ProtoQuant codes to nearest training-set features and save a projected checkpoint.

Usage (example):
 python codebook_train/project_prototypes.py \
   --proto-ckpt /path/to/pipnet_checkpoint.pth \
   --backbone-name convnext_tiny \
   --backbone-ckpt /path/to/backbone_ckpt.pth \
   --dataset cub200 --dataset-path /path/to/datasets --out-ckpt ./proj.pth \
   --batch-size 64 --num-workers 4 --device cuda:0 \
   --eval --baseline-ckpt /path/to/pipnet_checkpoint.pth

This script reuses dataset and model utilities from the repository. It:
 - scans the training set through the frozen backbone to collect spatial features
 - finds for each code the nearest training spatial feature (cosine similarity)
 - replaces the 'codes' tensor in the original ProtoQuant checkpoint with the matched
   training feature vectors and writes a new checkpoint file.
 - optionally (--eval) constructs the full ProtoQuantNet (backbone + codebook +
   interpretable head) for both the baseline and projected checkpoints and reports
   top-1 / top-5 accuracy deltas.

The script is intentionally self-contained and conservative: it does not modify other
checkpoint fields and preserves the original checkpoint layout when possible.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.construct_model import construct_model_no_cfg, get_backbone
from src.datasets.construct_dataset import get_dataset
from src.datasets.transforms import get_default_image_transforms
from src.training import extract_features_from_backbone
from src.models.proto_quantnet import ProtoQuantNet

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_codes_in_ckpt(ckpt_obj: object):
    """Return (container, key, codes_tensor) where codes tensor lives.

    Args:
        ckpt_obj: Loaded checkpoint object (dict or state_dict-like).

    Returns:
        Tuple of (container_dict, key_str, codes_tensor)
    """
    if isinstance(ckpt_obj, dict):
        # Common nested keys
        for candidate in ("model", "state_dict", "model_state_dict"):
            if candidate in ckpt_obj and isinstance(ckpt_obj[candidate], dict):
                inner = ckpt_obj[candidate]
                if "codes" in inner:
                    logger.info(f"Found 'codes' under nested key: {candidate}")
                    return (ckpt_obj, candidate, inner["codes"])
        # Top-level
        if "codes" in ckpt_obj:
            logger.info("Found 'codes' at top level")
            return (ckpt_obj, "codes", ckpt_obj["codes"])
    raise ValueError("Could not find 'codes' tensor in checkpoint")


def _extract_full_state_dict(ckpt_obj: object) -> dict:
    """Return the full state_dict (all keys, not just 'codes') from a checkpoint object.

    Mirrors the nested-key search in find_codes_in_ckpt, but returns the whole dict
    instead of just the codes tensor, so it can be loaded into ProtoQuantNet.
    """
    if isinstance(ckpt_obj, dict):
        for candidate in ("model", "state_dict", "model_state_dict"):
            if candidate in ckpt_obj and isinstance(ckpt_obj[candidate], dict):
                logger.info(f"Using state_dict under nested key: {candidate}")
                return ckpt_obj[candidate]
        # Assume ckpt_obj itself is already a flat state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise ValueError(
        "Could not locate a usable state_dict in checkpoint (expected a nested "
        "'model'/'state_dict'/'model_state_dict' key or a flat tensor dict)."
    )


def save_matches(matches, out_json: str):
    # convert tensor paths and ints to python-native
    serial = {}
    for k, v in matches.items():
        serial[k] = {
            "train_index": int(v["train_index"]),
            "spatial_idx": int(v["spatial_idx"]),
            "similarity": float(v["similarity"]),
            "feature_index": int(v["feature_index"]),
        }
    with open(out_json, "w") as f:
        json.dump(serial, f, indent=2)


def save_tensor_artifact(tensor: torch.Tensor, out_path: str) -> None:
    """Save a single tensor artifact to disk with a stable filename."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, out)


def _build_protoquant_model(
    ckpt_path: str,
    backbone_name: str,
    backbone_ckpt_path: str,
    num_classes: int,
    device: torch.device,
    temperature: float,
    bias: bool,
) -> ProtoQuantNet:
    """Construct a ProtoQuantNet and load a ProtoQuant checkpoint's weights into it.

    Unlike construct_model_no_cfg (which builds a plain backbone + linear head and
    will fail with strict=True on a ProtoQuant checkpoint's keys), this builds the
    actual architecture the checkpoint was trained with: frozen backbone + codebook
    + non-negative prototype classifier.
    """
    # Backbone weights come from backbone_ckpt_path (the frozen pretrained backbone),
    # NOT from the ProtoQuant checkpoint — the ProtoQuant checkpoint only adds codes
    # and the classifier head on top of this frozen backbone.
    base_model = construct_model_no_cfg(
        model_name=backbone_name,
        num_classes=num_classes,
        device=device,
        checkpoint_path=backbone_ckpt_path,
    )
    backbone = get_backbone(base_model)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    _, _, codes = find_codes_in_ckpt(ckpt)

    model = ProtoQuantNet(
        backbone=backbone,
        codes=codes,
        num_classes=num_classes,
        temperature=temperature,
        bias=bias,
        freeze_backbone=True,
    ).to(device)

    full_state_dict = _extract_full_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(full_state_dict, strict=False)
    if missing:
        logger.warning(f"[{ckpt_path}] Missing keys when loading state_dict: {missing}")
    if unexpected:
        logger.warning(f"[{ckpt_path}] Unexpected keys when loading state_dict: {unexpected}")

    model.eval()
    return model


@torch.no_grad()
def _validate_protoquant(
    model: ProtoQuantNet, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Compute top-1 / top-5 accuracy for a ProtoQuantNet.

    ProtoQuantNet.forward returns a ProtoQuantOutput dataclass, not a raw logits
    tensor, so this can't reuse a generic validate_epoch written for plain classifiers.
    """
    model.eval()
    correct1, correct5, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images).logits
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)
    return 100.0 * correct1 / total, 100.0 * correct5 / total


def eval_checkpoints(
    baseline_ckpt_path: str,
    projected_ckpt_path: str,
    backbone_name: str,
    backbone_ckpt_path: str,
    dataset_name: str,
    dataset_path: str,
    num_classes: int,
    device: torch.device,
    temperature: float,
    bias: bool,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict:
    """Evaluate baseline and projected ProtoQuant checkpoints using the same val setup.

    Both checkpoints are loaded into the *actual* ProtoQuantNet architecture
    (frozen backbone + codebook + interpretable head) — not a plain backbone
    classifier — so accuracy reflects the real interpretable model.

    Args:
        baseline_ckpt_path: Path to original (unprojected) ProtoQuant checkpoint.
        projected_ckpt_path: Path to projected checkpoint.
        backbone_name: Name of backbone model (e.g., "convnext_tiny").
        backbone_ckpt_path: Path to frozen backbone checkpoint.
        dataset_name: Dataset name (e.g., "cub200").
        dataset_path: Path to dataset.
        num_classes: Number of classes (computed once in main(), passed through so
            baseline/projected/feature-extraction all agree on the same value).
        device: Device to run evaluation on.
        temperature: Softmax temperature used by ProtoQuantNet's concept pooling.
            MUST match what training actually used, or accuracy won't reproduce
            the paper's numbers even with a correctly-loaded model.
        bias: Whether the classifier head used a bias term during training.
        batch_size: Evaluation batch size.
        num_workers: Number of data loader workers.

    Returns:
        dict with keys: baseline_top1, baseline_top5, projected_top1, projected_top5,
        delta_top1, delta_top5.
    """
    logger.info("=" * 50)
    logger.info("Evaluation: Baseline vs. Projected Checkpoints")
    logger.info("=" * 50)

    train_transform, val_transform = get_default_image_transforms(
        resize_value=224,
        crop_value=None,
        random_erase=None,
        horizontal_flip=None,
        is_precropped=True,
        autoaugment=False,
    )

    _, val_dataset = get_dataset(
        name=dataset_name,
        path=dataset_path,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    logger.info(f"Loading baseline checkpoint: {baseline_ckpt_path}")
    baseline_model = _build_protoquant_model(
        ckpt_path=baseline_ckpt_path,
        backbone_name=backbone_name,
        backbone_ckpt_path=backbone_ckpt_path,
        num_classes=num_classes,
        device=device,
        temperature=temperature,
        bias=bias,
    )
    baseline_top1, baseline_top5 = _validate_protoquant(baseline_model, val_loader, device)
    logger.info(f"Baseline - Top-1: {baseline_top1:.2f}%, Top-5: {baseline_top5:.2f}%")

    logger.info(f"Loading projected checkpoint: {projected_ckpt_path}")
    projected_model = _build_protoquant_model(
        ckpt_path=projected_ckpt_path,
        backbone_name=backbone_name,
        backbone_ckpt_path=backbone_ckpt_path,
        num_classes=num_classes,
        device=device,
        temperature=temperature,
        bias=bias,
    )
    projected_top1, projected_top5 = _validate_protoquant(projected_model, val_loader, device)
    logger.info(f"Projected - Top-1: {projected_top1:.2f}%, Top-5: {projected_top5:.2f}%")

    delta_top1 = projected_top1 - baseline_top1
    delta_top5 = projected_top5 - baseline_top5

    logger.info("=" * 50)
    logger.info(f"Delta Top-1: {delta_top1:+.2f}%")
    logger.info(f"Delta Top-5: {delta_top5:+.2f}%")
    logger.info("=" * 50)

    return {
        "baseline_top1": baseline_top1,
        "baseline_top5": baseline_top5,
        "projected_top1": projected_top1,
        "projected_top5": projected_top5,
        "delta_top1": delta_top1,
        "delta_top5": delta_top5,
    }


def _find_nearest_codes(
    codes: torch.Tensor,
    features_flat: torch.Tensor,
    device: torch.device,
    chunk_size: int = 131072,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the nearest feature vector for each code using chunked cosine similarity.

    This avoids materializing the full [num_codes x num_features] matrix and keeps the
    compute device-consistent by moving only one feature chunk at a time to `device`.
    """
    codes_norm = F.normalize(codes.to(device), dim=1)
    features_norm = F.normalize(features_flat, dim=1)

    best_sims = torch.full(
        (codes_norm.shape[0],), float("-inf"), device=device, dtype=codes_norm.dtype
    )
    best_indices = torch.full(
        (codes_norm.shape[0],), -1, device=device, dtype=torch.long
    )

    total_features = features_norm.shape[0]
    logger.info(
        f"Running chunked nearest-neighbor search over {total_features} features with chunk_size={chunk_size}"
    )

    for start in range(0, total_features, chunk_size):
        end = min(start + chunk_size, total_features)
        chunk = features_norm[start:end].to(device, non_blocking=True)
        sims = torch.matmul(codes_norm, chunk.T)
        chunk_best_sims, chunk_best_rel_idx = sims.max(dim=1)
        improved = chunk_best_sims > best_sims
        if improved.any():
            best_sims[improved] = chunk_best_sims[improved]
            best_indices[improved] = start + chunk_best_rel_idx[improved]

        if start == 0 or (start // chunk_size) % 10 == 0:
            logger.info(f"Processed feature chunk {start}-{end} / {total_features}")

    return best_sims, best_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto-ckpt", required=True)
    parser.add_argument("--backbone-name", required=True)
    parser.add_argument("--backbone-ckpt", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--out-ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--matches-out", default="prototype_matches.json")

    # --- Eval-related args ---
    parser.add_argument(
        "--eval", action="store_true",
        help="Run evaluation comparing baseline vs projected checkpoint after projection.",
    )
    parser.add_argument(
        "--baseline-ckpt", default=None,
        help="Path to baseline (unprojected) ProtoQuant checkpoint. Required if --eval is set.",
    )
    # TODO(you): confirm these two match what training actually used. If the
    # checkpoint's dict has a saved config (e.g. ckpt["args"]), prefer pulling
    # temperature/bias from there instead of these CLI defaults, since a mismatch
    # here will silently change accuracy for reasons unrelated to projection.
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--bias", action="store_true", default=False)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.eval and args.baseline_ckpt is None:
        raise ValueError("--baseline-ckpt is required when --eval is specified")

    # load checkpoint object
    ckpt = torch.load(args.proto_ckpt, map_location="cpu")
    container, key, codes_tensor = find_codes_in_ckpt(ckpt)
    codes = codes_tensor.clone().to(device)
    num_codes, code_dim = codes.shape
    print(f"Found codes tensor: {num_codes} x {code_dim}")

    # prepare dataset and transforms
    # CUB-200 images in this workspace are variable-sized crops, so use a fixed resize
    # pipeline to avoid DataLoader collation failures.
    train_transform, val_transform = get_default_image_transforms(
        resize_value=224,
        crop_value=None,
        is_precropped=True,
        random_erase=None,
        horizontal_flip=None,
    )

    train_ds, _ = get_dataset(
        name=args.dataset,
        path=args.dataset_path,
        train_transform=train_transform,
        val_transform=val_transform,
    )
    # Try to infer num_classes
    try:
        labels = getattr(train_ds, "labels", None)
        if labels is not None:
            num_classes = int(max(labels)) + 1
        else:
            num_classes = len(set([train_ds[i][1] for i in range(len(train_ds))]))
    except Exception:
        num_classes = 1000
    print(f"Inferred num_classes={num_classes}")

    base_model = construct_model_no_cfg(
        model_name=args.backbone_name,
        num_classes=num_classes,
        device=device,
        checkpoint_path=args.backbone_ckpt,
    )
    backbone = get_backbone(base_model)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    # extract features (N, D, H, W)
    features_tensor, _ = extract_features_from_backbone(
        feature_backbone=backbone,
        dataloader=train_loader,
        device=device,
        transforms=None,
    )

    # features_tensor shape: [N, D, H, W]
    N, D, H, W = features_tensor.shape
    features_flat = features_tensor.permute(0, 2, 3, 1).reshape(-1, D)  # (N*H*W, D)
    print(f"Collected features: {features_flat.shape}")

    if args.top_k != 1:
        logger.warning(
            "--top-k is currently used only for compatibility; projection snaps each code to its single nearest training feature."
        )

    topk_vals, topk_idx = _find_nearest_codes(
        codes=codes,
        features_flat=features_flat,
        device=device,
    )

    matches = {}
    new_codes = []
    matched_feature_indices: list[int] = []
    matched_feature_similarities: list[float] = []

    for i in range(num_codes):
        idx0 = int(topk_idx[i].item())
        sim0 = float(topk_vals[i].item())
        img_idx = idx0 // (H * W)
        spatial_idx = idx0 % (H * W)
        feat_vec = features_flat[idx0].cpu()
        new_codes.append(feat_vec.unsqueeze(0))
        matched_feature_indices.append(idx0)
        matched_feature_similarities.append(sim0)
        matches[i] = {
            "train_index": int(img_idx),
            "spatial_idx": int(spatial_idx),
            "similarity": sim0,
            "feature_index": idx0,
        }

    new_codes_tensor = torch.cat(new_codes, dim=0)
    assert new_codes_tensor.shape == (num_codes, D)

    # Report similarity distribution immediately — this is the direct "how anchored
    # are the codes already" measurement, independent of any accuracy delta.
    sims_tensor = torch.tensor(matched_feature_similarities)
    print(
        f"Code-to-nearest-training-feature cosine similarity: "
        f"mean={sims_tensor.mean():.4f} median={sims_tensor.median():.4f} "
        f"min={sims_tensor.min():.4f} p10={sims_tensor.quantile(0.1):.4f}"
    )

    # Replace codes in checkpoint container
    if key == "codes":
        container["codes"] = new_codes_tensor.cpu()
    else:
        container[key]["codes"] = new_codes_tensor.cpu()

    out_path = Path(args.out_ckpt)
    torch.save(ckpt, out_path)
    print(f"Saved projected checkpoint to {out_path}")
    print(f"Output directory: {out_path.parent}")

    save_tensor_artifact(new_codes_tensor.cpu(), str(out_path.with_name("projected_codes.pt")))
    save_tensor_artifact(
        torch.tensor(matched_feature_indices, dtype=torch.long),
        str(out_path.with_name("matched_feature_indices.pt")),
    )
    save_tensor_artifact(
        sims_tensor.float(),
        str(out_path.with_name("matched_feature_similarities.pt")),
    )

    save_matches(matches, args.matches_out)
    print(f"Saved prototype -> train matches to {args.matches_out}")

    if args.eval:
        logger.info("\n\n" + "=" * 60)
        eval_results = eval_checkpoints(
            baseline_ckpt_path=args.baseline_ckpt,
            projected_ckpt_path=str(out_path),
            backbone_name=args.backbone_name,
            backbone_ckpt_path=args.backbone_ckpt,
            dataset_name=args.dataset,
            dataset_path=args.dataset_path,
            num_classes=num_classes,
            device=device,
            temperature=args.temperature,
            bias=args.bias,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        eval_results_path = out_path.with_name("eval_results.json")
        with open(eval_results_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Saved evaluation results to {eval_results_path}")
        logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()