"""Shared helpers for the Phase 1 scripts (evaluation, feature cache, implementation tests).

Everything here runs in one process on one device (CUDA if available, else CPU), outside the
repository's Hydra/DDP pipelines, so it also works for small checks on a laptop.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "codebook_train"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.config.datasets import Cars196Config, CUB200Config  # noqa: E402
from src.construct_model import construct_model_no_cfg, get_backbone  # noqa: E402
from src.datasets.cub import CUB200, Cars196, get_protocol, get_split_file  # noqa: E402
from src.datasets.transforms import get_transforms_from_config  # noqa: E402
from src.pipnet_utils import build_pipnet_model  # noqa: E402

# Dataset switch for every script (eval, cache, tests, retraining): PQ_DATASET=cub200 (default) or cars196.
DATASETS = {"cub200": (CUB200, CUB200Config, 200), "cars196": (Cars196, Cars196Config, 196)}
DATASET = os.environ.get("PQ_DATASET", "cub200").strip().lower()
if DATASET not in DATASETS:
    raise SystemExit(f"PQ_DATASET must be one of {list(DATASETS)}, got '{DATASET}'")
DATASET_CLASS, DATASET_CONFIG, NUM_CLASSES = DATASETS[DATASET]


def dataset_config(path: str):
    """The Hydra dataset config of the active dataset (transforms as in the training pipelines)."""
    cfg = OmegaConf.structured(DATASET_CONFIG(_path=path))
    cfg.autoaugment = False
    cfg.use_raw_transforms = False
    return cfg


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_transform(cub_path: str, model_name: str = "convnext_tiny"):
    """The deterministic (validation) transform the training scripts use for CUB."""
    return get_transforms_from_config(dataset_config(cub_path), model_name=model_name)[1]


def dataset(cub_path: str, split: str, model_name: str = "convnext_tiny") -> CUB200:
    if split == "test":
        print(f"[{DATASET_CLASS.TAG}] protocol={get_protocol()} | TEST split requested", flush=True)
    return DATASET_CLASS(root=cub_path, split=split, transform=eval_transform(cub_path, model_name))


def loader(ds, batch_size: int, num_workers: int):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )


def load_backbone_classifier(backbone_ckpt: str, dev, model_name: str = "convnext_tiny"):
    """The fine-tuned backbone with its own linear head (the black-box baseline)."""
    model = construct_model_no_cfg(model_name, NUM_CLASSES, dev, backbone_ckpt, global_pool="")
    return model.eval()


def load_protoquant(backbone_ckpt: str, pq_ckpt: str, dev, model_name: str = "convnext_tiny"):
    """Frozen backbone + saved Stage 2 head (ProtoQuantNet), T = 0.1."""
    base = construct_model_no_cfg(model_name, NUM_CLASSES, dev, backbone_ckpt, global_pool="")
    model = build_pipnet_model(
        backbone=get_backbone(base), num_classes=NUM_CLASSES, device=dev,
        pipnet_checkpoint_path=pq_ckpt,
    )
    return model.eval()


@torch.no_grad()
def scores_from_features(model, feats: torch.Tensor) -> torch.Tensor:
    """Pooled code activations s (B, M) from backbone features (B, C, H, W): max over positions of
    softmax_m(cos(z, c_m) / T)."""
    sim = model._detect_prototypes(feats.float())
    return model.pool_layer(F.softmax(sim / model.temperature, dim=1))


def effective_weight(model) -> torch.Tensor:
    """ReLU(W), the classifier the model actually uses (C, M)."""
    return torch.relu(model.classifier.weight.detach())


def logits_from_scores(s: torch.Tensor, w_eff: torch.Tensor) -> torch.Tensor:
    return s @ w_eff.t()


def head_stats(model) -> dict:
    w = effective_weight(model)
    per_class = (w > 0).sum(dim=1).float()
    codes = model.codes.detach()
    return {
        "codebook_entries": int(codes.shape[0]),
        "codebook_mb_fp32": codes.numel() * 4 / 2**20,
        "nonzero_edges": int((w > 0).sum().item()),
        "active_codes": int((w > 0).any(dim=0).sum().item()),
        "edges_per_class_mean": per_class.mean().item(),
        "edges_per_class_min": int(per_class.min().item()),
        "edges_per_class_max": int(per_class.max().item()),
    }


def protocol_info() -> dict:
    split_file = get_split_file(DATASET_CLASS.SPLIT_ENV, DATASET_CLASS.DEFAULT_SPLIT)
    return {
        "dataset": DATASET,
        "protocol": get_protocol(),
        "split_file": str(split_file),
        "split_sha256": sha256(split_file) if split_file.is_file() else None,
    }


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"written": datetime.now().isoformat(timespec="seconds"), **obj}
    path.write_text(json.dumps(obj, indent=2))
    print(f"wrote {path}")


def env_flag(name: str) -> bool:
    return os.environ.get(name, "") in ("1", "true", "True", "yes")
