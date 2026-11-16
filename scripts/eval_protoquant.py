"""Evaluate a final model on one CUB split and save per-image predictions.

    python scripts/eval_protoquant.py --backbone_ckpt B.pth --pq_ckpt pipnet_*.pth --split test --out runs/.../eval_test
    python scripts/eval_protoquant.py --backbone_ckpt B.pth --split test --out ...   # black-box backbone

Writes <out>.json (top-1/top-5, capacity counts, protocol) and <out>.npz (img_ids, labels, preds, logits),
which Phase 3 uses for the paired bootstrap.

Strict protocol: the test set is read once per final model. A marker file <ckpt>.test_read_strict.json is
written after the first test read; a second read is refused unless --allow_test_reread is given (and
then it is recorded in the marker).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import pq_lib as L


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone_ckpt", required=True)
    p.add_argument("--pq_ckpt", default="", help="Stage 2 checkpoint; omit to evaluate the backbone's own head")
    p.add_argument("--split", choices=["train", "val", "test", "train_full"], required=True)
    p.add_argument("--cub_path", required=True)
    p.add_argument("--out", required=True, help="output prefix (writes .json and .npz)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--allow_test_reread", action="store_true")
    args = p.parse_args()

    dev = L.device()
    ckpt = Path(args.pq_ckpt or args.backbone_ckpt)
    protocol = L.protocol_info()["protocol"]
    marker = ckpt.with_name(f"{ckpt.name}.test_read_{protocol}.json")
    strict = protocol == "strict"
    if args.split == "test" and strict and marker.exists() and not args.allow_test_reread:
        raise SystemExit(
            f"Test set already read for {ckpt} (see {marker}). Strict protocol: test is read once per "
            "final model. Pass --allow_test_reread only for a documented reason."
        )

    if args.pq_ckpt:
        model = L.load_protoquant(args.backbone_ckpt, args.pq_ckpt, dev)
        forward = lambda x: model(x).logits  # noqa: E731
        stats = L.head_stats(model)
        kind = "protoquant"
    else:
        model = L.load_backbone_classifier(args.backbone_ckpt, dev)
        forward = model
        stats = {}
        kind = "backbone"

    ds = L.dataset(args.cub_path, args.split)
    dl = L.loader(ds, args.batch_size, args.num_workers)
    all_logits, all_labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for x, y in dl:
            all_logits.append(forward(x.to(dev, non_blocking=True)).float().cpu())
            all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = logits.argmax(1)
    top5 = logits.topk(5, dim=1).indices
    top1_acc = (preds == labels).float().mean().item() * 100
    top5_acc = (top5 == labels[:, None]).any(1).float().mean().item() * 100

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out.with_suffix(".npz"), img_ids=np.array(ds.img_ids), labels=labels.numpy(),
        preds=preds.numpy(), logits=logits.numpy().astype(np.float32),
    )
    result = {
        "kind": kind, "split": args.split, "n_images": len(ds),
        "top1": top1_acc, "top5": top5_acc, "n_correct": int((preds == labels).sum()),
        "backbone_ckpt": args.backbone_ckpt, "pq_ckpt": args.pq_ckpt,
        "seconds": time.time() - t0, "device": str(dev), **stats, **L.protocol_info(),
    }
    L.write_json(out.with_suffix(".json"), result)
    print(f"[EVAL] {kind} split={args.split} n={len(ds)} top1={top1_acc:.2f}% top5={top5_acc:.2f}%")

    if args.split == "test":
        history = json.loads(marker.read_text())["reads"] if marker.exists() else []
        history.append({"time": time.ctime(),
                        "out": str(out), "top1": top1_acc, "protocol": result["protocol"]})
        marker.write_text(json.dumps({"ckpt": str(ckpt), "reads": history}, indent=2))


if __name__ == "__main__":
    main()
