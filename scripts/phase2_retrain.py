"""Phase 2, retraining after pruning (B2 and the K = 10 retrains), one GPU, no DDP.

    python scripts/phase2_retrain.py --backbone_ckpt B.pth --pq_ckpt <dense Stage 2 .pth> \
        --masks runs/phase2/strict_s40/masking/masks.pt --mask_key P1_K10 --seed 40 \
        --cub_path ... --out_dir runs/phase2/strict_s40/retrain/P1_K10

Recipe (the same for every method, so only the kept edges differ):
  1. load the dense Stage 2 model and apply the mask (edges outside it set to zero);
  2. physically remove codes that no class uses any more (prune_inactive_prototypes), as in the
     paper's "Pruned & Tuned" -- B2 is this recipe with the B1 mask;
  3. retrain codes + head with the Stage 2 recipe (AdamW lr 0.05, wd 5e-4, cosine schedule with warm-up,
     MixUp/CutMix, label smoothing 0.1, 30 epochs), the mask held fixed: pruned edges are re-zeroed
     after every optimizer step, so each class keeps at most its K edges;
  4. keep the best epoch on the held-out validation split.
Output: <out_dir>/pipnet_retrained.pth (loadable by eval_protoquant.py) and retrain.json.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

import pq_lib as L
from src.config.datasets import CUB200Config
from src.datasets.cub import CUB200
from src.datasets.transforms import get_transforms_from_config
from src.pipnet_utils import TrainingWrapper
from src.utils import SchedulerArgs, create_schedulers, set_reproducibility


def evaluate(wrapper, dl, dev) -> float:
    wrapper.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dl:
            pred = wrapper(x.to(dev, non_blocking=True)).argmax(1).cpu()
            correct += int((pred == y).sum())
            total += len(y)
    return 100.0 * correct / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_ckpt", required=True)
    ap.add_argument("--pq_ckpt", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--mask_key", required=True, help="e.g. B1_K10 (= B2), P1_K10, P3_K10")
    ap.add_argument("--cub_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_workers", type=int, default=2)
    ap.add_argument("--no_physical_removal", action="store_true", help="keep every code in the softmax")
    ap.add_argument("--max_train_batches", type=int, default=0, help="smoke tests only")
    args = ap.parse_args()

    if L.protocol_info()["protocol"] != "strict":
        raise SystemExit("Phase 2 runs under PQ_CUB_PROTOCOL=strict")
    set_reproducibility(args.seed)
    dev = L.device()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- model + mask ----------------------------------------------------------------------------
    base = L.construct_model_no_cfg("convnext_tiny", L.NUM_CLASSES, dev, args.backbone_ckpt, global_pool="")
    model = L.build_pipnet_model(backbone=L.get_backbone(base), num_classes=L.NUM_CLASSES, device=dev,
                                 pipnet_checkpoint_path=args.pq_ckpt, train_codebook=True)
    saved = torch.load(args.masks, map_location="cpu")
    if Path(saved["pq_ckpt"]).name != Path(args.pq_ckpt).name:
        raise SystemExit("masks.pt was built from a different Stage 2 checkpoint")
    mask = saved["masks"][args.mask_key].to(dev)
    with torch.no_grad():
        model.classifier.weight.mul_(mask)
    codes_before = model.num_prototypes
    if not args.no_physical_removal:
        active = mask.any(0)
        model.prune_inactive_prototypes(min_weight=0.0)
        if model.num_prototypes != int(active.sum()):
            raise SystemExit("physical removal kept a different set of codes than the mask")
        mask = mask[:, active]
    maskf = mask.float()
    print(f"[RETRAIN] {args.mask_key}: codes {codes_before} -> {model.num_prototypes}, "
          f"edges {int(mask.sum())}", flush=True)

    # ---- data (strict protocol: train 5,394 / val 600) ------------------------------------------
    dcfg = OmegaConf.structured(CUB200Config(_path=args.cub_path))
    dcfg.autoaugment = False
    dcfg.use_raw_transforms = False
    train_tf, val_tf = get_transforms_from_config(dcfg, model_name="convnext_tiny")
    train_ds = CUB200(root=args.cub_path, split="train", transform=train_tf)
    val_ds = CUB200(root=args.cub_path, split="val", transform=val_tf)
    g = torch.Generator().manual_seed(args.seed)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
                                           num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                           persistent_workers=args.num_workers > 0)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=min(args.batch_size, 256), shuffle=False,
                                         num_workers=args.val_workers, pin_memory=True, drop_last=False,
                                         persistent_workers=args.val_workers > 0)
    print(f"[CUB200] protocol=strict | train {len(train_ds)} | val {len(val_ds)}", flush=True)

    wrapper = TrainingWrapper(model)
    val_before = evaluate(wrapper, val_dl, dev)
    print(f"[RETRAIN] val top-1 before retraining: {val_before:.2f}", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    scheduler = create_schedulers([optimizer], SchedulerArgs(epochs=args.epochs, warmup_epochs=args.warmup_epochs,
                                                             lr=args.lr))[0]
    mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode="batch",
                  label_smoothing=0.1, num_classes=L.NUM_CLASSES)
    criterion = SoftTargetCrossEntropy()

    best, history = -1.0, []
    ckpt_path = out / "pipnet_retrained.pth"
    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"Epoch: {epoch}", flush=True)
        wrapper.train()
        for i, (x, yb) in enumerate(train_dl):
            if args.max_train_batches and i >= args.max_train_batches:
                break
            x, yb = mixup(x.to(dev, non_blocking=True), yb.to(dev, non_blocking=True))
            optimizer.zero_grad()
            loss = criterion(wrapper(x), yb)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.classifier.weight.mul_(maskf)   # pruned edges stay pruned
        val = evaluate(wrapper, val_dl, dev)
        scheduler.step(epoch)
        edges = int((torch.relu(model.classifier.weight) > 0).sum())
        history.append({"epoch": epoch, "val_top1": val, "edges": edges, "seconds": time.time() - t0})
        print({"Validation Top1 Accuracy": val, "edges": edges, "loss": loss.item()}, flush=True)
        if val > best:
            best = val
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_top1": val, "mask": mask.cpu(),
                        "mask_key": args.mask_key, "source_pq_ckpt": args.pq_ckpt}, ckpt_path)
    print(f"[RETRAIN] best val top-1 {best:.2f}", flush=True)
    L.write_json(out / "retrain.json", {
        "mask_key": args.mask_key, "seed": args.seed, "codes_before": codes_before,
        "codes_after": int(model.num_prototypes), "physical_removal": not args.no_physical_removal,
        "val_before_retraining": val_before, "best_val_top1": best,
        "best_epoch": max(history, key=lambda h: h["val_top1"])["epoch"], "history": history,
        "checkpoint": str(ckpt_path), **{k: v for k, v in vars(args).items() if k not in ("masks",)},
    })


if __name__ == "__main__":
    main()
