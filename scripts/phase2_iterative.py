"""Phase 4 iterative pruning (docs/phase0_protocol.md, section 10), one seed and one score.

    python scripts/phase2_iterative.py --backbone_ckpt B.pth --pq_ckpt <dense Stage 2 .pth> \
        --cache_dir runs/phase1/feature_cache --score P1 --seed 40 --cub_path ... \
        --out_dir runs/phase2/strict_s40/iterative/P1_K10

Rounds, from all nonzero edges of ReLU(W) down to K per class:
  1. rescore (B1: current ReLU(W); P1: current ReLU(W) x E[s | y = k], class means recomputed from the
     training-split feature cache through the current codes);
  2. keep the top max(K, floor(0.8 * n_c)) of each class's current n_c edges;
  3. unless K is reached: retrain --round_epochs epochs with the mask held fixed (full codebook in the softmax).
The final mask and the model it was taken from are saved; the driver then runs phase2_retrain.py on them
(physical removal of unused codes + the B2 recipe), so the last step is identical to B2.

Output in out_dir: prefinal.pth (model state at the final pruning step, full codebook), masks.pt
({"masks": {"<score>_K<K>": mask}, "pq_ckpt": prefinal.pth}), iterative.json (per-round log).
"""

import argparse
import math
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

import phase2_lib as P
import pq_lib as L
from phase2_masking import load_split, pooled_scores
from phase2_retrain import evaluate
from src.config.datasets import CUB200Config
from src.datasets.cub import CUB200
from src.datasets.transforms import get_transforms_from_config
from src.pipnet_utils import TrainingWrapper
from src.utils import SchedulerArgs, create_schedulers, set_reproducibility


def prune_step(score: torch.Tensor, w_eff: torch.Tensor, mask: torch.Tensor, k: int, frac: float) -> torch.Tensor:
    """Per class: among current edges (mask & w > 0), keep the top max(k, floor((1 - frac) * n_c))."""
    eligible = mask & (w_eff > 0)
    n = eligible.sum(1)
    keep = torch.clamp(torch.floor(n.double() * (1 - frac)).long(), min=k)
    keep = torch.minimum(keep, n)
    s = score.masked_fill(~eligible, float("-inf"))
    order = s.argsort(dim=1, descending=True)
    rank = torch.empty_like(order).scatter_(1, order, torch.arange(s.shape[1]).expand_as(order).contiguous())
    return eligible & (rank < keep[:, None])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_ckpt", required=True)
    ap.add_argument("--pq_ckpt", required=True, help="dense Stage 2 checkpoint (Phase 1, strict)")
    ap.add_argument("--cache_dir", required=True, help="Phase 1 feature cache (train split used for P1)")
    ap.add_argument("--score", choices=["B1", "P1"], required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--prune_frac", type=float, default=0.2)
    ap.add_argument("--round_epochs", type=int, default=5)
    ap.add_argument("--round_warmup", type=int, default=1)
    ap.add_argument("--cub_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_workers", type=int, default=2)
    ap.add_argument("--max_train_batches", type=int, default=0, help="smoke tests only")
    args = ap.parse_args()

    if L.protocol_info()["protocol"] != "strict":
        raise SystemExit("Phase 4 runs under PQ_CUB_PROTOCOL=strict")
    set_reproducibility(args.seed)
    dev = L.device()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = L.construct_model_no_cfg("convnext_tiny", L.NUM_CLASSES, dev, args.backbone_ckpt, global_pool="")
    model = L.build_pipnet_model(backbone=L.get_backbone(base), num_classes=L.NUM_CLASSES, device=dev,
                                 pipnet_checkpoint_path=args.pq_ckpt, train_codebook=True)
    if args.score == "P1":
        train = load_split(Path(args.cache_dir), "train", L.sha256(args.backbone_ckpt))

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
    mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode="batch",
                  label_smoothing=0.1, num_classes=L.NUM_CLASSES)
    criterion = SoftTargetCrossEntropy()

    w0 = L.effective_weight(model).cpu()
    mask = w0 > 0
    n0 = mask.sum(1).double()
    est = math.ceil(math.log(args.k / n0.mean().item()) / math.log(1 - args.prune_frac))
    print(f"[ITER] {args.score}: start {int(mask.sum())} edges ({n0.mean():.0f}/class), "
          f"~{est} rounds to K={args.k}", flush=True)
    rounds, total_epochs, r = [], 0, 0
    while True:
        r += 1
        t0 = time.time()
        model.eval()
        w = L.effective_weight(model).cpu()
        if args.score == "P1":
            s = pooled_scores(model, train["features"], dev)
            mu_in, mu_out = P.class_means(s, train["labels"], L.NUM_CLASSES)
            score = P.edge_scores(w, mu_in, mu_out)["P1"]
        else:
            score = w.double()
        mask = prune_step(score, w, mask, args.k, args.prune_frac)
        with torch.no_grad():
            model.classifier.weight.mul_(mask.to(dev).float())
        per_class = mask.sum(1)
        final = bool((per_class <= args.k).all())
        val_pruned = evaluate(wrapper, val_dl, dev)
        info = {"round": r, "edges": int(mask.sum()), "edges_per_class_max": int(per_class.max()),
                "active_codes": int(mask.any(0).sum()), "val_after_pruning": val_pruned}
        if not final:
            maskf = mask.to(dev).float()
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                    weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
            sched = create_schedulers([opt], SchedulerArgs(epochs=args.round_epochs,
                                                           warmup_epochs=args.round_warmup, lr=args.lr))[0]
            for epoch in range(args.round_epochs):
                wrapper.train()
                for i, (x, yb) in enumerate(train_dl):
                    if args.max_train_batches and i >= args.max_train_batches:
                        break
                    x, yb = mixup(x.to(dev, non_blocking=True), yb.to(dev, non_blocking=True))
                    opt.zero_grad()
                    criterion(wrapper(x), yb).backward()
                    opt.step()
                    with torch.no_grad():
                        model.classifier.weight.mul_(maskf)
                sched.step(epoch)
            total_epochs += args.round_epochs
            info["val_after_round_training"] = evaluate(wrapper, val_dl, dev)
        info["seconds"] = time.time() - t0
        rounds.append(info)
        print(f"[ITER] {info}", flush=True)
        if final:
            break

    key = f"{args.score}_K{args.k}"
    prefinal = out / "prefinal.pth"
    torch.save({"model": model.state_dict(), "mask": mask, "rounds": len(rounds)}, prefinal)
    torch.save({"masks": {key: mask}, "pq_ckpt": str(prefinal), "seed": args.seed}, out / "masks.pt")
    L.write_json(out / "iterative.json", {
        "score": args.score, "k": args.k, "seed": args.seed, "rounds": len(rounds),
        "round_training_epochs": total_epochs, "final_retrain_epochs": 30,
        "total_training_epochs": total_epochs + 30, "start_edges": int(w0.gt(0).sum()),
        "final_edges": int(mask.sum()), "final_active_codes": int(mask.any(0).sum()), "history": rounds,
        "source_pq_ckpt": args.pq_ckpt, **{k: v for k, v in vars(args).items() if k != "pq_ckpt"},
    })


if __name__ == "__main__":
    main()
