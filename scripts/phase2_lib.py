"""Phase 2 pruning scores and masks.

All scores and sparsity counts use the effective classifier w = ReLU(W) (C x M). A mask is a boolean
(C x M) tensor of kept class-code edges; only edges with w > 0 can be kept.

Methods (methodology, Phase 2):
  B1          top-K per class by weight magnitude w_km (the paper's baseline, limit_prototypes)
  random      K random nonzero edges per class (control)
  P1          top-K per class by I_km = w_km * E[s_m | y = k]   (activation-aware, Wanda-like)
  P1contrast  top-K per class by w_km * (E[s_m | y = k] - E[s_m | y != k])   (control)
  P3          global budget: the N = K * C edges with the highest P1 score overall, every class keeps >= 1
E[.] is over the strict training split, from deterministic cached features.
"""

from __future__ import annotations

import torch

METHODS = ("B1", "random", "P1", "P1contrast", "P3")


def class_means(scores: torch.Tensor, labels: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean pooled activation per class, E[s | y = k] (C x M), and over all other classes, E[s | y != k]."""
    scores = scores.double()
    sums = torch.zeros(num_classes, scores.shape[1], dtype=torch.float64).index_add_(0, labels, scores)
    counts = torch.bincount(labels, minlength=num_classes).double()
    if (counts == 0).any():
        raise ValueError("every class needs training images for P1")
    mu_in = sums / counts[:, None]
    mu_out = (sums.sum(0, keepdim=True) - sums) / (counts.sum() - counts)[:, None]
    return mu_in, mu_out


def edge_scores(w_eff: torch.Tensor, mu_in: torch.Tensor, mu_out: torch.Tensor) -> dict[str, torch.Tensor]:
    w = w_eff.double()
    return {"B1": w, "P1": w * mu_in, "P1contrast": w * (mu_in - mu_out)}


def _eligible(score: torch.Tensor, w_eff: torch.Tensor) -> torch.Tensor:
    return score.masked_fill(w_eff <= 0, float("-inf"))


def topk_per_class(score: torch.Tensor, w_eff: torch.Tensor, k: int) -> torch.Tensor:
    s = _eligible(score, w_eff)
    k_eff = min(k, s.shape[1])
    idx = s.topk(k_eff, dim=1).indices
    mask = torch.zeros_like(w_eff, dtype=torch.bool).scatter_(1, idx, True)
    return mask & (w_eff > 0)


def random_per_class(w_eff: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    score = torch.rand(w_eff.shape, generator=g, dtype=torch.float64)
    return topk_per_class(score, w_eff, k)


def global_budget(score: torch.Tensor, w_eff: torch.Tensor, k: int,
                  min_one: bool = True) -> tuple[torch.Tensor, dict]:
    """Keep the N = k * C highest-scoring edges overall; every class keeps at least one edge.

    A class left with no edge gets its own best edge, and the lowest-scoring kept edge of a class with
    more than one edge is dropped for it, so the total stays exactly N. min_one=False (Phase 4 ablation)
    skips that step: classes may keep no edge at all and then can never be predicted."""
    C = w_eff.shape[0]
    n_total = k * C
    s = _eligible(score, w_eff)
    flat = s.flatten()
    keep = torch.zeros_like(flat, dtype=torch.bool)
    keep[flat.topk(n_total).indices] = True
    mask = keep.view_as(w_eff)
    forced = 0
    empty = torch.nonzero(mask.sum(1) == 0).flatten().tolist()
    for c in (empty if min_one else []):
        mask[c, s[c].argmax()] = True
        donors = mask & (mask.sum(1, keepdim=True) > 1)
        cand = s.masked_fill(~donors, float("inf"))
        drop = cand.flatten().argmin()
        mask.view(-1)[drop] = False
        forced += 1
    per_class = mask.sum(1)
    info = {"classes_forced_to_minimum": forced, "classes_at_minimum": int((per_class == 1).sum()),
            "classes_with_no_edge": int((per_class == 0).sum()),
            "edges_per_class_min": int(per_class.min()), "edges_per_class_median": float(per_class.float().median()),
            "edges_per_class_max": int(per_class.max())}
    return mask, info


def build_masks(w_eff: torch.Tensor, scores: dict[str, torch.Tensor], ks: list[int], seed: int) -> tuple[dict, dict]:
    masks, info = {}, {}
    for k in ks:
        masks[f"B1_K{k}"] = topk_per_class(scores["B1"], w_eff, k)
        masks[f"random_K{k}"] = random_per_class(w_eff, k, seed=seed * 1000 + k)
        masks[f"P1_K{k}"] = topk_per_class(scores["P1"], w_eff, k)
        masks[f"P1contrast_K{k}"] = topk_per_class(scores["P1contrast"], w_eff, k)
        masks[f"P3_K{k}"], info[f"P3_K{k}"] = global_budget(scores["P1"], w_eff, k)
    return masks, info


def mask_stats(mask: torch.Tensor) -> dict:
    per_class = mask.sum(1)
    return {"nonzero_edges": int(mask.sum()), "active_codes": int(mask.any(0).sum()),
            "edges_per_class_min": int(per_class.min()), "edges_per_class_max": int(per_class.max())}
