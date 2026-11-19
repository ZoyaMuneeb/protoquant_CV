"""Physical prototype deletion and exact cached-feature loss evaluation."""
from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def keep_indices(count: int, removed: list[int], device: torch.device) -> torch.Tensor:
    if len(set(removed)) != len(removed):
        raise ValueError("Deletion indices must be unique.")
    if not removed or min(removed) < 0 or max(removed) >= count:
        raise ValueError("Deletion indices are empty or outside the current codebook.")
    if len(removed) >= count:
        raise ValueError("At least one prototype must remain.")
    mask = torch.ones(count, dtype=torch.bool, device=device)
    mask[removed] = False
    return mask.nonzero(as_tuple=False).flatten()


@torch.no_grad()
def compact_model(model: nn.Module, retained: torch.Tensor) -> nn.Module:
    """Copy the model and retain precisely these code rows and raw classifier columns."""
    retained = torch.as_tensor(retained, dtype=torch.long, device=model.codes.device)
    if retained.ndim != 1 or retained.numel() == 0:
        raise ValueError("Retained prototype indices must be a nonempty vector.")
    if retained.unique().numel() != retained.numel():
        raise ValueError("Retained prototype indices must be unique.")
    if int(retained.min()) < 0 or int(retained.max()) >= model.num_prototypes:
        raise ValueError("Retained index is outside the source codebook.")
    result = copy.deepcopy(model)
    codes = model.codes.detach().index_select(0, retained).clone()
    if isinstance(model.codes, nn.Parameter):
        result.codes = nn.Parameter(codes, requires_grad=model.codes.requires_grad)
    else:
        result.codes = codes
    weights = model.classifier.weight.detach().index_select(1, retained).clone()
    result.classifier.weight = nn.Parameter(weights, requires_grad=model.classifier.weight.requires_grad)
    result.num_prototypes = int(retained.numel())
    return result.eval()


@torch.no_grad()
def forward_features(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    similarities = model._detect_prototypes(features.float())
    scores = model.pool_layer(F.softmax(similarities / model.temperature, dim=1))
    return model.classifier(scores)


@torch.no_grad()
def spatial_logits(model: nn.Module, features: torch.Tensor, batch_size: int) -> torch.Tensor:
    device = model.codes.device
    chunks = []
    for start in range(0, len(features), batch_size):
        similarity = model._detect_prototypes(features[start:start + batch_size].to(device).float())
        chunks.append((similarity / model.temperature).flatten(2).transpose(1, 2).contiguous())
    result = torch.cat(chunks)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite prototype similarities in calibration features.")
    return result


@torch.no_grad()
def p1_ranking(model: nn.Module, features: torch.Tensor, labels: torch.Tensor,
               batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse the existing class means and P1 scores; aggregate by sum over classes."""
    from phase2_lib import class_means, edge_scores

    scores = []
    for start in range(0, len(features), batch_size):
        similarity = model._detect_prototypes(features[start:start + batch_size].to(model.codes.device).float())
        scores.append(model.pool_layer(F.softmax(similarity / model.temperature, dim=1)).double().cpu())
    pooled = torch.cat(scores)
    classes = model.classifier.weight.shape[0]
    mu_in, mu_out = class_means(pooled, labels.cpu(), classes)
    weights = torch.relu(model.classifier.weight.detach()).cpu()
    importance = edge_scores(weights, mu_in, mu_out)["P1"].sum(0)
    if not torch.isfinite(importance).all():
        raise ValueError("Nonfinite P1 importance scores.")
    return importance, torch.argsort(importance, stable=True)


class DeletionEvaluator:
    """Current spatial softmax state. IDs always refer to the pilot's starting model."""

    @torch.no_grad()
    def __init__(self, logits: torch.Tensor, weights: torch.Tensor, labels: torch.Tensor,
                 bias: torch.Tensor | None = None, image_batch_size: int = 32,
                 candidate_chunk_size: int = 4):
        if logits.ndim != 3 or weights.ndim != 2 or logits.shape[2] != weights.shape[1]:
            raise ValueError("Expected logits [images, positions, codes] and weights [classes, codes].")
        if len(logits) != len(labels) or len(labels) == 0:
            raise ValueError("Calibration features and labels must have the same nonzero length.")
        if image_batch_size < 1 or candidate_chunk_size < 1:
            raise ValueError("Image and candidate chunk sizes must be positive.")
        self.logits = logits.detach().float().clone()
        self.weights = torch.relu(weights.detach()).to(logits.device).float().clone()
        self.labels = labels.to(logits.device).long()
        self.bias = bias.detach().to(logits.device).float() if bias is not None else None
        self.ids = torch.arange(logits.shape[2], device=logits.device)
        self.image_batch_size = image_batch_size
        self.candidate_chunk_size = candidate_chunk_size
        self.candidate_evaluations = 0
        self.joint_evaluations = 0
        self.refresh()

    @torch.no_grad()
    def refresh(self) -> None:
        # Rebase from original surviving logits to avoid accumulated renormalization error.
        self.probabilities = self.logits.softmax(-1)
        self.predictions = self.probabilities.amax(1) @ self.weights.t()
        if self.bias is not None:
            self.predictions += self.bias
        self.loss = float(F.cross_entropy(self.predictions, self.labels))
        if not math.isfinite(self.loss):
            raise ValueError("Nonfinite calibration cross-entropy.")

    @torch.no_grad()
    def candidate_losses(self, candidates: list[int]) -> torch.Tensor:
        if not candidates:
            raise ValueError("Candidate shortlist is empty.")
        if len(self.ids) < 2:
            raise ValueError("Cannot delete the last remaining prototype.")
        if len(set(candidates)) != len(candidates) or min(candidates) < 0 or max(candidates) >= len(self.ids):
            raise ValueError("Candidates must be unique indices inside the current codebook.")
        sums = torch.zeros(len(candidates), dtype=torch.float64, device=self.logits.device)
        for offset in range(0, len(candidates), self.candidate_chunk_size):
            indices = torch.tensor(candidates[offset:offset + self.candidate_chunk_size], device=self.logits.device)
            for start in range(0, len(self.labels), self.image_batch_size):
                p = self.probabilities[start:start + self.image_batch_size]
                denominator = 1.0 - p.index_select(2, indices).permute(0, 2, 1).unsqueeze(-1)
                if bool((denominator <= 1e-6).any()):
                    # Stable fallback when subtracting a near-one probability loses precision.
                    trial = self.logits[start:start + len(p), None].expand(-1, len(indices), -1, -1).clone()
                    trial.scatter_(3, indices[None, :, None, None].expand(len(p), -1, p.shape[1], 1), -torch.inf)
                    remaining = trial.softmax(-1)
                else:
                    remaining = p[:, None] / denominator
                    remaining.scatter_(3, indices[None, :, None, None].expand(len(p), -1, p.shape[1], 1), 0.0)
                predictions = remaining.amax(2) @ self.weights.t()
                if self.bias is not None:
                    predictions += self.bias
                targets = self.labels[start:start + len(p), None].expand(-1, len(indices))
                losses = F.cross_entropy(predictions.flatten(0, 1), targets.reshape(-1), reduction="none")
                sums[offset:offset + len(indices)] += losses.reshape(len(p), -1).double().sum(0)
        self.candidate_evaluations += len(candidates)
        result = sums / len(self.labels)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite candidate deletion loss.")
        return result

    @torch.no_grad()
    def loss_after(self, removed: list[int]) -> float:
        retained = keep_indices(len(self.ids), removed, self.logits.device)
        total = 0.0
        weights = self.weights.index_select(1, retained)
        for start in range(0, len(self.labels), self.image_batch_size):
            trial = self.logits[start:start + self.image_batch_size].index_select(2, retained)
            predictions = trial.softmax(-1).amax(1) @ weights.t()
            if self.bias is not None:
                predictions += self.bias
            total += float(F.cross_entropy(predictions, self.labels[start:start + len(trial)], reduction="sum"))
        self.joint_evaluations += 1
        result = total / len(self.labels)
        if not math.isfinite(result):
            raise ValueError("Nonfinite joint deletion loss.")
        return result

    @torch.no_grad()
    def remove(self, removed: list[int]) -> list[int]:
        deleted_ids = self.ids[removed].cpu().tolist()
        retained = keep_indices(len(self.ids), removed, self.logits.device)
        self.logits = self.logits.index_select(2, retained)
        self.weights = self.weights.index_select(1, retained)
        self.ids = self.ids.index_select(0, retained)
        self.refresh()
        return deleted_ids


def prune(evaluator: DeletionEvaluator, ranking: torch.Tensor, count: int, method: str,
          shortlist_size: int = 64, batch_size: int = 8, extra_loss_tolerance: float = 0.005,
          extra_loss_relative: float = 0.25, record: Callable[[dict], None] | None = None) -> dict:
    """Fixed P1 ordering; every deletion loss is conditional on the current codebook.

    A batch is accepted if it decreases loss, or joint_delta - sum(single_deltas) is at most
    abs_tolerance + relative_tolerance * sum(abs(single_deltas)), measured in CE/image.
    This permits real interaction damage. Failed proposals are halved and reevaluated.
    """
    if method not in ("p1", "greedy", "batched"):
        raise ValueError(f"Unknown pruning method: {method}")
    if not 0 < count < len(evaluator.ids):
        raise ValueError("Removal count must be positive and leave at least one prototype.")
    if shortlist_size < 1 or batch_size < 1:
        raise ValueError("Shortlist and batch sizes must be positive.")
    if not math.isfinite(extra_loss_tolerance) or not math.isfinite(extra_loss_relative):
        raise ValueError("Batch tolerances must be finite.")
    if extra_loss_tolerance < 0 or extra_loss_relative < 0:
        raise ValueError("Batch tolerances must be nonnegative.")
    order = ranking.cpu().tolist()
    if sorted(order) != sorted(evaluator.ids.cpu().tolist()):
        raise ValueError("P1 ranking must contain every starting prototype exactly once.")
    started = time.perf_counter()
    initial_loss = evaluator.loss
    removed_total, fallbacks, accepted = 0, 0, []
    while removed_total < count:
        positions = {original: current for current, original in enumerate(evaluator.ids.cpu().tolist())}
        remaining_order = [positions[i] for i in order if i in positions]
        if method == "p1":
            proposed = remaining_order[:count]
            loss_before = evaluator.loss
            deleted = evaluator.remove(proposed)
            event = {"event": "accepted", "method": method, "removed_ids": deleted,
                     "batch_size": len(deleted), "loss_before": loss_before, "loss_after": evaluator.loss}
        else:
            candidates = remaining_order[:min(shortlist_size, len(remaining_order))]
            losses = evaluator.candidate_losses(candidates)
            best = torch.argsort(losses, stable=True).cpu().tolist()
            size = min(batch_size if method == "batched" else 1, count - removed_total, len(best))
            loss_before = evaluator.loss
            while True:
                selected = best[:size]
                proposed = [candidates[i] for i in selected]
                deltas = [float(losses[i]) - loss_before for i in selected]
                joint_loss = evaluator.loss_after(proposed)
                joint_delta = joint_loss - loss_before
                additional_damage = joint_delta - sum(deltas)
                allowance = extra_loss_tolerance + extra_loss_relative * sum(abs(d) for d in deltas)
                accepted_batch = size == 1 or joint_delta <= 0 or additional_damage <= allowance
                decision = {"event": "batch_check", "method": method, "proposed_size": size,
                            "proposed_ids": evaluator.ids[proposed].cpu().tolist(),
                            "individual_loss_changes": deltas, "joint_loss_change": joint_delta,
                            "additional_damage": additional_damage, "allowed_additional_damage": allowance,
                            "accepted": accepted_batch, "elapsed_seconds": time.perf_counter() - started}
                if not accepted_batch:
                    new_size = max(1, size // 2)
                    decision.update(reason="interaction_damage_exceeds_tolerance", fallback_to=new_size)
                    fallbacks += 1
                else:
                    decision["reason"] = ("single_deletion" if size == 1 else
                                          "combined_loss_does_not_increase" if joint_delta <= 0 else
                                          "within_interaction_damage_tolerance")
                if record:
                    record(decision)
                if accepted_batch:
                    break
                size = new_size
            deleted = evaluator.remove(proposed)
            if abs(evaluator.loss - joint_loss) > 1e-4 * max(1.0, abs(joint_loss)):
                raise RuntimeError("Committed deletion disagrees with its joint-loss evaluation.")
            event = {"event": "accepted", "method": method, "removed_ids": deleted, "batch_size": size,
                     "loss_before": loss_before, "loss_after": evaluator.loss}
        removed_total += len(deleted)
        accepted.append(len(deleted))
        event.update(removed_total=removed_total, remaining_codes=len(evaluator.ids),
                     elapsed_seconds=time.perf_counter() - started)
        if record:
            record(event)
    return {"calibration_loss_before": initial_loss, "calibration_loss_after": evaluator.loss,
            "removed_count": removed_total, "remaining_codes": len(evaluator.ids),
            "candidate_evaluations": evaluator.candidate_evaluations,
            "joint_evaluations": evaluator.joint_evaluations, "fallback_count": fallbacks,
            "accepted_batch_sizes": accepted}
