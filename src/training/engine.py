"""A/B epoch execution; optional C integration reuses WANG's loss."""

import math

import torch
from torch.nn import functional as F

from src.losses import paired_view_loss


def binary_auc(labels, scores):
    """Mann-Whitney ROC AUC, assigning average ranks to ties."""
    if len(labels) != len(scores) or not scores:
        raise ValueError("AUC requires equally sized, nonempty inputs")
    if not all(label in (0, 1) for label in labels):
        raise ValueError("AUC labels must be binary")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("AUC scores must be finite")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both classes")
    ordered = sorted(zip(scores, labels))
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(label for _, label in ordered[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def per_image_loss(logits, targets, condition, view_logits=None, consistency_weight=1.0):
    if condition == "A":
        if logits.shape != targets.shape or logits.ndim != 1:
            raise ValueError("A requires logits and targets shaped (N,)")
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("labels must be binary")
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if condition not in ("B", "C") or view_logits is None:
        raise ValueError("B/C require paired logits")
    return paired_view_loss(
        logits, view_logits, targets,
        consistency_weight=0.0 if condition == "B" else consistency_weight,
        reduction="none",
    )


def train_epoch(model, loader, optimizer, scaler, device, condition, accumulation_steps=1,
                amp=False, consistency_weight=1.0):
    """Sample-weighted accumulation also handles a short final batch/group."""
    model.train()
    total_loss, total_samples, group_samples, updates, skipped_updates = 0.0, 0, 0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float()
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            view_logits = model(batch["view"].to(device)) if condition != "A" else None
            losses = per_image_loss(logits, labels, condition, view_logits, consistency_weight)
        if not torch.isfinite(losses).all():
            raise FloatingPointError("nonfinite training loss")
        scaler.scale(losses.sum()).backward()
        count = len(labels)
        group_samples += count
        total_samples += count
        total_loss += losses.detach().sum().item()
        if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader):
            scaler.unscale_(optimizer)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(group_samples)
                    if not scaler.is_enabled() and not torch.isfinite(param.grad).all():
                        raise FloatingPointError("nonfinite training gradient")
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            group_samples = 0
            skipped = scaler.is_enabled() and scaler.get_scale() < previous_scale
            skipped_updates += int(skipped)
            updates += int(not skipped)
    if not total_samples:
        raise ValueError("empty training loader")
    return {"train_loss": total_loss / total_samples, "optimizer_steps": updates,
            "amp_skipped_steps": skipped_updates,
            "train_images": total_samples}


@torch.no_grad()
def validate(model, loader, device):
    """Only clean validation inputs are used to select checkpoints."""
    model.eval()
    labels, scores, ranks, image_ids = [], [], [], []
    loss_sum = 0.0
    for batch in loader:
        targets = batch["label"].to(device).float()
        logits = model(batch["image"].to(device)).float()
        loss_sum += F.binary_cross_entropy_with_logits(logits, targets, reduction="sum").item()
        labels.extend(targets.cpu().tolist())
        scores.extend(logits.sigmoid().cpu().tolist())
        ranks.extend(logits.cpu().tolist())
        image_ids.extend(batch["image_id"])
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("validation contains duplicate image_id")
    # Sigmoid is monotonic mathematically, but FP32 saturation can erase ranks.
    auc = binary_auc(labels, ranks)
    accuracy = sum((score >= 0.5) == bool(label) for label, score in zip(labels, scores)) / len(labels)
    return {"val_loss": loss_sum / len(labels), "val_auc": auc, "val_accuracy": accuracy,
            "val_images": len(labels)}, list(zip(image_ids, labels, scores))
