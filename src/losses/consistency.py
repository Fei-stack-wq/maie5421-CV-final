"""Binary paired-view supervision and prediction consistency."""

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _binary_logits(logits: Tensor) -> Tensor:
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("expected nonempty logits with shape (N,) or (N, 1)")
    # Keep probability and log calculations in FP32 during mixed precision.
    return logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits


def _paired_logits(logits: Tensor, view_logits: Tensor):
    logits, view_logits = _binary_logits(logits), _binary_logits(view_logits)
    if logits.shape != view_logits.shape:
        raise ValueError("both views must contain the same number of samples")
    if logits.device != view_logits.device:
        raise ValueError("both views must be on the same device")
    return logits, view_logits


def _reduce(values: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction must be 'none', 'mean' or 'sum'")


def _bernoulli_js(logits: Tensor, view_logits: Tensor) -> Tensor:
    # [P(real), P(AI)], calculated from logits without clipping probabilities.
    log_p = torch.stack((F.logsigmoid(-logits), F.logsigmoid(logits)), dim=-1)
    log_q = torch.stack((F.logsigmoid(-view_logits), F.logsigmoid(view_logits)), dim=-1)
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    divergence = 0.5 * (
        (log_p.exp() * (log_p - log_m)).sum(dim=-1)
        + (log_q.exp() * (log_q - log_m)).sum(dim=-1)
    )
    # Roundoff can make the theoretical lower bound slightly negative.
    return divergence.clamp_min(0.0)


def bernoulli_js_divergence(
    logits: Tensor, view_logits: Tensor, reduction: str = "mean"
) -> Tensor:
    """JS(Bernoulli(sigmoid(logits)), Bernoulli(sigmoid(view_logits))).

    Accepts one AI logit per image, shaped (N,) or (N, 1). Uses natural
    logarithms, so each divergence lies in [0, log(2)]. Both views retain
    gradients. With reduction='none', returns one value per image, shape (N,).
    """
    logits, view_logits = _paired_logits(logits, view_logits)
    return _reduce(_bernoulli_js(logits, view_logits), reduction)


def paired_view_loss(
    logits: Tensor,
    view_logits: Tensor,
    targets: Tensor,
    consistency_weight: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Mean of the two view BCEs plus consistency_weight * Bernoulli JS.

    targets contains binary labels (0=real, 1=AI), shaped (N,) or (N, 1).
    A weight of zero gives condition B; a weight of one gives condition C.
    Pass raw model logits, not sigmoid probabilities.
    """
    if not math.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError("consistency_weight must be finite and nonnegative")
    logits, view_logits = _paired_logits(logits, view_logits)
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    if targets.shape != logits.shape:
        raise ValueError("targets must have one label per image")
    if targets.device != logits.device:
        raise ValueError("targets and logits must be on the same device")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("targets must contain only 0 (real) or 1 (AI)")
    targets = targets.to(dtype=logits.dtype)
    supervised = 0.5 * (
        F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        + F.binary_cross_entropy_with_logits(view_logits, targets, reduction="none")
    )
    if consistency_weight == 0:
        return _reduce(supervised, reduction)
    return _reduce(supervised + consistency_weight * _bernoulli_js(logits, view_logits), reduction)
