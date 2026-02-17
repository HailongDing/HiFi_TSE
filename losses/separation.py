"""Scene-aware separation losses: SI-SDR for target-present, energy loss for target-absent."""

import torch
import torch.nn.functional as F


def si_sdr(estimate, target, eps=1e-8):
    """Scale-Invariant Signal-to-Distortion Ratio.

    Args:
        estimate: (B, L) predicted waveform
        target: (B, L) ground truth waveform

    Returns:
        (B,) SI-SDR values in dB (higher is better)
    """
    # Zero-mean
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    dot = (target * estimate).sum(dim=-1, keepdim=True)
    s_target_energy = (target * target).sum(dim=-1, keepdim=True).clamp(min=eps)
    proj = dot * target / s_target_energy

    noise = estimate - proj
    si_sdr_val = 10.0 * torch.log10(
        proj.pow(2).sum(dim=-1).clamp(min=eps)
        / noise.pow(2).sum(dim=-1).clamp(min=eps)
    )
    return si_sdr_val


def si_sdr_loss(estimate, target):
    """Negative SI-SDR loss (lower is better).

    Args:
        estimate: (B, L) predicted waveform
        target: (B, L) ground truth waveform

    Returns:
        scalar loss
    """
    return -si_sdr(estimate, target).mean()


def energy_loss(estimate, target_db=-30.0, eps=1e-8):
    """Energy loss for target-absent samples.

    Penalizes output energy above target_db. Once energy is suppressed to
    the target threshold, gradient stops — preventing over-suppression of
    shared model weights that also serve TP samples.

    Returns relu(energy_db - target_db), which is always >= 0.

    Args:
        estimate: (B, L) predicted waveform
        target_db: suppression target in dB (gradient stops below this)

    Returns:
        (B,) non-negative loss values (0 when sufficiently suppressed)
    """
    energy = estimate.pow(2).mean(dim=-1)
    energy_db = 10.0 * torch.log10(energy + eps)
    return torch.relu(energy_db - target_db)


def amplitude_loss(estimate, target, eps=1e-8):
    """Penalize RMS ratio deviation from 1.0 in log-space.

    Uses log(ratio)^2 so that:
    - Gradient is exactly zero at ratio=1.0 (no interference when correct)
    - Stronger penalty than linear-space L2 for large deviations
    - Symmetric in log-space (0.5x and 2x penalized equally)

    Args:
        estimate: (B, L) predicted waveform
        target: (B, L) ground truth waveform

    Returns:
        scalar loss = mean(log(rms_est / rms_tgt)^2)
    """
    rms_est = estimate.pow(2).mean(dim=-1).sqrt()
    rms_tgt = target.pow(2).mean(dim=-1).sqrt().clamp(min=eps)
    ratio = (rms_est / rms_tgt).clamp(min=0.05)  # prevent log explosion
    return torch.log(ratio).pow(2).mean()


def l1_waveform_loss(estimate, target):
    """Scale-sensitive L1 loss on time-domain waveform.

    Args:
        estimate: (B, L) predicted waveform
        target: (B, L) ground truth waveform

    Returns:
        scalar loss
    """
    return F.l1_loss(estimate, target)


def scene_aware_loss(estimate, target, target_present, ta_weight=0.2):
    """Scene-aware loss dispatching SI-SDR for TP and energy for TA.

    Uses explicit TP/TA weighting instead of equal averaging, so that TA
    energy loss doesn't dominate over TP extraction quality.

    Args:
        estimate: (B, L) predicted waveform
        target: (B, L) ground truth waveform (zeros for TA samples)
        target_present: (B,) bool/float tensor, 1.0 for TP, 0.0 for TA
        ta_weight: weight for TA energy loss (default 0.2, matching ta_ratio)

    Returns:
        scalar loss
    """
    tp_mask = target_present.bool()
    ta_mask = ~tp_mask

    loss = torch.tensor(0.0, device=estimate.device)

    if tp_mask.any():
        tp_loss = -si_sdr(estimate[tp_mask], target[tp_mask]).mean()
        loss = loss + (1.0 - ta_weight) * tp_loss

    if ta_mask.any():
        ta_loss = energy_loss(estimate[ta_mask]).mean()
        loss = loss + ta_weight * ta_loss

    return loss
