"""Scene-aware separation losses: SI-SDR for target-present, energy loss for target-absent."""

import torch


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


def energy_loss(estimate, eps=1e-8):
    """Energy loss for target-absent samples.

    Forces output energy toward zero: 10 * log10(mean(estimate^2) + eps)

    Args:
        estimate: (B, L) predicted waveform

    Returns:
        (B,) energy values in dB
    """
    energy = estimate.pow(2).mean(dim=-1)
    return 10.0 * torch.log10(energy + eps)


def scene_aware_loss(estimate, target, target_present):
    """Scene-aware loss dispatching SI-SDR for TP and energy for TA.

    Args:
        estimate: (B, L) predicted waveform
        target: (B, L) ground truth waveform (zeros for TA samples)
        target_present: (B,) bool/float tensor, 1.0 for TP, 0.0 for TA

    Returns:
        scalar loss
    """
    tp_mask = target_present.bool()
    ta_mask = ~tp_mask

    loss = torch.tensor(0.0, device=estimate.device)
    count = 0

    if tp_mask.any():
        tp_loss = -si_sdr(estimate[tp_mask], target[tp_mask]).mean()
        loss = loss + tp_loss
        count += 1

    if ta_mask.any():
        ta_loss = energy_loss(estimate[ta_mask]).mean()
        loss = loss + ta_loss
        count += 1

    if count > 0:
        loss = loss / count

    return loss
