"""Adversarial and feature matching losses for HiFi-GAN discriminators."""

import torch


def generator_adv_loss(disc_outputs):
    """Generator adversarial loss: LS-GAN formulation.

    The generator wants the discriminator to output 1 (real) for its outputs.

    Args:
        disc_outputs: list of (output_tensor,) from each sub-discriminator

    Returns:
        scalar loss
    """
    loss = torch.tensor(0.0, device=disc_outputs[0].device)
    for out in disc_outputs:
        loss = loss + torch.mean((out - 1.0) ** 2)
    return loss / len(disc_outputs)


def discriminator_loss(real_outputs, fake_outputs):
    """Discriminator loss: LS-GAN formulation.

    Real outputs should be 1, fake outputs should be 0.

    Args:
        real_outputs: list of output tensors from each sub-discriminator on real data
        fake_outputs: list of output tensors from each sub-discriminator on fake data

    Returns:
        scalar loss
    """
    loss = torch.tensor(0.0, device=real_outputs[0].device)
    for real_out, fake_out in zip(real_outputs, fake_outputs):
        loss = loss + torch.mean((real_out - 1.0) ** 2) + torch.mean(fake_out ** 2)
    return loss / len(real_outputs)


def feature_matching_loss(real_features, fake_features):
    """Feature matching loss: L1 between intermediate discriminator features.

    Args:
        real_features: list of lists of feature tensors, one list per sub-discriminator
        fake_features: same structure as real_features

    Returns:
        scalar loss
    """
    loss = torch.tensor(0.0, device=real_features[0][0].device)
    count = 0
    for real_feats, fake_feats in zip(real_features, fake_features):
        for real_f, fake_f in zip(real_feats, fake_feats):
            loss = loss + torch.mean(torch.abs(real_f - fake_f))
            count += 1
    if count > 0:
        loss = loss / count
    return loss
