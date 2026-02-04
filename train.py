#!/usr/bin/env python3
"""HiFi-TSE training script with 3-phase curriculum learning.

Run with the USEF-TFGridNet conda environment:
    conda run -n USEF-TFGridNet python train.py --config configs/hifi_tse.yaml

Phase 1 (0 - 100k steps):   TP only, SI-SDR + multi-res STFT loss
Phase 2 (100k - 300k steps): TP + TA, add energy loss for target-absent
Phase 3 (300k+ steps):       Enable GAN (adversarial + feature matching)
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from data.dataset import HiFiTSEDataset, tse_collate_fn
from losses.adversarial import (
    discriminator_loss,
    feature_matching_loss,
    generator_adv_loss,
)
from losses.separation import scene_aware_loss, si_sdr_loss
from losses.stft_loss import MultiResolutionSTFTLoss
from models.discriminator import Discriminator
from models.generator import Generator


def infinite_loader(loader):
    """Yield batches endlessly, restarting the loader when exhausted."""
    while True:
        for batch in loader:
            yield batch


def save_checkpoint(path, step, generator, discriminator, opt_G, opt_D, sched_G, sched_D):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "sched_G": sched_G.state_dict(),
        "sched_D": sched_D.state_dict(),
    }, path)


def load_checkpoint(path, generator, discriminator, opt_G, opt_D, sched_G, sched_D):
    """Load training checkpoint. Returns the step number."""
    ckpt = torch.load(path, map_location="cpu")
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    sched_G.load_state_dict(ckpt["sched_G"])
    sched_D.load_state_dict(ckpt["sched_D"])
    return ckpt["step"]


def validate(generator, val_loader, device, writer, step):
    """Run validation and log metrics."""
    generator.eval()
    total_loss = 0.0
    count = 0
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)

    with torch.no_grad():
        for batch in val_loader:
            mix_wav, ref_wav, target_wav, tp_flag = [x.to(device) for x in batch]
            est_wav = generator(mix_wav, ref_wav)
            loss = scene_aware_loss(est_wav, target_wav, tp_flag)
            total_loss += loss.item()
            count += 1
            if count >= 50:  # cap validation batches
                break

    avg_loss = total_loss / max(count, 1)
    if writer is not None:
        writer.add_scalar("val/scene_aware_loss", avg_loss, step)
    generator.train()
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train HiFi-TSE")
    parser.add_argument("--config", type=str, default="configs/hifi_tse.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- Models ----
    generator = Generator(cfg).to(device)
    discriminator = Discriminator(cfg).to(device)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print("Generator params: {:.2f}M".format(g_params / 1e6))
    print("Discriminator params: {:.2f}M".format(d_params / 1e6))

    # ---- Optimizers ----
    train_cfg = cfg["training"]
    opt_G = AdamW(generator.parameters(), lr=train_cfg["lr"],
                  betas=tuple(train_cfg["betas"]))
    opt_D = AdamW(discriminator.parameters(), lr=train_cfg["lr"],
                  betas=tuple(train_cfg["betas"]))
    total_steps = train_cfg["total_steps"]
    sched_G = CosineAnnealingLR(opt_G, T_max=total_steps)
    sched_D = CosineAnnealingLR(opt_D, T_max=total_steps)

    # ---- Losses ----
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)
    loss_w = cfg["loss_weights"]

    # ---- Data ----
    train_dataset = HiFiTSEDataset(cfg, phase=1)
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"],
        num_workers=4, pin_memory=True, shuffle=True, drop_last=True,
        collate_fn=tse_collate_fn,
    )
    train_iter = infinite_loader(train_loader)

    # Validation: reuse training dataset structure with a small subset
    # (Full validation would use a separate held-out set)
    val_dataset = HiFiTSEDataset(cfg, phase=2)
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"],
        num_workers=2, pin_memory=True, shuffle=True, drop_last=True,
        collate_fn=tse_collate_fn,
    )

    # ---- Logging ----
    ckpt_dir = cfg["checkpoint"]["dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "logs"))

    # ---- Resume ----
    start_step = 0
    if cfg["checkpoint"]["resume"]:
        resume_path = cfg["checkpoint"]["resume"]
        if os.path.exists(resume_path):
            print("Resuming from", resume_path)
            start_step = load_checkpoint(
                resume_path, generator, discriminator,
                opt_G, opt_D, sched_G, sched_D,
            )
            print("Resumed at step", start_step)

    # ---- Curriculum config ----
    phase1_steps = cfg["curriculum"]["phase1_steps"]
    phase2_steps = cfg["curriculum"]["phase2_steps"]
    grad_accum = train_cfg["grad_accum_steps"]
    grad_clip = train_cfg["grad_clip"]
    log_interval = train_cfg["log_interval"]
    save_interval = train_cfg["save_interval"]
    val_interval = train_cfg["val_interval"]

    current_phase = 1
    if start_step >= phase2_steps:
        current_phase = 3
        train_dataset.set_phase(3)
    elif start_step >= phase1_steps:
        current_phase = 2
        train_dataset.set_phase(2)

    print("Starting training from step {} (phase {})".format(start_step, current_phase))

    # ---- Training loop ----
    generator.train()
    discriminator.train()

    for step in range(start_step, total_steps):
        t0 = time.time()

        # Phase transitions
        if step == phase1_steps and current_phase < 2:
            current_phase = 2
            train_dataset.set_phase(2)
            print("==> Phase 2: enabling TA data + energy loss (step {})".format(step))
        elif step == phase2_steps and current_phase < 3:
            current_phase = 3
            train_dataset.set_phase(3)
            print("==> Phase 3: enabling GAN losses (step {})".format(step))

        # Get batch
        mix_wav, ref_wav, target_wav, tp_flag = next(train_iter)
        mix_wav = mix_wav.to(device)
        ref_wav = ref_wav.to(device)
        target_wav = target_wav.to(device)
        tp_flag = tp_flag.to(device)

        # ---- Forward G ----
        est_wav = generator(mix_wav, ref_wav)

        # ---- Separation + STFT loss (always active) ----
        if current_phase == 1:
            loss_sep = si_sdr_loss(est_wav, target_wav)
        else:
            loss_sep = scene_aware_loss(est_wav, target_wav, tp_flag)

        loss_stft = stft_loss_fn(est_wav, target_wav)
        loss_G = loss_w["lambda_sep"] * loss_sep + loss_w["lambda_stft"] * loss_stft

        # ---- GAN losses (phase 3 only) ----
        loss_D_val = 0.0
        loss_adv_val = 0.0
        loss_fm_val = 0.0

        if current_phase >= 3:
            # Update D
            est_wav_detach = est_wav.detach()
            d_real_out, d_real_feat = discriminator(target_wav)
            d_fake_out, d_fake_feat = discriminator(est_wav_detach)
            loss_D = discriminator_loss(d_real_out, d_fake_out)
            (loss_D / grad_accum).backward()
            loss_D_val = loss_D.item()

            # G adversarial + feature matching
            # Detach real features: they are fixed targets, and their graph
            # was already freed by loss_D.backward() above.
            d_real_feat_det = [[f.detach() for f in feats] for feats in d_real_feat]
            d_fake_out_g, d_fake_feat_g = discriminator(est_wav)
            loss_adv = generator_adv_loss(d_fake_out_g)
            loss_fm = feature_matching_loss(d_real_feat_det, d_fake_feat_g)
            loss_G = loss_G + loss_w["lambda_adv"] * loss_adv + loss_w["lambda_fm"] * loss_fm
            loss_adv_val = loss_adv.item()
            loss_fm_val = loss_fm.item()

        # ---- Backward G ----
        (loss_G / grad_accum).backward()

        # ---- Accumulate & step ----
        if (step + 1) % grad_accum == 0:
            clip_grad_norm_(generator.parameters(), grad_clip)
            opt_G.step()
            opt_G.zero_grad()

            if current_phase >= 3:
                clip_grad_norm_(discriminator.parameters(), grad_clip)
                opt_D.step()
                opt_D.zero_grad()

            sched_G.step()
            sched_D.step()

        dt = time.time() - t0

        # ---- Logging ----
        if step % log_interval == 0:
            print("step {:>7d} | phase {} | loss_G {:.4f} | sep {:.4f} | stft {:.4f}"
                  " | adv {:.4f} | fm {:.4f} | D {:.4f} | {:.2f}s".format(
                      step, current_phase, loss_G.item(), loss_sep.item(),
                      loss_stft.item(), loss_adv_val, loss_fm_val, loss_D_val, dt))
            writer.add_scalar("train/loss_G", loss_G.item(), step)
            writer.add_scalar("train/loss_sep", loss_sep.item(), step)
            writer.add_scalar("train/loss_stft", loss_stft.item(), step)
            writer.add_scalar("train/lr", sched_G.get_last_lr()[0], step)
            if current_phase >= 3:
                writer.add_scalar("train/loss_D", loss_D_val, step)
                writer.add_scalar("train/loss_adv", loss_adv_val, step)
                writer.add_scalar("train/loss_fm", loss_fm_val, step)

        # ---- Checkpoint ----
        if step > 0 and step % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint_{:07d}.pt".format(step))
            save_checkpoint(ckpt_path, step, generator, discriminator,
                            opt_G, opt_D, sched_G, sched_D)
            print("Saved checkpoint:", ckpt_path)

        # ---- Validation ----
        if step > 0 and step % val_interval == 0:
            val_loss = validate(generator, val_loader, device, writer, step)
            print("Validation loss: {:.4f}".format(val_loss))

    # Final checkpoint
    final_path = os.path.join(ckpt_dir, "checkpoint_final.pt")
    save_checkpoint(final_path, total_steps, generator, discriminator,
                    opt_G, opt_D, sched_G, sched_D)
    print("Training complete. Final checkpoint:", final_path)
    writer.close()


if __name__ == "__main__":
    main()
