#!/usr/bin/env python3
"""HiFi-TSE training script with 3-phase curriculum learning.

Run with the USEF-TFGridNet conda environment:
    conda run -n USEF-TFGridNet python train.py --config configs/hifi_tse.yaml

Phase 1 (0 - 100k steps):   TP only, SI-SDR + multi-res STFT loss
Phase 2 (100k - 300k steps): TP + TA, add energy loss for target-absent
Phase 3 (300k+ steps):       Enable GAN (adversarial + feature matching)
"""

import argparse
import gc
import math
import os
import random
import resource
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
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


def _worker_init_fn(worker_id):
    """Seed each DataLoader worker independently to avoid correlated augmentations."""
    seed = torch.initial_seed() % 2**32 + worker_id
    random.seed(seed)
    np.random.seed(seed)


def make_train_loader(dataset, batch_size):
    """Create a DataLoader for training (recreated at phase transitions)."""
    return DataLoader(
        dataset, batch_size=batch_size,
        num_workers=4, pin_memory=True, shuffle=True, drop_last=True,
        collate_fn=tse_collate_fn,
        worker_init_fn=_worker_init_fn,
    )


def save_checkpoint(path, step, generator, discriminator, opt_G, opt_D, sched_G, sched_D,
                    best_val_loss=None, patience_counter=0, keep_last=5):
    """Save training checkpoint and remove old ones beyond keep_last."""
    ckpt_dir = os.path.dirname(path)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        "step": step,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "sched_G": sched_G.state_dict(),
        "sched_D": sched_D.state_dict(),
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
    }, path)

    # Rotate: keep only the last N numbered checkpoints
    numbered = sorted(
        f for f in os.listdir(ckpt_dir)
        if f.startswith("checkpoint_") and f.endswith(".pt") and f != "checkpoint_final.pt"
    )
    while len(numbered) > keep_last:
        old = os.path.join(ckpt_dir, numbered.pop(0))
        os.remove(old)
        print("Removed old checkpoint:", old)


def load_checkpoint(path, generator, discriminator, opt_G, opt_D, sched_G, sched_D, device):
    """Load training checkpoint. Returns (step, best_val_loss, patience_counter)."""
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    sched_G.load_state_dict(ckpt["sched_G"])
    sched_D.load_state_dict(ckpt["sched_D"])
    best_val_loss = ckpt.get("best_val_loss", None)
    patience_counter = ckpt.get("patience_counter", 0)
    return ckpt["step"], best_val_loss, patience_counter


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
    grad_accum = train_cfg["grad_accum_steps"]
    optimizer_steps = total_steps // grad_accum
    warmup_steps = train_cfg.get("warmup_steps", 2000)  # optimizer steps

    def warmup_cosine_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(warmup_steps, 1)
        progress = (current_step - warmup_steps) / max(optimizer_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=warmup_cosine_lambda)

    # D scheduler: separate warmup for GAN phase
    gan_optimizer_steps = (total_steps - cfg["curriculum"]["phase2_steps"]) // grad_accum

    def warmup_cosine_lambda_d(current_step):
        if current_step < warmup_steps:
            return current_step / max(warmup_steps, 1)
        progress = (current_step - warmup_steps) / max(gan_optimizer_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=warmup_cosine_lambda_d)

    # ---- Losses ----
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)
    loss_w = cfg["loss_weights"]

    # ---- Data ----
    train_dataset = HiFiTSEDataset(cfg, phase=1)
    train_loader = make_train_loader(train_dataset, train_cfg["batch_size"])
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
    best_val_loss = None
    patience_counter = 0
    if cfg["checkpoint"]["resume"]:
        resume_path = cfg["checkpoint"]["resume"]
        if os.path.exists(resume_path):
            print("Resuming from", resume_path)
            start_step, best_val_loss, patience_counter = load_checkpoint(
                resume_path, generator, discriminator,
                opt_G, opt_D, sched_G, sched_D, device,
            )
            print("Resumed at step {} (best_val_loss={}, patience={})".format(
                start_step, best_val_loss, patience_counter))

    # ---- Curriculum config ----
    phase1_steps = cfg["curriculum"]["phase1_steps"]
    phase2_steps = cfg["curriculum"]["phase2_steps"]
    grad_clip = train_cfg["grad_clip"]
    log_interval = train_cfg["log_interval"]
    save_interval = train_cfg["save_interval"]
    val_interval = train_cfg["val_interval"]

    current_phase = 1
    if start_step >= phase2_steps:
        current_phase = 3
        train_dataset.set_phase(3)
        train_dataset.close_handles()
        del train_iter, train_loader
        gc.collect()
        train_loader = make_train_loader(train_dataset, train_cfg["batch_size"])
        train_iter = infinite_loader(train_loader)
    elif start_step >= phase1_steps:
        current_phase = 2
        train_dataset.set_phase(2)
        train_dataset.close_handles()
        del train_iter, train_loader
        gc.collect()
        train_loader = make_train_loader(train_dataset, train_cfg["batch_size"])
        train_iter = infinite_loader(train_loader)

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
            train_dataset.close_handles()
            del train_iter, train_loader
            gc.collect()
            train_loader = make_train_loader(train_dataset, train_cfg["batch_size"])
            train_iter = infinite_loader(train_loader)
            print("==> Phase 2: enabling TA data + energy loss (step {})".format(step))
        elif step == phase2_steps and current_phase < 3:
            current_phase = 3
            train_dataset.set_phase(3)
            train_dataset.close_handles()
            del train_iter, train_loader
            gc.collect()
            train_loader = make_train_loader(train_dataset, train_cfg["batch_size"])
            train_iter = infinite_loader(train_loader)
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

        if current_phase == 1:
            loss_stft = stft_loss_fn(est_wav, target_wav)
        else:
            tp_mask_stft = tp_flag.bool()
            if tp_mask_stft.any():
                loss_stft = stft_loss_fn(est_wav[tp_mask_stft], target_wav[tp_mask_stft])
            else:
                loss_stft = torch.tensor(0.0, device=device)
        loss_G = loss_w["lambda_sep"] * loss_sep + loss_w["lambda_stft"] * loss_stft

        # ---- GAN losses (phase 3 only) ----
        # D and G backward passes are separated so their graphs don't
        # coexist in GPU memory.  G backward runs first (freeing G graph),
        # then D forward+backward runs with only D's graph in memory.
        loss_D_val = 0.0
        loss_adv_val = 0.0
        loss_fm_val = 0.0
        gan_active = False

        if current_phase >= 3:
            tp_mask = tp_flag.bool()

            if tp_mask.any():
                gan_active = True
                # Step 1: G adversarial forward (D frozen)
                discriminator.requires_grad_(False)
                d_real_out_ref, d_real_feat_ref = discriminator(target_wav[tp_mask])
                d_real_feat_det = [[f.detach() for f in feats]
                                   for feats in d_real_feat_ref]
                d_fake_out_g, d_fake_feat_g = discriminator(est_wav[tp_mask])
                loss_adv = generator_adv_loss(d_fake_out_g)
                loss_fm = feature_matching_loss(d_real_feat_det, d_fake_feat_g)
                loss_G = loss_G + loss_w["lambda_adv"] * loss_adv \
                    + loss_w["lambda_fm"] * loss_fm
                loss_adv_val = loss_adv.item()
                loss_fm_val = loss_fm.item()
                discriminator.requires_grad_(True)

        # ---- Backward G (frees G graph before D step) ----
        (loss_G / grad_accum).backward()

        if gan_active:
            # Step 2: D forward+backward (G graph already freed)
            est_wav_d = est_wav.detach()
            d_real_out, _ = discriminator(target_wav[tp_mask])
            d_fake_out, _ = discriminator(est_wav_d[tp_mask])
            loss_D = discriminator_loss(d_real_out, d_fake_out)
            (loss_D / grad_accum).backward()
            loss_D_val = loss_D.item()

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
            if current_phase >= 3:
                sched_D.step()

        dt = time.time() - t0

        # ---- Logging ----
        if step % log_interval == 0:
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print("step {:>7d} | phase {} | loss_G {:.4f} | sep {:.4f} | stft {:.4f}"
                  " | adv {:.4f} | fm {:.4f} | D {:.4f} | rss {:.0f}MB"
                  " | {:.2f}s".format(
                      step, current_phase, loss_G.item(), loss_sep.item(),
                      loss_stft.item(), loss_adv_val, loss_fm_val, loss_D_val,
                      rss_mb, dt))
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
                            opt_G, opt_D, sched_G, sched_D,
                            best_val_loss=best_val_loss,
                            patience_counter=patience_counter)
            print("Saved checkpoint:", ckpt_path)

        # ---- Validation + best model tracking ----
        if step > 0 and step % val_interval == 0:
            val_loss = validate(generator, val_loader, device, writer, step)

            # Best model tracking (active in phase 3 where overfitting risk exists)
            early_stop_patience = 10  # stop after 10 validations without improvement
            if current_phase >= 3:
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_path = os.path.join(ckpt_dir, "checkpoint_best.pt")
                    save_checkpoint(best_path, step, generator, discriminator,
                                    opt_G, opt_D, sched_G, sched_D,
                                    best_val_loss=best_val_loss,
                                    patience_counter=patience_counter,
                                    keep_last=999)
                    print("Validation loss: {:.4f} (new best, saved checkpoint_best.pt)".format(val_loss))
                else:
                    patience_counter += 1
                    print("Validation loss: {:.4f} (no improvement for {}/{} evals)".format(
                        val_loss, patience_counter, early_stop_patience))
                    if patience_counter >= early_stop_patience:
                        print("Early stopping triggered. Best val loss: {:.4f}".format(best_val_loss))
                        break
            else:
                print("Validation loss: {:.4f}".format(val_loss))

    # Final checkpoint
    final_path = os.path.join(ckpt_dir, "checkpoint_final.pt")
    save_checkpoint(final_path, total_steps, generator, discriminator,
                    opt_G, opt_D, sched_G, sched_D,
                    best_val_loss=best_val_loss,
                    patience_counter=patience_counter)
    print("Training complete. Final checkpoint:", final_path)
    writer.close()


if __name__ == "__main__":
    main()
