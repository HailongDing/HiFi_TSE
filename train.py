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
from losses.separation import amplitude_loss, l1_waveform_loss, scene_aware_loss, si_sdr, si_sdr_loss
from losses.stft_loss import MultiResolutionSTFTLoss, PhaseSensitiveLoss
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


def _step_from_filename(fname):
    """Extract step number from checkpoint_XXXXXXX.pt filename, or None."""
    base = os.path.splitext(fname)[0]  # checkpoint_0100000
    parts = base.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return None


def save_checkpoint(path, step, generator, discriminator, opt_G, opt_D, sched_G, sched_D,
                    best_val_loss=None, patience_counter=0, keep_last=5,
                    protected_steps=None):
    """Save training checkpoint and remove old ones beyond keep_last.

    Args:
        protected_steps: set of step numbers whose checkpoints are never deleted.
    """
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

    if protected_steps is None:
        protected_steps = set()

    # Rotate: keep only the last N numbered checkpoints (skip protected + best)
    special = {"checkpoint_final.pt", "checkpoint_best.pt"}
    numbered = sorted(
        f for f in os.listdir(ckpt_dir)
        if f.startswith("checkpoint_") and f.endswith(".pt") and f not in special
    )
    # Filter out protected checkpoints before counting
    deletable = [f for f in numbered if _step_from_filename(f) not in protected_steps]
    while len(deletable) > keep_last:
        old = os.path.join(ckpt_dir, deletable.pop(0))
        os.remove(old)
        print("Removed old checkpoint:", old)


def load_checkpoint(path, generator, discriminator, opt_G, opt_D, sched_G, sched_D, device,
                    reset_optimizer=False):
    """Load training checkpoint. Returns (step, best_val_loss, patience_counter)."""
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    if reset_optimizer:
        print("  Optimizer state RESET (not loaded from checkpoint)")
        # Still advance scheduler to match checkpoint step
        step = ckpt["step"]
        # Infer grad_accum from scheduler state in checkpoint
        # sched state last_epoch == number of optimizer steps taken
        sched_steps = ckpt["sched_G"]["last_epoch"]
        for _ in range(sched_steps):
            sched_G.step()
            sched_D.step()
    else:
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        sched_G.load_state_dict(ckpt["sched_G"])
        sched_D.load_state_dict(ckpt["sched_D"])
    best_val_loss = ckpt.get("best_val_loss", None)
    patience_counter = ckpt.get("patience_counter", 0)
    return ckpt["step"], best_val_loss, patience_counter


def validate(generator, val_loader, device, writer, step):
    """Run validation and log decomposed metrics."""
    generator.eval()
    total_loss = 0.0
    total_si_sdr = 0.0
    total_rms_num = 0.0
    total_rms_den = 0.0
    total_ta_energy = 0.0
    count = 0
    tp_samples = 0
    ta_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            mix_wav, ref_wav, target_wav, tp_flag = [x.to(device) for x in batch]
            est_wav = generator(mix_wav, ref_wav)
            loss = scene_aware_loss(est_wav, target_wav, tp_flag)
            total_loss += loss.item()

            # Decomposed metrics
            tp_mask = tp_flag.bool()
            if tp_mask.any():
                n_tp = tp_mask.sum().item()
                sdr_vals = si_sdr(est_wav[tp_mask], target_wav[tp_mask])
                total_si_sdr += sdr_vals.sum().item()
                # Per-sample RMS: compute per sample then sum
                est_rms = est_wav[tp_mask].pow(2).mean(dim=-1).sqrt()
                tgt_rms = target_wav[tp_mask].pow(2).mean(dim=-1).sqrt().clamp(min=1e-8)
                total_rms_num += est_rms.sum().item()
                total_rms_den += tgt_rms.sum().item()
                tp_samples += n_tp

            ta_mask = ~tp_mask
            if ta_mask.any():
                ta_energy = 10 * torch.log10(est_wav[ta_mask].pow(2).mean().clamp(min=1e-10)).item()
                total_ta_energy += ta_energy
                ta_batches += 1

            count += 1
            if count >= 50:
                break

    avg_loss = total_loss / max(count, 1)
    avg_si_sdr = total_si_sdr / max(tp_samples, 1)
    avg_rms_ratio = (total_rms_num / max(tp_samples, 1)) / max(total_rms_den / max(tp_samples, 1), 1e-8)
    avg_ta_energy = total_ta_energy / max(ta_batches, 1) if ta_batches > 0 else float('nan')

    print("Validation loss: {:.4f} | si_sdr: {:.2f} dB | rms_ratio: {:.3f} | ta_energy: {:.1f} dB".format(
        avg_loss, avg_si_sdr, avg_rms_ratio, avg_ta_energy))

    if writer is not None:
        writer.add_scalar("val/scene_aware_loss", avg_loss, step)
        writer.add_scalar("val/si_sdr", avg_si_sdr, step)
        writer.add_scalar("val/rms_ratio", avg_rms_ratio, step)
        if ta_batches > 0:
            writer.add_scalar("val/ta_energy", avg_ta_energy, step)

    generator.train()
    return avg_loss


def mini_evaluate(generator, val_loader, device, step):
    """Quick evaluation at milestones: 10 batches, report SI-SDR + rms_ratio."""
    generator.eval()
    total_si_sdr = 0.0
    total_rms_num = 0.0
    total_rms_den = 0.0
    n_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 10:
                break
            mix_wav, ref_wav, target_wav, tp_flag = [x.to(device) for x in batch]
            est_wav = generator(mix_wav, ref_wav)
            tp_mask = tp_flag.bool()
            if tp_mask.any():
                sdr_vals = si_sdr(est_wav[tp_mask], target_wav[tp_mask])
                total_si_sdr += sdr_vals.sum().item()
                n_samples += tp_mask.sum().item()
                total_rms_num += est_wav[tp_mask].pow(2).mean().sqrt().item()
                total_rms_den += target_wav[tp_mask].pow(2).mean().sqrt().clamp(min=1e-8).item()

    avg_si_sdr = total_si_sdr / max(n_samples, 1)
    avg_rms_ratio = total_rms_num / max(total_rms_den, 1e-8) if total_rms_den > 0 else 0.0

    print("MILESTONE_EVAL step {} | si_sdr: {:.2f} dB | rms_ratio: {:.3f}".format(
        step, avg_si_sdr, avg_rms_ratio))

    if step <= 5000 and avg_rms_ratio < 0.3:
        print("EARLY_CHECK WARNING step {} | rms_ratio {:.3f} < 0.3 — possible over-suppression!".format(
            step, avg_rms_ratio))

    generator.train()
    return avg_si_sdr, avg_rms_ratio


def real_audio_check(generator, device, step, ckpt_dir):
    """Run inference on real audio files and save output wav at milestones."""
    import torchaudio

    mix_path = "real_audio/mix_16k.wav"
    ref_path = "real_audio/ref_16k.wav"
    if not os.path.exists(mix_path) or not os.path.exists(ref_path):
        print("REAL_AUDIO_CHECK step {} | skipped (files not found)".format(step))
        return

    generator.eval()
    with torch.no_grad():
        mix_wav, mix_sr = torchaudio.load(mix_path)
        ref_wav, ref_sr = torchaudio.load(ref_path)

        # Resample to 48kHz if needed
        if mix_sr != 48000:
            mix_wav = torchaudio.functional.resample(mix_wav, mix_sr, 48000)
        if ref_sr != 48000:
            ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, 48000)

        # Truncate to 4s (192000 samples) to match training segment length
        # and avoid OOM from TFGridNet's O(T^2) self-attention on long audio
        max_samples = 192000
        if mix_wav.shape[-1] > max_samples:
            mix_wav = mix_wav[..., :max_samples]
        if ref_wav.shape[-1] > max_samples:
            ref_wav = ref_wav[..., :max_samples]

        # Add batch dim, move to device
        mix_wav = mix_wav.unsqueeze(0).to(device)  # (1, C, L) or (1, L)
        ref_wav = ref_wav.unsqueeze(0).to(device)

        # Ensure mono (1, L)
        if mix_wav.dim() == 3:
            mix_wav = mix_wav[:, 0, :]
        if ref_wav.dim() == 3:
            ref_wav = ref_wav[:, 0, :]

        est_wav = generator(mix_wav, ref_wav)

        est_rms = est_wav.pow(2).mean().sqrt().item()
        mix_rms = mix_wav.pow(2).mean().sqrt().item()
        ratio = est_rms / max(mix_rms, 1e-8)

        print("REAL_AUDIO_CHECK step {} | est_rms: {:.4f} | mix_rms: {:.4f} | ratio: {:.3f}".format(
            step, est_rms, mix_rms, ratio))

        # Save output wav
        out_path = os.path.join(ckpt_dir, "real_audio_step_{:07d}.wav".format(step))
        torchaudio.save(out_path, est_wav.cpu().squeeze(0).unsqueeze(0), 48000)
        print("  -> Saved:", out_path)

    generator.train()


def main():
    parser = argparse.ArgumentParser(description="Train HiFi-TSE")
    parser.add_argument("--config", type=str, default="configs/hifi_tse.yaml")
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="Reset Adam state when resuming (use after loss weight changes)")
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
    phase_loss_fn = PhaseSensitiveLoss().to(device)
    loss_w = cfg["loss_weights"]

    # ---- Data ----
    train_dataset = HiFiTSEDataset(cfg, phase=1)
    train_loader = make_train_loader(train_dataset, train_cfg["batch_size"])
    train_iter = infinite_loader(train_loader)

    # Noisy reference probability ramp
    data_cfg = cfg["data"]
    noisy_ref_max = data_cfg["noisy_ref_prob"]
    noisy_ref_ramp_start = data_cfg.get("noisy_ref_ramp_start", 0)
    noisy_ref_ramp_end = data_cfg.get("noisy_ref_ramp_end", 0)

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
                reset_optimizer=args.reset_optimizer,
            )
            print("Resumed at step {} (best_val_loss={}, patience={})".format(
                start_step, best_val_loss, patience_counter))

    # ---- Curriculum config ----
    phase1_steps = cfg["curriculum"]["phase1_steps"]
    phase2_steps = cfg["curriculum"]["phase2_steps"]
    gan_d_only_steps = cfg["curriculum"].get("gan_d_only_steps", 5000)
    gan_warmup_steps = cfg["curriculum"].get("gan_warmup_steps", 20000)
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

    # Phase boundary checkpoints are never deleted by rotation
    PROTECTED_STEPS = {phase1_steps, phase2_steps, phase2_steps + gan_warmup_steps}
    MILESTONE_STEPS = {5000, phase1_steps, 200000, phase2_steps,
                       phase2_steps + gan_warmup_steps, 400000, total_steps}

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

        # Update noisy_ref_prob ramp (propagates to workers via shared memory)
        if noisy_ref_ramp_end > noisy_ref_ramp_start:
            progress = max(0.0, min(1.0, (step - noisy_ref_ramp_start) / (noisy_ref_ramp_end - noisy_ref_ramp_start)))
            current_noisy_ref_prob = noisy_ref_max * progress
        else:
            current_noisy_ref_prob = noisy_ref_max
        train_dataset.set_noisy_ref_prob(current_noisy_ref_prob)

        # Get batch
        mix_wav, ref_wav, target_wav, tp_flag = next(train_iter)
        mix_wav = mix_wav.to(device)
        ref_wav = ref_wav.to(device)
        target_wav = target_wav.to(device)
        tp_flag = tp_flag.to(device)

        # ---- Forward G ----
        est_wav = generator(mix_wav, ref_wav)

        # ---- RMS ratio monitoring (negligible cost) ----
        with torch.no_grad():
            tp_mon = tp_flag.bool() if current_phase >= 2 else torch.ones(est_wav.shape[0], dtype=torch.bool, device=device)
            if tp_mon.any():
                rms_ratio = (est_wav[tp_mon].pow(2).mean().sqrt() / target_wav[tp_mon].pow(2).mean().sqrt().clamp(min=1e-8)).item()
            else:
                rms_ratio = 0.0

        # ---- Separation + STFT + L1 loss (always active) ----
        ta_weight = loss_w.get("ta_weight", 0.1)
        if current_phase == 1:
            loss_sep = si_sdr_loss(est_wav, target_wav)
        else:
            loss_sep = scene_aware_loss(est_wav, target_wav, tp_flag,
                                        ta_weight=ta_weight)

        if current_phase == 1:
            loss_stft = stft_loss_fn(est_wav, target_wav)
            loss_phase = phase_loss_fn(est_wav, target_wav)
            loss_l1 = l1_waveform_loss(est_wav, target_wav)
            loss_amp = amplitude_loss(est_wav, target_wav)
        else:
            tp_mask_stft = tp_flag.bool()
            if tp_mask_stft.any():
                loss_stft = stft_loss_fn(est_wav[tp_mask_stft], target_wav[tp_mask_stft])
                loss_phase = phase_loss_fn(est_wav[tp_mask_stft], target_wav[tp_mask_stft])
                loss_l1 = l1_waveform_loss(est_wav[tp_mask_stft], target_wav[tp_mask_stft])
                loss_amp = amplitude_loss(est_wav[tp_mask_stft], target_wav[tp_mask_stft])
            else:
                loss_stft = torch.tensor(0.0, device=device)
                loss_phase = torch.tensor(0.0, device=device)
                loss_l1 = torch.tensor(0.0, device=device)
                loss_amp = torch.tensor(0.0, device=device)

        # Amplitude loss warmup: ramp lambda_amp from 0 to full over 5K steps
        # after phase 1 ends (avoids optimizer shock from sudden new gradient)
        amp_warmup_steps = 5000
        if step < phase1_steps + amp_warmup_steps:
            amp_scale = max(0.0, (step - phase1_steps) / amp_warmup_steps) if step >= phase1_steps else 0.0
        else:
            amp_scale = 1.0

        loss_G = loss_w["lambda_sep"] * loss_sep + loss_w["lambda_stft"] * loss_stft \
            + loss_w.get("lambda_phase", 0.0) * loss_phase \
            + loss_w["lambda_l1"] * loss_l1 \
            + loss_w["lambda_amp"] * amp_scale * loss_amp

        # ---- GAN losses (phase 3 only) ----
        # D and G backward passes are separated so their graphs don't
        # coexist in GPU memory.  G backward runs first (freeing G graph),
        # then D forward+backward runs with only D's graph in memory.
        loss_D_val = 0.0
        loss_adv_val = 0.0
        loss_fm_val = 0.0
        gan_active = False

        gan_scale = 0.0
        if current_phase >= 3:
            tp_mask = tp_flag.bool()

            # GAN warmup: D-only pre-training, then linear ramp
            steps_into_phase3 = step - phase2_steps
            if steps_into_phase3 < gan_d_only_steps:
                gan_scale = 0.0  # D-only: no adversarial gradient to G
            elif steps_into_phase3 < gan_warmup_steps:
                gan_scale = (steps_into_phase3 - gan_d_only_steps) / max(gan_warmup_steps - gan_d_only_steps, 1)
            else:
                gan_scale = 1.0

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
                loss_G = loss_G + gan_scale * (loss_w["lambda_adv"] * loss_adv
                                               + loss_w["lambda_fm"] * loss_fm)
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
            # Compute gradient norm before clipping, then clip
            total_norm = clip_grad_norm_(generator.parameters(), grad_clip)
            # Log gradient norms (every log_interval)
            if step % log_interval == 0:
                # total_norm is the pre-clip norm; actual clipping happens in-place
                clipped = min(total_norm.item(), grad_clip)
                writer.add_scalar("grad/norm_before_clip", total_norm.item(), step)
                writer.add_scalar("grad/norm_after_clip", clipped, step)
                writer.add_scalar("grad/clipped", float(total_norm.item() > grad_clip), step)
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
                  " | phase_l {:.4f} | l1 {:.4f} | amp {:.4f} | adv {:.4f} | fm {:.4f}"
                  " | D {:.4f} | gan_s {:.2f} | rms {:.2f}"
                  " | rss {:.0f}MB | {:.2f}s".format(
                      step, current_phase, loss_G.item(), loss_sep.item(),
                      loss_stft.item(), loss_phase.item(), loss_l1.item(), loss_amp.item(),
                      loss_adv_val, loss_fm_val, loss_D_val,
                      gan_scale, rms_ratio, rss_mb, dt))
            writer.add_scalar("train/loss_G", loss_G.item(), step)
            writer.add_scalar("train/loss_sep", loss_sep.item(), step)
            writer.add_scalar("train/loss_stft", loss_stft.item(), step)
            writer.add_scalar("train/loss_phase", loss_phase.item(), step)
            writer.add_scalar("train/loss_l1", loss_l1.item(), step)
            writer.add_scalar("train/loss_amp", loss_amp.item(), step)
            writer.add_scalar("train/rms_ratio", rms_ratio, step)
            writer.add_scalar("train/lr", sched_G.get_last_lr()[0], step)
            writer.add_scalar("data/noisy_ref_prob", current_noisy_ref_prob, step)
            n_tp = tp_flag.sum().item()
            writer.add_scalar("data/n_tp_per_batch", n_tp, step)
            writer.add_scalar("data/all_ta_batch", float(n_tp == 0), step)
            if current_phase >= 3:
                writer.add_scalar("train/loss_D", loss_D_val, step)
                writer.add_scalar("train/loss_adv", loss_adv_val, step)
                writer.add_scalar("train/loss_fm", loss_fm_val, step)
                writer.add_scalar("train/gan_scale", gan_scale, step)

            # GAN stability warnings (phase 3 only)
            if current_phase >= 3 and gan_active:
                if loss_D_val < 0.01:
                    print("GAN_WARNING step {} | loss_D={:.4f} near zero — D may be too strong".format(
                        step, loss_D_val))
                if loss_adv_val > 50.0:
                    print("GAN_WARNING step {} | loss_adv={:.2f} very high — possible instability".format(
                        step, loss_adv_val))
                if rms_ratio < 0.5 or rms_ratio > 2.0:
                    print("GAN_WARNING step {} | rms_ratio={:.3f} drifting — check GAN impact".format(
                        step, rms_ratio))

        # ---- Checkpoint ----
        if step > 0 and step % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint_{:07d}.pt".format(step))
            save_checkpoint(ckpt_path, step, generator, discriminator,
                            opt_G, opt_D, sched_G, sched_D,
                            best_val_loss=best_val_loss,
                            patience_counter=patience_counter,
                            protected_steps=PROTECTED_STEPS)
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
                    print("  -> New best model, saved checkpoint_best.pt")
                else:
                    patience_counter += 1
                    print("  -> No improvement for {}/{} evals".format(
                        patience_counter, early_stop_patience))
                    if patience_counter >= early_stop_patience:
                        print("Early stopping triggered. Best val loss: {:.4f}".format(best_val_loss))
                        break

        # ---- Milestone evaluation ----
        if step in MILESTONE_STEPS:
            mini_evaluate(generator, val_loader, device, step)
            real_audio_check(generator, device, step, ckpt_dir)

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
