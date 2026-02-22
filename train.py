#!/usr/bin/env python3
"""HiFi-TSE v2 training script with 2-phase curriculum learning.

Run with the USEF-TFGridNet conda environment:
    conda run -n USEF-TFGridNet python train.py --config configs/hifi_tse.yaml

Phase 1 (0 - 200k steps):    TP only, SI-SDR + phase loss
Phase 2 (200k - 1.5M steps): TP + TA, scene-aware SI-SDR + phase loss

v2 changes vs v1:
  - GAN removed (no discriminator, no adversarial/feature matching losses)
  - Loss simplified to SI-SDR + phase-sensitive loss only
  - EMA of generator weights for validation/checkpointing
  - Mixed precision (bfloat16) with SI-SDR/phase loss in FP32
  - min_lr floor in cosine schedule
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
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from data.dataset import HiFiTSEDataset, tse_collate_fn
from losses.separation import amplitude_loss, scene_aware_loss, si_sdr, si_sdr_loss
from losses.stft_loss import PhaseSensitiveLoss
from models.generator import Generator


def infinite_loader(loader):
    """Yield batches endlessly, restarting the loader when exhausted."""
    while True:
        for batch in loader:
            yield batch


def _worker_init_fn(worker_id):
    """Seed each DataLoader worker independently to avoid correlated augmentations."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    seed = torch.initial_seed() % 2**32 + worker_id
    random.seed(seed)
    np.random.seed(seed)


def make_train_loader(dataset, batch_size):
    """Create a DataLoader for training (recreated at phase transitions)."""
    return DataLoader(
        dataset, batch_size=batch_size,
        num_workers=8, pin_memory=True, shuffle=True, drop_last=True,
        collate_fn=tse_collate_fn,
        worker_init_fn=_worker_init_fn,
        persistent_workers=False,
        prefetch_factor=4,
    )


def _step_from_filename(fname):
    """Extract step number from checkpoint_XXXXXXX.pt filename, or None."""
    base = os.path.splitext(fname)[0]  # checkpoint_0100000
    parts = base.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return None


def save_checkpoint(path, step, generator, opt_G, sched_G,
                    ema_generator=None, best_val_loss=None,
                    best_val_si_sdr=None, patience_counter=0,
                    keep_last=5, protected_steps=None):
    """Save training checkpoint and remove old ones beyond keep_last."""
    ckpt_dir = os.path.dirname(path)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = {
        "step": step,
        "generator": generator.state_dict(),
        "opt_G": opt_G.state_dict(),
        "sched_G": sched_G.state_dict(),
        "best_val_loss": best_val_loss,
        "best_val_si_sdr": best_val_si_sdr,
        "patience_counter": patience_counter,
    }
    if ema_generator is not None:
        ckpt["ema_generator"] = ema_generator.state_dict()
    torch.save(ckpt, path)

    if protected_steps is None:
        protected_steps = set()

    # Rotate: keep only the last N numbered checkpoints (skip protected + best)
    special = {"checkpoint_final.pt", "checkpoint_best.pt"}
    numbered = sorted(
        f for f in os.listdir(ckpt_dir)
        if f.startswith("checkpoint_") and f.endswith(".pt") and f not in special
    )
    deletable = [f for f in numbered if _step_from_filename(f) not in protected_steps]
    while len(deletable) > keep_last:
        old = os.path.join(ckpt_dir, deletable.pop(0))
        os.remove(old)
        print("Removed old checkpoint:", old)


def load_checkpoint(path, generator, opt_G, sched_G, device,
                    ema_generator=None, reset_optimizer=False):
    """Load training checkpoint. Returns (step, best_val_loss, best_val_si_sdr, patience_counter)."""
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    if ema_generator is not None:
        if "ema_generator" in ckpt:
            ema_generator.load_state_dict(ckpt["ema_generator"])
        else:
            # Bootstrap EMA from generator if checkpoint lacks EMA state
            ema_generator.module.load_state_dict(ckpt["generator"])
    if reset_optimizer:
        print("  Optimizer state RESET (not loaded from checkpoint)")
        step = ckpt["step"]
        sched_steps = ckpt["sched_G"]["last_epoch"]
        for _ in range(sched_steps):
            sched_G.step()
    else:
        opt_G.load_state_dict(ckpt["opt_G"])
        sched_G.load_state_dict(ckpt["sched_G"])
    best_val_loss = ckpt.get("best_val_loss", None)
    best_val_si_sdr = ckpt.get("best_val_si_sdr", None)
    patience_counter = ckpt.get("patience_counter", 0)
    return ckpt["step"], best_val_loss, best_val_si_sdr, patience_counter


def validate(model, val_loader, device, writer, step, ta_weight=0.1,
             use_amp=False, amp_dtype=torch.bfloat16,
             max_batches=200, seed=42):
    """Run validation and log decomposed metrics.

    Returns (avg_loss, avg_si_sdr) — avg_si_sdr is used for best model selection.
    Uses a fixed seed for reproducible dynamic mixing across evaluations.
    """
    model.eval()

    # Save RNG states and set fixed seed for deterministic validation
    rng_state_py = random.getstate()
    rng_state_np = np.random.get_state()
    rng_state_torch = torch.random.get_rng_state()
    rng_state_cuda = torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

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
            with autocast(dtype=amp_dtype, enabled=use_amp):
                est_wav = model(mix_wav, ref_wav)
            est_wav = est_wav.float()
            target_wav = target_wav.float()
            loss = scene_aware_loss(est_wav, target_wav, tp_flag,
                                    ta_weight=ta_weight)
            total_loss += loss.item()

            # Decomposed metrics
            tp_mask = tp_flag.bool()
            if tp_mask.any():
                n_tp = tp_mask.sum().item()
                sdr_vals = si_sdr(est_wav[tp_mask], target_wav[tp_mask])
                total_si_sdr += sdr_vals.sum().item()
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
            if count >= max_batches:
                break

    # Restore RNG states so training randomness is unaffected
    random.setstate(rng_state_py)
    np.random.set_state(rng_state_np)
    torch.random.set_rng_state(rng_state_torch)
    if rng_state_cuda is not None:
        torch.cuda.set_rng_state(rng_state_cuda, device)

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

    model.train()
    return avg_loss, avg_si_sdr


def mini_evaluate(model, val_loader, device, step, use_amp=False, amp_dtype=torch.bfloat16):
    """Quick evaluation at milestones: 10 batches, report SI-SDR + rms_ratio."""
    model.eval()
    total_si_sdr = 0.0
    total_rms_num = 0.0
    total_rms_den = 0.0
    n_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 10:
                break
            mix_wav, ref_wav, target_wav, tp_flag = [x.to(device) for x in batch]
            with autocast(dtype=amp_dtype, enabled=use_amp):
                est_wav = model(mix_wav, ref_wav)
            est_wav = est_wav.float()
            target_wav = target_wav.float()
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

    model.train()
    return avg_si_sdr, avg_rms_ratio


def real_audio_check(model, device, step, ckpt_dir, use_amp=False, amp_dtype=torch.bfloat16):
    """Run inference on real audio files and save output wav at milestones."""
    import torchaudio

    mix_path = "real_audio/mix_16k.wav"
    ref_path = "real_audio/ref_16k.wav"
    if not os.path.exists(mix_path) or not os.path.exists(ref_path):
        print("REAL_AUDIO_CHECK step {} | skipped (files not found)".format(step))
        return

    model.eval()
    with torch.no_grad():
        mix_wav, mix_sr = torchaudio.load(mix_path)
        ref_wav, ref_sr = torchaudio.load(ref_path)

        if mix_sr != 48000:
            mix_wav = torchaudio.functional.resample(mix_wav, mix_sr, 48000)
        if ref_sr != 48000:
            ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, 48000)

        max_samples = 192000
        if mix_wav.shape[-1] > max_samples:
            mix_wav = mix_wav[..., :max_samples]
        if ref_wav.shape[-1] > max_samples:
            ref_wav = ref_wav[..., :max_samples]

        mix_wav = mix_wav.unsqueeze(0).to(device)
        ref_wav = ref_wav.unsqueeze(0).to(device)

        if mix_wav.dim() == 3:
            mix_wav = mix_wav[:, 0, :]
        if ref_wav.dim() == 3:
            ref_wav = ref_wav[:, 0, :]

        with autocast(dtype=amp_dtype, enabled=use_amp):
            est_wav = model(mix_wav, ref_wav)
        est_wav = est_wav.float()

        est_rms = est_wav.pow(2).mean().sqrt().item()
        mix_rms = mix_wav.float().pow(2).mean().sqrt().item()
        ratio = est_rms / max(mix_rms, 1e-8)

        print("REAL_AUDIO_CHECK step {} | est_rms: {:.4f} | mix_rms: {:.4f} | ratio: {:.3f}".format(
            step, est_rms, mix_rms, ratio))

        out_path = os.path.join(ckpt_dir, "real_audio_step_{:07d}.wav".format(step))
        torchaudio.save(out_path, est_wav.cpu().squeeze(0).unsqueeze(0), 48000)
        print("  -> Saved:", out_path)

    model.train()


def main():
    parser = argparse.ArgumentParser(description="Train HiFi-TSE v2")
    parser.add_argument("--config", type=str, default="configs/hifi_tse.yaml")
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="Reset Adam state when resuming")
    parser.add_argument("--reset-patience", action="store_true",
                        help="Reset best_val_si_sdr and patience counter when resuming")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- AMP setup ----
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

    # ---- Model ----
    generator = Generator(cfg).to(device)

    g_params = sum(p.numel() for p in generator.parameters())
    print("Generator params: {:.2f}M".format(g_params / 1e6))

    # ---- EMA ----
    train_cfg = cfg["training"]
    ema_decay = train_cfg.get("ema_decay", 0.999)

    def ema_avg_fn(avg_param, param, num_averaged):
        return avg_param * ema_decay + param * (1.0 - ema_decay)

    ema_generator = AveragedModel(generator, avg_fn=ema_avg_fn).to(device)

    # ---- Optimizer ----
    opt_G = AdamW(generator.parameters(), lr=train_cfg["lr"],
                  betas=tuple(train_cfg["betas"]))
    total_steps = train_cfg["total_steps"]
    grad_accum = train_cfg["grad_accum_steps"]
    optimizer_steps = total_steps // grad_accum
    warmup_steps = train_cfg.get("warmup_steps", 5000)  # optimizer steps
    min_lr = train_cfg.get("min_lr", 1e-5)
    min_lr_scale = min_lr / train_cfg["lr"]

    def warmup_cosine_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(warmup_steps, 1)
        progress = (current_step - warmup_steps) / max(optimizer_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=warmup_cosine_lambda)

    # ---- Loss ----
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

    # Validation
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
    best_val_si_sdr = None  # TP-only SI-SDR for best model selection
    patience_counter = 0
    resume_path = cfg["checkpoint"].get("resume")
    if resume_path and os.path.exists(resume_path):
        print("Resuming from", resume_path)
        start_step, best_val_loss, best_val_si_sdr, patience_counter = load_checkpoint(
            resume_path, generator, opt_G, sched_G, device,
            ema_generator=ema_generator,
            reset_optimizer=args.reset_optimizer,
        )
        if args.reset_patience:
            best_val_si_sdr = None
            patience_counter = 0
            print("  Patience counter and best_val_si_sdr RESET")
        print("Resumed at step {} (best_val_loss={}, best_val_si_sdr={}, patience={})".format(
            start_step, best_val_loss, best_val_si_sdr, patience_counter))

    # ---- Curriculum config ----
    phase1_steps = cfg["curriculum"]["phase1_steps"]
    grad_clip = train_cfg["grad_clip"]
    log_interval = train_cfg["log_interval"]
    save_interval = train_cfg["save_interval"]
    val_interval = train_cfg["val_interval"]

    current_phase = 1
    if start_step >= phase1_steps:
        current_phase = 2
        train_dataset.set_phase(2)
        train_dataset.close_handles()
        del train_iter, train_loader
        gc.collect()
        train_loader = make_train_loader(train_dataset, train_cfg["batch_size"])
        train_iter = infinite_loader(train_loader)

    # Phase boundary checkpoints are never deleted by rotation
    PROTECTED_STEPS = {phase1_steps}
    MILESTONE_STEPS = {5000, phase1_steps, 200000, 400000, 600000, 800000,
                       1000000, 1200000, total_steps}

    print("Starting training from step {} (phase {})".format(start_step, current_phase))
    print("EMA decay: {}, AMP: {} ({})".format(ema_decay, use_amp, amp_dtype))

    # ---- Training loop ----
    generator.train()

    for step in range(start_step, total_steps):
        t0 = time.time()

        # Phase transition: Phase 1 -> Phase 2
        if step == phase1_steps and current_phase < 2:
            current_phase = 2
            train_dataset.set_phase(2)
            train_dataset.close_handles()
            del train_iter, train_loader
            gc.collect()
            train_loader = make_train_loader(train_dataset, train_cfg["batch_size"])
            train_iter = infinite_loader(train_loader)
            print("==> Phase 2: enabling TA data + scene-aware loss (step {})".format(step))

        # Update noisy_ref_prob ramp
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

        # ---- Forward (AMP) ----
        with autocast(dtype=amp_dtype, enabled=use_amp):
            est_wav = generator(mix_wav, ref_wav)

        # ---- RMS ratio monitoring ----
        with torch.no_grad():
            tp_mon = tp_flag.bool() if current_phase >= 2 else torch.ones(est_wav.shape[0], dtype=torch.bool, device=device)
            if tp_mon.any():
                rms_ratio = (est_wav[tp_mon].float().pow(2).mean().sqrt()
                             / target_wav[tp_mon].float().pow(2).mean().sqrt().clamp(min=1e-8)).item()
            else:
                rms_ratio = 0.0

        # ---- Loss computation (FP32 for SI-SDR, FP32 inputs for phase loss) ----
        est_wav_fp32 = est_wav.float()
        target_wav_fp32 = target_wav.float()

        ta_weight = loss_w.get("ta_weight", 0.1)
        if current_phase == 1:
            loss_sep = si_sdr_loss(est_wav_fp32, target_wav_fp32)
        else:
            loss_sep = scene_aware_loss(est_wav_fp32, target_wav_fp32, tp_flag,
                                        ta_weight=ta_weight)

        # Phase-sensitive loss (TP only in phase 2+)
        if current_phase == 1:
            loss_phase = phase_loss_fn(est_wav_fp32, target_wav_fp32)
        else:
            tp_mask_stft = tp_flag.bool()
            if tp_mask_stft.any():
                loss_phase = phase_loss_fn(est_wav_fp32[tp_mask_stft],
                                           target_wav_fp32[tp_mask_stft])
            else:
                loss_phase = torch.tensor(0.0, device=device)

        loss_G = loss_w["lambda_sep"] * loss_sep + loss_w.get("lambda_phase", 0.5) * loss_phase

        # Amplitude loss for TP samples (anchor RMS ratio near 1.0)
        lambda_amp = loss_w.get("lambda_amp", 0.0)
        if lambda_amp > 0:
            if current_phase == 1:
                loss_amp = amplitude_loss(est_wav_fp32, target_wav_fp32)
            else:
                tp_mask_amp = tp_flag.bool()
                if tp_mask_amp.any():
                    loss_amp = amplitude_loss(est_wav_fp32[tp_mask_amp],
                                              target_wav_fp32[tp_mask_amp])
                else:
                    loss_amp = torch.tensor(0.0, device=device)
            loss_G = loss_G + lambda_amp * loss_amp
        else:
            loss_amp = torch.tensor(0.0, device=device)

        # ---- Backward ----
        (loss_G / grad_accum).backward()

        # ---- Accumulate & step ----
        if (step + 1) % grad_accum == 0:
            total_norm = clip_grad_norm_(generator.parameters(), grad_clip)
            if step % log_interval == 0:
                clipped = min(total_norm.item(), grad_clip)
                writer.add_scalar("grad/norm_before_clip", total_norm.item(), step)
                writer.add_scalar("grad/norm_after_clip", clipped, step)
                writer.add_scalar("grad/clipped", float(total_norm.item() > grad_clip), step)
            opt_G.step()
            ema_generator.update_parameters(generator)
            opt_G.zero_grad()
            sched_G.step()

        dt = time.time() - t0

        # ---- Logging ----
        if step % log_interval == 0:
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print("step {:>7d} | phase {} | loss_G {:.4f} | sep {:.4f}"
                  " | phase_l {:.4f} | amp_l {:.4f} | rms {:.2f}"
                  " | lr {:.2e} | rss {:.0f}MB | {:.2f}s".format(
                      step, current_phase, loss_G.item(), loss_sep.item(),
                      loss_phase.item(), loss_amp.item(), rms_ratio,
                      sched_G.get_last_lr()[0], rss_mb, dt))
            writer.add_scalar("train/loss_G", loss_G.item(), step)
            writer.add_scalar("train/loss_sep", loss_sep.item(), step)
            writer.add_scalar("train/loss_phase", loss_phase.item(), step)
            writer.add_scalar("train/loss_amp", loss_amp.item(), step)
            writer.add_scalar("train/rms_ratio", rms_ratio, step)
            writer.add_scalar("train/lr", sched_G.get_last_lr()[0], step)
            writer.add_scalar("data/noisy_ref_prob", current_noisy_ref_prob, step)
            n_tp = tp_flag.sum().item()
            writer.add_scalar("data/n_tp_per_batch", n_tp, step)
            writer.add_scalar("data/all_ta_batch", float(n_tp == 0), step)

        # ---- Checkpoint ----
        if step > 0 and step % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint_{:07d}.pt".format(step))
            save_checkpoint(ckpt_path, step, generator, opt_G, sched_G,
                            ema_generator=ema_generator,
                            best_val_loss=best_val_loss,
                            best_val_si_sdr=best_val_si_sdr,
                            patience_counter=patience_counter,
                            protected_steps=PROTECTED_STEPS)
            print("Saved checkpoint:", ckpt_path)

        # ---- Validation (using EMA model) + best model tracking ----
        if step > 0 and step % val_interval == 0:
            val_loss, val_si_sdr = validate(
                ema_generator, val_loader, device, writer, step,
                ta_weight=ta_weight, use_amp=use_amp, amp_dtype=amp_dtype)

            # Best model tracking by TP-only SI-SDR (higher is better)
            early_stop_patience = 20  # stop after 20 validations without improvement
            if current_phase >= 2:
                if best_val_si_sdr is None or val_si_sdr > best_val_si_sdr:
                    best_val_si_sdr = val_si_sdr
                    patience_counter = 0
                    best_path = os.path.join(ckpt_dir, "checkpoint_best.pt")
                    save_checkpoint(best_path, step, generator, opt_G, sched_G,
                                    ema_generator=ema_generator,
                                    best_val_loss=best_val_loss,
                                    best_val_si_sdr=best_val_si_sdr,
                                    patience_counter=patience_counter,
                                    keep_last=999)
                    print("  -> New best SI-SDR ({:.2f} dB), saved checkpoint_best.pt".format(val_si_sdr))
                else:
                    patience_counter += 1
                    print("  -> No SI-SDR improvement for {}/{} evals (best: {:.2f} dB)".format(
                        patience_counter, early_stop_patience, best_val_si_sdr))
                    if patience_counter >= early_stop_patience:
                        print("Early stopping triggered. Best val SI-SDR: {:.2f} dB".format(best_val_si_sdr))
                        break

        # ---- Milestone evaluation (using EMA model) ----
        if step in MILESTONE_STEPS:
            mini_evaluate(ema_generator, val_loader, device, step,
                          use_amp=use_amp, amp_dtype=amp_dtype)
            real_audio_check(ema_generator, device, step, ckpt_dir,
                             use_amp=use_amp, amp_dtype=amp_dtype)

    # Final checkpoint
    final_path = os.path.join(ckpt_dir, "checkpoint_final.pt")
    save_checkpoint(final_path, total_steps, generator, opt_G, sched_G,
                    ema_generator=ema_generator,
                    best_val_loss=best_val_loss,
                    best_val_si_sdr=best_val_si_sdr,
                    patience_counter=patience_counter)
    print("Training complete. Final checkpoint:", final_path)
    writer.close()


if __name__ == "__main__":
    main()
