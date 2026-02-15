I've now read the complete v2 source code and all documentation. Here's my review of the `implementation_plan.md`:

### Issues and Concerns

**1. FiLMLayer init is incorrect (Step 2)**

The improvement plan's FiLMLayer code says:
```python
nn.init.ones_(self.gamma_proj.weight.data[:feature_dim, :feature_dim].diagonal())
```
This doesn't produce an identity transform — `nn.Linear(D, D)` with an identity weight means `gamma = input`, not `gamma = 1`. The plan should use:
```python
nn.init.zeros_(self.gamma_proj.weight)
nn.init.ones_(self.gamma_proj.bias)
```
This way `gamma_proj(spk) = 1` at init regardless of input, giving the identity transform `1 * z_cond + 0 = z_cond`.

**2. Conditioning dropout placement unclear (Step 3)**

The plan says "zero out z_ref per-sample" in `generator.py`, but it doesn't specify whether this applies to just the USEF input or also the TFGridNet reinject inputs. Zeroing z_ref before USEF would disable both USEF cross-attention AND all reinject layers AND FiLM in one go — a very strong dropout. Alternatively, you could zero it only at the FiLM stage. The implementation needs to be precise about scope.

**3. Speed perturbation with scipy-based pipeline (Step 5)**

The plan uses `torchaudio.sox_effects.apply_effects_tensor`, but `dataset.py` operates entirely in numpy (scipy FFT convolution). Converting numpy→tensor→sox→numpy adds overhead in the dataloader workers. Since `torchaudio.sox_effects` may not be available in all builds, a `scipy.signal.resample` or `librosa.effects.time_stretch` approach in pure numpy would be more robust and consistent with the existing pipeline.

**4. Mixed precision scope needs care (Change 9)**

The plan correctly flags that SI-SDR needs FP32 (log10, ratio ops), but doesn't mention that the STFT/iSTFT in `audio_utils.py` also create `hann_window` on every forward call. Under autocast, these windows may be created in bf16. The `PhaseSensitiveLoss` also calls `torch.stft` which may need explicit FP32 casting.

**5. EMA integration with validation (Change 8)**

The plan says EMA checkpoint should be used for evaluation, but the validation function in `train.py` currently uses the live generator. The plan should clarify: validate with EMA weights (swap in, validate, swap back) or maintain a separate EMA model for validation.

**6. Checkpoint format breaking change (Step 6)**

The plan removes discriminator state from checkpoints and adds EMA state. This means the `load_checkpoint` function signature and dict keys change. `evaluate.py` and `inference.py` only load `ckpt["generator"]`, so they're unaffected — that's correct.

**7. Realistic impact estimates**

The Codex o3 review correctly noted the combined +8-10 dB estimate is optimistic (individual estimates appear additive but interact non-linearly). The conservative +4.5 to +5.5 dB estimate in the plan's context section is more credible. This is a documentation note, not a code issue.

### Missing Items

- **No change to `evaluate.py` or `inference.py`** — This is correct since they only load generator weights and don't reference the discriminator or loss config.
- **`losses/__init__.py`** and `models/__init__.py` are empty — no changes needed.
- The `discriminator.py` file is kept in the codebase for future GAN fine-tuning — reasonable.
- **No changes to data packing scripts** (`prepare_manifest.py`, `pack_noise_rir.py`, `pack_noise_only.py`) — correct, HDF5 data is already packed.

### Summary

The plan is well-structured and technically sound for the most part. The main implementation risks are:
1. Getting FiLMLayer initialization right (currently wrong in the plan)
2. Properly scoping conditioning dropout
3. Ensuring mixed precision doesn't corrupt STFT/iSTFT numerics