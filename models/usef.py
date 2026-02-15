"""USEF Cross-Multi-Head-Attention conditioning module + FiLM layer.

Computes cross-attention between mixture and reference speaker embeddings
in the sub-band feature space, enabling noise-robust speaker conditioning.
FiLMLayer provides complementary feature-wise linear modulation.
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation from speaker embedding.

    Projects a time-pooled reference embedding to per-band gamma (scale)
    and beta (shift) vectors, then applies: gamma * z_cond + beta.

    Initialized to identity transform (gamma=1, beta=0) so FiLM has no
    effect at the start of training.
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.gamma_proj = nn.Linear(feature_dim, feature_dim)
        self.beta_proj = nn.Linear(feature_dim, feature_dim)
        # Identity init: gamma_proj(any_input) = 1, beta_proj(any_input) = 0
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, z_cond, z_ref):
        """
        Args:
            z_cond: (B, T, N, D) conditioned mixture features
            z_ref:  (B, T_ref, N, D) reference features

        Returns:
            (B, T, N, D) FiLM-modulated features
        """
        # Time-pool reference: (B, T_ref, N, D) -> (B, N, D)
        spk = z_ref.mean(dim=1)
        gamma = self.gamma_proj(spk).unsqueeze(1)  # (B, 1, N, D)
        beta = self.beta_proj(spk).unsqueeze(1)     # (B, 1, N, D)
        return gamma * z_cond + beta


class USEFModule(nn.Module):
    """Per-band cross-attention conditioning.

    Reshapes (B, T, bands, D) to (B*bands, T, D) for memory-efficient
    cross-attention, then reshapes back.
    """

    def __init__(self, feature_dim, num_heads):
        """
        Args:
            feature_dim: feature dimension per band (D)
            num_heads: number of attention heads
        """
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, z_mix, z_ref):
        """
        Args:
            z_mix: (B, T_mix, num_bands, D) mixture features
            z_ref: (B, T_ref, num_bands, D) reference features

        Returns:
            (B, T_mix, num_bands, D) conditioned features
        """
        B, T_mix, N, D = z_mix.shape
        T_ref = z_ref.shape[1]

        # Merge batch and band dims: (B*N, T, D)
        q = z_mix.permute(0, 2, 1, 3).reshape(B * N, T_mix, D)
        kv = z_ref.permute(0, 2, 1, 3).reshape(B * N, T_ref, D)

        # Cross-attention: Q=mix, K=V=ref
        attn_out, _ = self.cross_attn(q, kv, kv)

        # Residual + LayerNorm
        out = self.norm(q + attn_out)

        # Reshape back: (B, T_mix, N, D)
        out = out.reshape(B, N, T_mix, D).permute(0, 2, 1, 3)
        return out
