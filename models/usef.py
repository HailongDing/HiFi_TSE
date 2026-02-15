"""USEF Cross-Multi-Head-Attention conditioning module.

Computes cross-attention between mixture and reference speaker embeddings
in the sub-band feature space, enabling noise-robust speaker conditioning.
"""

import torch
import torch.nn as nn


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
