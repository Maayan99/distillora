"""
Coefficient head and refinement blocks for HyperDistill.

CoefficientHead: pools perceiver output → 8 softmax coefficients for basis mixing.
RefinementBlocks: injects coefficient info back into latent queries via self-attention.
"""

import logging

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from ctx_to_lora.modeling.idefics2 import Idefics2Perceiver, Idefics2PerceiverConfig

logger = logging.getLogger()


class CoefficientHead(nn.Module):
    """Compute basis mixing logits from perceiver latent output.

    Input: Perceiver output (batch, n_queries, d_latent)
    Output: Logits (batch, n_basis) — softmax applied downstream in BasisLoRABank.mix()
    """

    def __init__(self, d_latent: int = 512, n_basis: int = 8):
        super().__init__()
        self.d_latent = d_latent
        self.n_basis = n_basis

        self.pool_norm = nn.LayerNorm(d_latent)
        self.linear = nn.Linear(d_latent, n_basis)

        # Initialize to produce near-uniform coefficients at start
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.linear.bias, mean=0, std=0.1)

    def forward(
        self,
        latents: Float[Tensor, "batch n_queries d_latent"],
    ) -> Float[Tensor, "batch n_basis"]:
        """Pool latents and produce basis mixing logits (pre-softmax)."""
        # Mean pool across query dimension
        pooled = latents.mean(dim=1)  # (batch, d_latent)
        pooled = self.pool_norm(pooled)

        # Project to n_basis logits (softmax applied in BasisLoRABank.mix)
        logits = self.linear(pooled)  # (batch, n_basis)

        return logits


class RefinementBlocks(nn.Module):
    """Inject coefficient information into latent queries and refine via self-attention.

    Projects coefficient vector to latent dimension, adds to each query,
    then runs self-attention blocks for refinement.
    """

    def __init__(
        self,
        d_latent: int = 512,
        n_basis: int = 8,
        n_blocks: int = 2,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.n_basis = n_basis

        # Project coefficients to latent space
        self.coeff_proj = nn.Linear(n_basis, d_latent)

        # Self-attention refinement blocks
        # Reuse the Idefics2Perceiver in self-attention-only mode
        # Each block is a self-attention layer
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                RefinementSelfAttnBlock(d_latent)
            )

    def forward(
        self,
        latents: Float[Tensor, "batch n_queries d_latent"],
        coefficients: Float[Tensor, "batch n_basis"],
    ) -> Float[Tensor, "batch n_queries d_latent"]:
        """Inject coefficient info and refine latents."""
        # Project coefficients to latent dimension
        coeff_embedding = self.coeff_proj(coefficients)  # (batch, d_latent)

        # Add to each query (broadcast)
        latents = latents + coeff_embedding.unsqueeze(1)

        # Run self-attention refinement
        for block in self.blocks:
            latents = block(latents)

        return latents


class RefinementSelfAttnBlock(nn.Module):
    """Single self-attention refinement block with pre-norm and residual."""

    def __init__(self, d_latent: int = 512):
        super().__init__()
        n_heads = max(1, d_latent // 64)
        self.norm = nn.LayerNorm(d_latent)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_latent,
            num_heads=n_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(d_latent)
        self.ff = nn.Sequential(
            nn.Linear(d_latent, d_latent * 4),
            nn.SiLU(),
            nn.Linear(d_latent * 4, d_latent),
        )

    def forward(
        self,
        x: Float[Tensor, "batch n_queries d_latent"],
    ) -> Float[Tensor, "batch n_queries d_latent"]:
        # Self-attention with pre-norm and residual
        normed = self.norm(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + attn_out

        # Feedforward with pre-norm and residual
        normed = self.ff_norm(x)
        x = x + self.ff(normed)

        return x
