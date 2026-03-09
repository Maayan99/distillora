"""
Basis LoRA Bank for HyperDistill.

Maintains a bank of N basis LoRA adapters that are mixed via learned coefficients.
Each basis contains per-layer A and B matrices for all targeted modules.
"""

import logging
from math import sqrt

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

logger = logging.getLogger()


class BasisLoRABank(nn.Module):
    """Bank of N basis LoRA adapters mixed by coefficients.

    Parameters:
        n_basis: Number of basis LoRAs in the bank
        rank: LoRA rank for each basis
        module_specs: Dict mapping virtual_module_name -> (n_layers, d_in, d_out)
        per_module_routing: If True, learn per-module scaling of global coefficients
    """

    def __init__(
        self,
        n_basis: int = 8,
        rank: int = 8,
        module_specs: dict[str, tuple[int, int, int]] = None,
        per_module_routing: bool = True,
    ):
        super().__init__()
        self.n_basis = n_basis
        self.rank = rank
        self.module_specs = module_specs
        self.per_module_routing = per_module_routing

        if module_specs is None:
            raise ValueError("module_specs must be provided")

        # Create basis parameters: basis_A and basis_B for each module
        # Shape: (n_basis, n_layers, r, d)
        self.basis_A = nn.ParameterDict()
        self.basis_B = nn.ParameterDict()

        for vname, (n_layers, d_in, d_out) in module_specs.items():
            # A matrices: (n_basis, n_layers, r, d_in)
            self.basis_A[vname] = nn.Parameter(
                torch.empty(n_basis, n_layers, rank, d_in)
            )
            # B matrices: (n_basis, n_layers, r, d_out) — small random init
            # to break symmetry so each basis gets differentiated gradients
            self.basis_B[vname] = nn.Parameter(
                torch.empty(n_basis, n_layers, rank, d_out)
            )
            # Scale by rank only (not d_out). Use large std so basis output
            # dominates hyper at init (~10x), forcing coefficient head to learn
            # basis selection via strong gradient signal.
            nn.init.normal_(
                self.basis_B[vname],
                mean=0,
                std=0.03 / sqrt(rank),
            )

            # Initialize A ~ N(0, 0.2 / sqrt(d_in * r))
            nn.init.normal_(
                self.basis_A[vname],
                mean=0,
                std=0.2 / sqrt(d_in * rank),
            )

        # Per-module routing: learnable scaling vectors α_module ∈ R^{n_basis}
        # that modulate global coefficients per module
        if per_module_routing:
            self.module_routing = nn.ParameterDict({
                vname: nn.Parameter(torch.ones(n_basis))
                for vname in module_specs
            })

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"BasisLoRABank: {n_basis} bases, rank={rank}, "
            f"{len(module_specs)} modules, {total_params/1e6:.2f}M params"
        )

    def mix(
        self,
        logits: Float[Tensor, "batch n_basis"],
    ) -> dict[str, dict[str, Float[Tensor, "batch n_layers r d"]]]:
        """Mix basis LoRAs using logits from CoefficientHead.

        Args:
            logits: Pre-softmax logits (batch, n_basis)

        Returns:
            Dict mapping virtual_module_name -> {A: (batch, n_layers, r, d_in),
                                                   B: (batch, n_layers, r, d_out)}
        """
        lora_dict = {}

        for vname in self.module_specs:
            if self.per_module_routing:
                # Per-module modulated coefficients in logit space
                alpha = self.module_routing[vname]  # (n_basis,)
                scaled_logits = logits * alpha.unsqueeze(0)
            else:
                scaled_logits = logits

            if self.training:
                # Gumbel-softmax breaks symmetry and explores basis combinations
                c = F.gumbel_softmax(scaled_logits, tau=1.0, hard=False, dim=-1)
            else:
                c = F.softmax(scaled_logits, dim=-1)  # (batch, n_basis)

            # Weighted sum: c @ basis -> (batch, n_layers, r, d)
            # basis_A[vname]: (n_basis, n_layers, r, d_in)
            # c: (batch, n_basis) -> (batch, n_basis, 1, 1, 1)
            c_expanded = c[:, :, None, None, None]

            A = (c_expanded * self.basis_A[vname].unsqueeze(0)).sum(dim=1)
            B = (c_expanded * self.basis_B[vname].unsqueeze(0)).sum(dim=1)

            lora_dict[vname] = {"A": A, "B": B}

        return lora_dict

    def compute_diversity_loss(self) -> Float[Tensor, ""]:
        """Compute cosine similarity penalty between basis LoRAs.

        Penalizes all pairs (i,j) with i<j to encourage diversity.
        """
        total_sim = torch.tensor(0.0, device=next(self.parameters()).device)
        n_pairs = 0

        for vname in self.module_specs:
            # Flatten each basis: (n_basis, n_layers * r * d)
            A_flat = self.basis_A[vname].flatten(start_dim=1)  # (n_basis, -1)
            B_flat = self.basis_B[vname].flatten(start_dim=1)

            # Concatenate A and B for full basis representation
            basis_flat = torch.cat([A_flat, B_flat], dim=1)  # (n_basis, -1)

            # Normalize
            basis_norm = F.normalize(basis_flat, dim=1)

            # Compute pairwise cosine similarity
            sim_matrix = basis_norm @ basis_norm.T  # (n_basis, n_basis)

            # Sum upper triangle (excluding diagonal)
            mask = torch.triu(torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1)
            total_sim = total_sim + sim_matrix[mask].abs().sum()
            n_pairs += mask.sum().item()

        if n_pairs > 0:
            total_sim = total_sim / n_pairs

        return total_sim
