"""
Online teacher module for cross-model distillation.
Wraps a frozen teacher model (Qwen3.5-4B) for autoregressive generation
and logprob extraction.
"""

import logging

import torch
from jaxtyping import Float, Integer
from torch import Tensor, nn
from transformers import AutoModelForCausalLM

logger = logging.getLogger()


class OnlineTeacher(nn.Module):
    """Frozen teacher model for online distillation.

    Generates responses autoregressively, then extracts top-k logprobs
    in a single forward pass over the full sequence.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        top_k: int = 16,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def config(self):
        return self.model.config

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def generate_response(
        self,
        input_ids: Integer[Tensor, "batch seq_len"],
        attention_mask: Integer[Tensor, "batch seq_len"],
    ) -> Integer[Tensor, "batch response_len"]:
        """Autoregressively generate a response."""
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            pad_token_id=self.model.config.eos_token_id,
        )
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        outputs = self.model.generate(**gen_kwargs)
        # outputs includes the prompt; extract only the generated part
        response_ids = outputs[:, input_ids.shape[1]:]
        return response_ids

    @torch.no_grad()
    def extract_logprobs(
        self,
        input_ids: Integer[Tensor, "batch full_seq_len"],
        attention_mask: Integer[Tensor, "batch full_seq_len"],
        response_start_idx: int,
    ) -> tuple[
        Float[Tensor, "batch response_len top_k"],
        Integer[Tensor, "batch response_len top_k"],
    ]:
        """Single forward pass to extract top-k logprobs at response positions.

        Args:
            input_ids: Full sequence (prompt + response)
            attention_mask: Attention mask for full sequence
            response_start_idx: Index where response tokens begin

        Returns:
            (top_k_logprobs, top_k_indices) at each response token position
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Logits at positions [response_start-1 .. end-1] predict response tokens
        # We want logprobs for predicting tokens at positions [response_start .. end]
        response_logits = outputs.logits[:, response_start_idx - 1:-1]

        # Compute log-softmax
        log_probs = torch.log_softmax(response_logits.float(), dim=-1)

        # Get top-k
        top_k_logprobs, top_k_indices = torch.topk(log_probs, self.top_k, dim=-1)

        return top_k_logprobs, top_k_indices

    @torch.no_grad()
    def generate_and_score(
        self,
        input_ids: Integer[Tensor, "batch seq_len"],
        attention_mask: Integer[Tensor, "batch seq_len"],
    ) -> tuple[
        Integer[Tensor, "batch response_len"],
        Float[Tensor, "batch response_len top_k"],
        Integer[Tensor, "batch response_len top_k"],
    ]:
        """Generate response and extract top-k logprobs.

        1. Autoregressively generate response given prompt
        2. Single forward pass over (prompt + response) for logprobs

        Args:
            input_ids: Prompt token IDs
            attention_mask: Prompt attention mask

        Returns:
            (response_ids, top_k_logprobs, top_k_indices)
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Step 1: Generate response
            response_ids = self.generate_response(input_ids, attention_mask)

            if response_ids.shape[1] == 0:
                # Empty response - return empty tensors
                batch = input_ids.shape[0]
                return (
                    response_ids,
                    torch.zeros(batch, 0, self.top_k, device=self.device),
                    torch.zeros(batch, 0, self.top_k, dtype=torch.long, device=self.device),
                )

            # Step 2: Concatenate prompt + response for scoring
            prompt_len = input_ids.shape[1]
            full_ids = torch.cat([input_ids, response_ids], dim=1)
            full_mask = torch.cat(
                [attention_mask, torch.ones_like(response_ids)],
                dim=1,
            )

            # Step 3: Extract logprobs at response positions
            top_k_logprobs, top_k_indices = self.extract_logprobs(
                full_ids, full_mask, response_start_idx=prompt_len
            )

        return response_ids, top_k_logprobs, top_k_indices


def load_teacher(
    model_name_or_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    top_k: int = 16,
    max_new_tokens: int = 256,
) -> OnlineTeacher:
    """Load a frozen teacher model."""
    logger.info(f"Loading teacher model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    teacher = OnlineTeacher(
        model=model,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
    )
    logger.info(f"Teacher model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return teacher
