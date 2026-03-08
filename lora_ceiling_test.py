#!/usr/bin/env python3
"""LoRA Ceiling Test: Train a standard PEFT LoRA to find the KL floor.

Trains a direct LoRA on Qwen3.5-0.8B to match a teacher model on a single
system prompt, bypassing the hypernetwork entirely. This establishes the
best-case KL achievable with rank-8 LoRA on these target modules.
"""

import argparse
import json
import random
from collections import defaultdict
from math import cos, pi

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_data(data_path: str, prompt_override: str | None = None):
    """Load QA pairs for the first system prompt group."""
    groups = defaultdict(list)
    with open(data_path) as f:
        for line in f:
            sample = json.loads(line)
            groups[sample["system_prompt_id"]].append(sample)

    # Pick first group
    first_id = next(iter(groups))
    samples = groups[first_id]
    context = samples[0]["context"] if not prompt_override else prompt_override

    qa_pairs = []
    for s in samples:
        qa_pairs.append({"question": s["question"], "answer": s["answer"]})

    print(f"Loaded {len(qa_pairs)} QA pairs for prompt: {first_id}")
    print(f"System prompt ({len(context)} chars): {context[:120]}...")
    return context, qa_pairs


def tokenize_qa(tokenizer, context: str, question: str, answer: str, max_len: int = 2048):
    """Tokenize a single QA pair as a chat conversation."""
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)

    # Find where the assistant response starts to compute loss only on response tokens
    # Tokenize without the answer to find the boundary
    messages_no_answer = [
        {"role": "system", "content": context},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages_no_answer, tokenize=False, add_generation_prompt=True
    )
    prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=max_len)["input_ids"])

    return tokens["input_ids"], tokens["attention_mask"], prompt_len


@torch.no_grad()
def compute_kl(student_model, teacher_model, input_ids, attention_mask, prompt_len, top_k=16):
    """Compute truncated KL divergence matching the HyperDistill trainer."""
    device = input_ids.device

    # Teacher forward
    teacher_out = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
    # Response positions: predict tokens at [prompt_len .. end]
    # Logits at position i predict token i+1, so we use positions [prompt_len-1 .. -2]
    teacher_logits = teacher_out.logits[:, prompt_len - 1:-1].float()
    teacher_logp = F.log_softmax(teacher_logits, dim=-1)
    top_k_logp, top_k_idx = torch.topk(teacher_logp, top_k, dim=-1)

    # Student forward
    student_out = student_model(input_ids=input_ids, attention_mask=attention_mask)
    student_logits = student_out.logits[:, prompt_len - 1:-1].float()

    # KL at teacher's top-k indices
    logq_denom = torch.logsumexp(student_logits, dim=-1, keepdim=True)
    logq_selected = student_logits.gather(-1, top_k_idx) - logq_denom
    p = top_k_logp.exp()
    kl = -(p * logq_selected).sum(dim=-1).mean()

    return kl.item()


def compute_kl_with_grad(student_model, teacher_model, input_ids, attention_mask, prompt_len, top_k=16):
    """Compute KL with gradient for training."""
    # Teacher forward (no grad)
    with torch.no_grad():
        teacher_out = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        teacher_logits = teacher_out.logits[:, prompt_len - 1:-1].float()
        teacher_logp = F.log_softmax(teacher_logits, dim=-1)
        top_k_logp, top_k_idx = torch.topk(teacher_logp, top_k, dim=-1)

    # Student forward (with grad)
    student_out = student_model(input_ids=input_ids, attention_mask=attention_mask)
    student_logits = student_out.logits[:, prompt_len - 1:-1].float()

    logq_denom = torch.logsumexp(student_logits, dim=-1, keepdim=True)
    logq_selected = student_logits.gather(-1, top_k_idx) - logq_denom
    p = top_k_logp.exp()
    loss = -(p * logq_selected).sum(dim=-1).mean()

    return loss


def evaluate_predictions(student_model, teacher_model, tokenizer, input_ids, attention_mask, prompt_len, n_positions=5):
    """Show teacher vs student top-5 predictions at sample positions."""
    with torch.no_grad():
        teacher_out = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        student_out = student_model(input_ids=input_ids, attention_mask=attention_mask)

    response_len = input_ids.shape[1] - prompt_len
    if response_len <= 0:
        print("  (response too short for eval)")
        return

    # Pick evenly spaced positions in the response
    positions = [prompt_len - 1 + int(i * (response_len - 1) / max(n_positions - 1, 1))
                 for i in range(min(n_positions, response_len))]

    for pos in positions:
        actual_token = tokenizer.decode([input_ids[0, pos + 1].item()])
        teacher_logits = teacher_out.logits[0, pos].float()
        student_logits = student_out.logits[0, pos].float()

        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)

        teacher_top5 = torch.topk(teacher_probs, 5)
        student_top5 = torch.topk(student_probs, 5)

        t_tokens = [tokenizer.decode([idx.item()]) for idx in teacher_top5.indices]
        s_tokens = [tokenizer.decode([idx.item()]) for idx in student_top5.indices]
        t_probs = [f"{p:.3f}" for p in teacher_top5.values.tolist()]
        s_probs = [f"{p:.3f}" for p in student_top5.values.tolist()]

        print(f"  pos {pos - prompt_len + 1:3d} | actual: {actual_token!r:10s} | "
              f"teacher: {list(zip(t_tokens, t_probs))}")
        print(f"         {'':10s} | "
              f"student: {list(zip(s_tokens, s_probs))}")


def main():
    parser = argparse.ArgumentParser(description="LoRA ceiling test for KL distillation")
    parser.add_argument("--teacher", default="4b",
                        help="Teacher model: '4b' for Qwen3.5-4B, or any HF model name")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--prompt", type=str, default=None, help="Override system prompt")
    parser.add_argument("--data", type=str, default="data/splits/train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Resolve teacher model name
    teacher_map = {"4b": "Qwen/Qwen3.5-4B"}
    teacher_name = teacher_map.get(args.teacher, args.teacher)

    student_name = "Qwen/Qwen3.5-0.8B"
    print(f"Student: {student_name}")
    print(f"Teacher: {teacher_name}")
    print(f"LoRA rank: {args.rank}, targets: [q_proj, v_proj, down_proj]")
    print(f"Training: {args.steps} steps, lr={args.lr}")
    print()

    # Load data
    context, qa_pairs = load_data(args.data, args.prompt)
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(student_name, trust_remote_code=True)

    # Pre-tokenize all QA pairs
    print("Tokenizing QA pairs...")
    tokenized_pairs = []
    for qa in qa_pairs:
        ids, mask, plen = tokenize_qa(tokenizer, context, qa["question"], qa["answer"])
        tokenized_pairs.append((ids.cuda(), mask.cuda(), plen))
    print(f"Tokenized {len(tokenized_pairs)} pairs, "
          f"lengths: {[ids.shape[1] for ids, _, _ in tokenized_pairs]}")
    print()

    # Load teacher (frozen)
    print("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    print(f"Teacher: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.1f}M params")

    # Load student
    print("Loading student model...")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_name, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    student_model.eval()
    print(f"Student: {sum(p.numel() for p in student_model.parameters()) / 1e6:.1f}M params")

    # Compute baseline KL (no LoRA)
    print("\n--- Baseline KL (no LoRA) ---")
    baseline_kls = []
    for ids, mask, plen in tokenized_pairs:
        kl = compute_kl(student_model, teacher_model, ids, mask, plen)
        baseline_kls.append(kl)
    baseline_kl = sum(baseline_kls) / len(baseline_kls)
    print(f"Baseline KL (avg over {len(baseline_kls)} pairs): {baseline_kl:.4f}")

    # Apply LoRA
    print("\n--- Applying LoRA ---")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,  # alpha = rank (matching config)
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj", "down_proj"],
        bias="none",
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()

    # Verify initial KL with LoRA (should be ~same as baseline since LoRA B=0)
    student_model.eval()
    init_kls = []
    for ids, mask, plen in tokenized_pairs:
        kl = compute_kl(student_model, teacher_model, ids, mask, plen)
        init_kls.append(kl)
    init_kl = sum(init_kls) / len(init_kls)
    print(f"Initial KL with LoRA (should ≈ baseline): {init_kl:.4f}")

    # Training
    print(f"\n--- Training for {args.steps} steps ---")
    student_model.train()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=0.01)

    for step in range(1, args.steps + 1):
        # Pick random QA pair
        ids, mask, plen = random.choice(tokenized_pairs)

        # Cosine LR schedule
        progress = step / args.steps
        lr = args.lr * 0.5 * (1 + cos(pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        loss = compute_kl_with_grad(student_model, teacher_model, ids, mask, plen)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5.0)
        optimizer.step()

        if step % 10 == 0 or step == 1:
            print(f"  step {step:4d}/{args.steps} | KL={loss.item():.4f} | lr={lr:.2e}")

    # Eval after training
    print(f"\n--- Eval after {args.steps} steps ---")
    student_model.eval()
    trained_kls = []
    for ids, mask, plen in tokenized_pairs:
        kl = compute_kl(student_model, teacher_model, ids, mask, plen)
        trained_kls.append(kl)
    trained_kl = sum(trained_kls) / len(trained_kls)
    print(f"Trained KL (avg over {len(trained_kls)} pairs): {trained_kl:.4f}")

    # Show predictions for first 5 QA pairs
    print(f"\n--- Token predictions (5 examples) ---")
    for i, (ids, mask, plen) in enumerate(tokenized_pairs[:5]):
        print(f"\nExample {i + 1}: {qa_pairs[i]['question'][:80]}...")
        evaluate_predictions(student_model, teacher_model, tokenizer, ids, mask, plen)

    # Summary
    reduction = (baseline_kl - trained_kl) / baseline_kl * 100
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Baseline KL (no LoRA):  {baseline_kl:.4f}")
    print(f"  Trained KL (LoRA):      {trained_kl:.4f}")
    print(f"  Reduction:              {reduction:.1f}%")
    print(f"  Steps:                  {args.steps}")
    print(f"  LoRA rank:              {args.rank}")
    print(f"  Teacher:                {teacher_name}")
    print("=" * 60)

    if trained_kl < 0.5:
        print("CONCLUSION: LoRA can reach KL < 0.5 → hypernetwork is the bottleneck")
    elif trained_kl < 1.0:
        print("CONCLUSION: LoRA reaches KL < 1.0 → hypernetwork has room to improve")
    elif trained_kl > 1.3:
        print("CONCLUSION: LoRA itself plateaus near 1.5 → capacity ceiling, not hypernetwork")
    else:
        print("CONCLUSION: Marginal improvement — may need more steps or higher rank")


if __name__ == "__main__":
    main()
