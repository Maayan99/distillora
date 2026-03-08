"""
Model inspection script for Qwen3.5-0.8B and Qwen3.5-4B.
Identifies projection names, shapes, layer types (DeltaNet vs Full Attention),
and outputs a constants dict for use by all subsequent code.
"""

import json
import sys
from collections import defaultdict

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def inspect_model(model_name: str):
    print(f"\n{'='*80}")
    print(f"Inspecting: {model_name}")
    print(f"{'='*80}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    config = model.config
    print(f"\nConfig type: {type(config).__name__}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num hidden layers: {config.num_hidden_layers}")
    print(f"Num attention heads: {config.num_attention_heads}")
    print(f"Num KV heads: {getattr(config, 'num_key_value_heads', 'N/A')}")
    print(f"Intermediate size: {config.intermediate_size}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Max position embeddings: {getattr(config, 'max_position_embeddings', 'N/A')}")

    # Check for DeltaNet-specific config
    for attr in ["attention_type", "layer_types", "attn_layer_indices",
                  "attention_pattern", "chunk_size", "use_gate", "use_short_conv"]:
        if hasattr(config, attr):
            val = getattr(config, attr)
            if isinstance(val, (list, tuple)) and len(val) > 10:
                print(f"{attr}: {val[:5]}...{val[-5:]} (len={len(val)})")
            else:
                print(f"{attr}: {val}")

    # Identify layer types
    print(f"\n--- Layer structure ---")
    layers = model.model.layers
    layer_types = {}
    deltanet_indices = []
    full_attn_indices = []

    for i, layer in enumerate(layers):
        layer_type = type(layer).__name__
        attn_type = type(layer.self_attn).__name__ if hasattr(layer, "self_attn") else "N/A"
        if i < 3 or i >= len(layers) - 1:
            print(f"  Layer {i}: {layer_type}, attn: {attn_type}")
        elif i == 3:
            print(f"  ...")

        layer_types[i] = attn_type
        if "delta" in attn_type.lower() or "gated" in attn_type.lower():
            deltanet_indices.append(i)
        else:
            full_attn_indices.append(i)

    print(f"\nDeltaNet layers ({len(deltanet_indices)}): {deltanet_indices}")
    print(f"Full Attention layers ({len(full_attn_indices)}): {full_attn_indices}")

    # Inspect all Linear modules per layer type
    print(f"\n--- Linear modules per layer type ---")
    linears_by_type = defaultdict(lambda: defaultdict(list))

    for i, layer in enumerate(layers):
        attn_type = layer_types[i]
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                key = (name, module.in_features, module.out_features)
                linears_by_type[attn_type][key].append(i)

    for attn_type, modules in linears_by_type.items():
        print(f"\n  [{attn_type}]")
        for (name, d_in, d_out), layer_ids in sorted(modules.items()):
            count = len(layer_ids)
            sample = layer_ids[:3]
            print(f"    {name:30s}  in={d_in:5d}  out={d_out:5d}  count={count}  layers={sample}{'...' if count > 3 else ''}")

    # Detailed first layer of each type
    for layer_idx_set, label in [(deltanet_indices[:1], "DeltaNet"), (full_attn_indices[:1], "FullAttn")]:
        if not layer_idx_set:
            continue
        idx = layer_idx_set[0]
        layer = layers[idx]
        print(f"\n  Detailed [{label}] layer {idx}:")
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                print(f"    {name:40s}  ({module.in_features} -> {module.out_features})")
            elif isinstance(module, (nn.Conv1d,)):
                print(f"    {name:40s}  Conv1d(in={module.in_channels}, out={module.out_channels}, kernel={module.kernel_size})")

    return {
        "model_name": model_name,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_kv_heads": getattr(config, "num_key_value_heads", None),
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "deltanet_indices": deltanet_indices,
        "full_attn_indices": full_attn_indices,
        "layer_types": layer_types,
        "linears_by_type": {
            attn_type: {
                name: {"in": d_in, "out": d_out, "count": len(ids), "layers": ids}
                for (name, d_in, d_out), ids in modules.items()
            }
            for attn_type, modules in linears_by_type.items()
        },
    }


def check_tokenizer(model_names):
    print(f"\n{'='*80}")
    print("Tokenizer comparison")
    print(f"{'='*80}")

    tokenizers = {}
    for name in model_names:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        tokenizers[name] = tok
        print(f"\n{name}:")
        print(f"  Vocab size: {tok.vocab_size}")
        print(f"  Model max length: {tok.model_max_length}")
        print(f"  Special tokens: {tok.special_tokens_map}")

    names = list(tokenizers.keys())
    if len(names) == 2:
        t1, t2 = tokenizers[names[0]], tokenizers[names[1]]
        same_vocab = t1.vocab_size == t2.vocab_size
        test_text = "Hello world, this is a test. The quick brown fox jumps."
        enc1 = t1.encode(test_text)
        enc2 = t2.encode(test_text)
        same_encoding = enc1 == enc2
        print(f"\nSame vocab size: {same_vocab}")
        print(f"Same encoding for test text: {same_encoding}")
        if not same_encoding:
            print(f"  {names[0]}: {enc1[:10]}...")
            print(f"  {names[1]}: {enc2[:10]}...")


def generate_constants(info_08b, info_4b):
    """Generate the constants dict used by all subsequent code."""
    print(f"\n{'='*80}")
    print("Generated constants")
    print(f"{'='*80}")

    # Use 0.8B model info for student constants
    info = info_08b
    constants = {
        "STUDENT_MODEL": info["model_name"],
        "TEACHER_MODEL": info_4b["model_name"],
        "STUDENT_HIDDEN_SIZE": info["hidden_size"],
        "TEACHER_HIDDEN_SIZE": info_4b["hidden_size"],
        "STUDENT_NUM_LAYERS": info["num_hidden_layers"],
        "TEACHER_NUM_LAYERS": info_4b["num_hidden_layers"],
        "STUDENT_INTERMEDIATE_SIZE": info["intermediate_size"],
        "DELTANET_LAYER_INDICES": info["deltanet_indices"],
        "FULL_ATTN_LAYER_INDICES": info["full_attn_indices"],
        "ALL_LAYER_INDICES": list(range(info["num_hidden_layers"])),
    }

    print(json.dumps(constants, indent=2))
    return constants


if __name__ == "__main__":
    models = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"]

    results = {}
    for model_name in models:
        try:
            results[model_name] = inspect_model(model_name)
        except Exception as e:
            print(f"Error inspecting {model_name}: {e}")
            import traceback
            traceback.print_exc()

    check_tokenizer(models)

    if len(results) == 2:
        constants = generate_constants(results[models[0]], results[models[1]])
