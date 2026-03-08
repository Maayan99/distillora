"""
Model architecture constants for Qwen3.5 DeltaNet/Full Attention models.

Derived from model inspection (scripts/inspect_models.py, 2026-03-08).

Qwen3.5-0.8B (student): 24 layers
  - 18 DeltaNet (linear_attention) layers: 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22
  - 6 Full Attention layers: 3, 7, 11, 15, 19, 23

  DeltaNet layers have NO self_attn.{q,k,v,o}_proj.
  Instead they use:
    linear_attn.in_proj_qkv  (1024 -> 6144)  — fused QKV
    linear_attn.out_proj     (2048 -> 1024)
    linear_attn.in_proj_z    (1024 -> 2048)
    linear_attn.in_proj_a/b  (1024 -> 16)

  Full Attention layers use standard GQA (8 heads, 2 KV heads, head_dim=128):
    self_attn.q_proj  (1024 -> 4096)   — 8 heads * 512? Actually 8*128=1024... wait
    Actually: num_heads=8, head_dim=hidden_size*4/num_heads? No, q_proj out=4096 with 8 heads
    means 512 per head. This is likely using multi-head latent attention or similar.
    self_attn.k_proj  (1024 -> 512)    — 2 KV heads * 256
    self_attn.v_proj  (1024 -> 512)    — 2 KV heads * 256
    self_attn.o_proj  (2048 -> 1024)   — projects from attention output dim

Qwen3.5-4B (teacher): 32 layers (24 DeltaNet + 8 Full Attention)
"""

# ============================================================================
# Layer indices for 0.8B student
# Pattern: 6 groups of (3 DeltaNet + 1 Full Attn)
# ============================================================================
DELTANET_LAYER_INDICES = [
    i for i in range(24) if i % 4 != 3
]  # [0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22]

FULL_ATTN_LAYER_INDICES = [
    i for i in range(24) if i % 4 == 3
]  # [3, 7, 11, 15, 19, 23]

ALL_LAYER_INDICES = list(range(24))

STUDENT_NUM_LAYERS = 24
TEACHER_NUM_LAYERS = 32

# ============================================================================
# Hidden dimensions for Qwen3.5-0.8B (student) — from inspect_models.py
# ============================================================================
STUDENT_HIDDEN_SIZE = 1024
STUDENT_INTERMEDIATE_SIZE = 3584
STUDENT_NUM_HEADS = 8
STUDENT_NUM_KV_HEADS = 2
STUDENT_HEAD_DIM = 128  # inferred from o_proj input (2048 / num_heads=8 = 256? TBD)

# ============================================================================
# Projection dimensions (student) — from inspect_models.py
# ============================================================================

# Full Attention projections (6 layers: 3,7,11,15,19,23)
FA_Q_PROJ = (1024, 4096)   # self_attn.q_proj
FA_K_PROJ = (1024, 512)    # self_attn.k_proj
FA_V_PROJ = (1024, 512)    # self_attn.v_proj
FA_O_PROJ = (2048, 1024)   # self_attn.o_proj

# DeltaNet projections (18 layers) — fused, different module names
DN_IN_PROJ_QKV = (1024, 6144)   # linear_attn.in_proj_qkv
DN_OUT_PROJ = (2048, 1024)      # linear_attn.out_proj
DN_IN_PROJ_Z = (1024, 2048)     # linear_attn.in_proj_z
DN_IN_PROJ_A = (1024, 16)       # linear_attn.in_proj_a
DN_IN_PROJ_B = (1024, 16)       # linear_attn.in_proj_b

# MLP projections (same across all 24 layers)
GATE_PROJ = (1024, 3584)   # mlp.gate_proj
UP_PROJ = (1024, 3584)     # mlp.up_proj
DOWN_PROJ = (3584, 1024)   # mlp.down_proj

# ============================================================================
# Teacher dimensions (Qwen3.5-4B)
# ============================================================================
TEACHER_HIDDEN_SIZE = 2560
TEACHER_INTERMEDIATE_SIZE = 9216

# ============================================================================
# Virtual module name mapping
# ============================================================================
# Maps virtual names to (real_module_path, layer_indices, d_in, d_out).
#
# IMPORTANT: DeltaNet layers do NOT have self_attn.{q,k,v,o}_proj.
# Attention LoRA targets only apply to Full Attention layers (6 out of 24).
# MLP modules exist on all layers with the same dimensions.
# DeltaNet-specific modules (linear_attn.*) can be targeted separately.

VIRTUAL_MODULE_SPECS = {
    # --- Full Attention projections (FA layers only) ---
    "q_proj": {
        "real_name": "q_proj",
        "layer_indices": FULL_ATTN_LAYER_INDICES,
        "d_in": FA_Q_PROJ[0],
        "d_out": FA_Q_PROJ[1],
        "path_prefix": "self_attn",
    },
    "k_proj": {
        "real_name": "k_proj",
        "layer_indices": FULL_ATTN_LAYER_INDICES,
        "d_in": FA_K_PROJ[0],
        "d_out": FA_K_PROJ[1],
        "path_prefix": "self_attn",
    },
    "v_proj": {
        "real_name": "v_proj",
        "layer_indices": FULL_ATTN_LAYER_INDICES,
        "d_in": FA_V_PROJ[0],
        "d_out": FA_V_PROJ[1],
        "path_prefix": "self_attn",
    },
    "o_proj": {
        "real_name": "o_proj",
        "layer_indices": FULL_ATTN_LAYER_INDICES,
        "d_in": FA_O_PROJ[0],
        "d_out": FA_O_PROJ[1],
        "path_prefix": "self_attn",
    },
    # --- DeltaNet attention projections (DN layers only) ---
    "dn_out_proj": {
        "real_name": "out_proj",
        "layer_indices": DELTANET_LAYER_INDICES,
        "d_in": DN_OUT_PROJ[0],
        "d_out": DN_OUT_PROJ[1],
        "path_prefix": "linear_attn",
    },
    # --- MLP projections (all layers) ---
    "gate_proj": {
        "real_name": "gate_proj",
        "layer_indices": ALL_LAYER_INDICES,
        "d_in": GATE_PROJ[0],
        "d_out": GATE_PROJ[1],
        "path_prefix": "mlp",
    },
    "up_proj": {
        "real_name": "up_proj",
        "layer_indices": ALL_LAYER_INDICES,
        "d_in": UP_PROJ[0],
        "d_out": UP_PROJ[1],
        "path_prefix": "mlp",
    },
    "down_proj": {
        "real_name": "down_proj",
        "layer_indices": ALL_LAYER_INDICES,
        "d_in": DOWN_PROJ[0],
        "d_out": DOWN_PROJ[1],
        "path_prefix": "mlp",
    },
}


def get_module_specs_for_targets(target_modules: list[str]) -> dict:
    """Get the virtual module specs for the given target modules.

    Matches by real_name. For example, target "down_proj" matches the
    "down_proj" virtual spec. Target "out_proj" matches "dn_out_proj".
    """
    specs = {}
    for vname, vspec in VIRTUAL_MODULE_SPECS.items():
        base_name = vspec["real_name"]
        if base_name in target_modules:
            specs[vname] = vspec
    return specs


def get_basis_module_specs(target_modules: list[str]) -> dict:
    """Get module specs for basis LoRA bank.

    Returns dict mapping virtual_name -> (n_layers, d_in, d_out)
    for all modules targeted by basis LoRAs.
    """
    specs = {}
    for vname, vspec in VIRTUAL_MODULE_SPECS.items():
        if vspec["real_name"] in target_modules:
            n_layers = len(vspec["layer_indices"])
            specs[vname] = (n_layers, vspec["d_in"], vspec["d_out"])
    return specs
