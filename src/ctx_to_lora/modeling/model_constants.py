"""
Model architecture constants for Qwen3.5 DeltaNet/Full Attention models.

These constants are derived from model inspection (scripts/inspect_models.py)
and must be updated if model architecture changes.

Qwen3.5-0.8B (student): 24 layers
  - 18 DeltaNet layers (groups of 3)
  - 6 Full Attention layers (every 4th layer)

Qwen3.5-4B (teacher): 36 layers
"""

# Layer indices for 0.8B student
# Pattern: 6 groups of (3 DeltaNet + 1 Full Attn)
# DeltaNet: 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22
# Full Attn: 3, 7, 11, 15, 19, 23
DELTANET_LAYER_INDICES = [
    i for i in range(24)
    if i % 4 != 3
]  # [0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22]

FULL_ATTN_LAYER_INDICES = [
    i for i in range(24)
    if i % 4 == 3
]  # [3, 7, 11, 15, 19, 23]

ALL_LAYER_INDICES = list(range(24))

STUDENT_NUM_LAYERS = 24
TEACHER_NUM_LAYERS = 36

# Hidden dimensions for Qwen3.5-0.8B (student)
STUDENT_HIDDEN_SIZE = 1024
STUDENT_INTERMEDIATE_SIZE = 4608  # MLP intermediate
STUDENT_NUM_HEADS = 16
STUDENT_NUM_KV_HEADS = 4
STUDENT_HEAD_DIM = STUDENT_HIDDEN_SIZE // STUDENT_NUM_HEADS  # 64

# Projection dimensions for student
# These may differ between DeltaNet and Full Attention layers
# DeltaNet layers may have different k_proj and v_proj dimensions

# Full Attention projections (standard GQA)
FA_Q_PROJ = (STUDENT_HIDDEN_SIZE, STUDENT_HIDDEN_SIZE)  # (1024, 1024)
FA_K_PROJ = (STUDENT_HIDDEN_SIZE, STUDENT_NUM_KV_HEADS * STUDENT_HEAD_DIM)  # (1024, 256)
FA_V_PROJ = (STUDENT_HIDDEN_SIZE, STUDENT_NUM_KV_HEADS * STUDENT_HEAD_DIM)  # (1024, 256)
FA_O_PROJ = (STUDENT_HIDDEN_SIZE, STUDENT_HIDDEN_SIZE)  # (1024, 1024)

# DeltaNet projections - assumed same as FA for q/o, may differ for k/v
# Update after running inspect_models.py
DN_Q_PROJ = (STUDENT_HIDDEN_SIZE, STUDENT_HIDDEN_SIZE)  # (1024, 1024)
DN_K_PROJ = (STUDENT_HIDDEN_SIZE, STUDENT_HIDDEN_SIZE)  # DeltaNet may use full-size keys
DN_V_PROJ = (STUDENT_HIDDEN_SIZE, STUDENT_HIDDEN_SIZE)  # DeltaNet may use full-size values
DN_O_PROJ = (STUDENT_HIDDEN_SIZE, STUDENT_HIDDEN_SIZE)  # (1024, 1024)

# MLP projections (same across all layers)
GATE_PROJ = (STUDENT_HIDDEN_SIZE, STUDENT_INTERMEDIATE_SIZE)  # (1024, 4608)
UP_PROJ = (STUDENT_HIDDEN_SIZE, STUDENT_INTERMEDIATE_SIZE)  # (1024, 4608)
DOWN_PROJ = (STUDENT_INTERMEDIATE_SIZE, STUDENT_HIDDEN_SIZE)  # (4608, 1024)

# Teacher dimensions (Qwen3.5-4B)
TEACHER_HIDDEN_SIZE = 2560
TEACHER_INTERMEDIATE_SIZE = 9216

# Virtual module name mapping
# Maps virtual names to (real_projection_name, layer_indices, d_in, d_out)
# Virtual names distinguish between DeltaNet and FullAttn when dims differ.
# This must be updated after inspect_models.py confirms actual dimensions.

VIRTUAL_MODULE_SPECS = {
    # Attention projections
    "q_proj": {
        "real_name": "q_proj",
        "layer_indices": ALL_LAYER_INDICES,
        "d_in": DN_Q_PROJ[0],
        "d_out": DN_Q_PROJ[1],
        "path_prefix": "self_attn",
    },
    "k_proj_dn": {
        "real_name": "k_proj",
        "layer_indices": DELTANET_LAYER_INDICES,
        "d_in": DN_K_PROJ[0],
        "d_out": DN_K_PROJ[1],
        "path_prefix": "self_attn",
    },
    "k_proj_fa": {
        "real_name": "k_proj",
        "layer_indices": FULL_ATTN_LAYER_INDICES,
        "d_in": FA_K_PROJ[0],
        "d_out": FA_K_PROJ[1],
        "path_prefix": "self_attn",
    },
    "v_proj_dn": {
        "real_name": "v_proj",
        "layer_indices": DELTANET_LAYER_INDICES,
        "d_in": DN_V_PROJ[0],
        "d_out": DN_V_PROJ[1],
        "path_prefix": "self_attn",
    },
    "v_proj_fa": {
        "real_name": "v_proj",
        "layer_indices": FULL_ATTN_LAYER_INDICES,
        "d_in": FA_V_PROJ[0],
        "d_out": FA_V_PROJ[1],
        "path_prefix": "self_attn",
    },
    "o_proj": {
        "real_name": "o_proj",
        "layer_indices": ALL_LAYER_INDICES,
        "d_in": DN_O_PROJ[1],  # Note: o_proj input is hidden_size (after attention)
        "d_out": DN_O_PROJ[0],
        "path_prefix": "self_attn",
    },
    # MLP projections
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

    For modules where DeltaNet and FullAttn have different dimensions,
    returns separate virtual entries. For uniform modules, returns one entry.
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
