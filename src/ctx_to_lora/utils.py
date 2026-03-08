import ast
import gc
import hashlib
import logging
import os
import random
import string
import time
from collections.abc import Iterable
from contextlib import contextmanager
from enum import Enum

import torch
import yaml
from peft import PeftConfig, PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from peft.utils import get_peft_model_state_dict

TRAINING_TASK = Enum("TRAINING_TASK", ["CAUSAL_LM", "COMPLETION"])


logger = logging.getLogger()


# taken from https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
@contextmanager
def evaluating(*models):
    """Temporarily switch to evaluation mode."""
    is_training = [model.training if model is not None else False for model in models]
    try:
        for model in models:
            if model is not None:
                model.eval()
        yield models
    finally:
        for model, training in zip(models, is_training):
            if model is not None:
                model.train(training)


def get_layers(model):
    if hasattr(model, "model"):
        return get_layers(model.model)
    return model.layers


def get_num_layers(model):
    return len(get_layers(model))


def get_base_model(model):
    if hasattr(model, "model"):
        return get_base_model(model.model)
    return model


def get_num_params(model):
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    return total_params, trainable_params


def log_num_train_params(model):
    logger.debug("Trainable model parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.debug(f"{name}, dtype:{p.dtype}")

    num_total_params, num_trainable_params = get_num_params(model)
    logger.info(
        f"trainable params: {num_trainable_params:,d} "
        f"|| all params: {num_total_params:,d} "
        f"|| trainable%: {100 * num_trainable_params / num_total_params:.4f}"
    )


def get_run_name(seed_str: str | None = None):
    if not seed_str:
        uuid = "".join(
            [random.choice(string.ascii_letters + string.digits) for _ in range(8)]
        )
        run_name = time.strftime("%Y%m%d-%H%M%S") + f"_{uuid}"
    else:
        # Generate a UUID from the seed string
        hash_object = hashlib.sha256(seed_str.encode())
        uuid = hash_object.hexdigest()[:8]  # Take the first 8 characters of the hash
        run_name = seed_str + f"_{uuid}"
    return run_name


def try_convert(s):
    try:
        return ast.literal_eval(s)
    except:
        return s


def extract_cli_args(argv: list[str]):
    out = dict()
    i = 0
    while i < len(argv):
        elem = argv[i]
        if elem.endswith(".yaml"):
            out["config"] = elem
        elif elem.startswith("--"):
            if "=" in elem:
                k, v = elem.split("=", 1)
                k = k.lstrip("-")
                out[k] = try_convert(v)
            else:
                k = elem.lstrip("-")
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    out[k] = try_convert(argv[i + 1])
                    i += 1
                else:
                    out[k] = True
        i += 1
    return out


def setup_logging(output_dir, debug=False):
    global logger

    os.makedirs(output_dir, exist_ok=True)

    log_formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_level = logging.DEBUG if debug else logging.INFO
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(stream_level)
    logger.addHandler(stream_handler)

    log_path = f"{output_dir}/debug.log"
    debug_handler = logging.FileHandler(log_path, delay=True)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(log_formatter)
    logger.addHandler(debug_handler)
    logger.setLevel(logging.DEBUG)
    logger.info(f"Logging to: {log_path}")


def validate_args(args_list):
    # there shouldn't be overlap between args
    keys = set()
    for args in args_list:
        logger.debug(args)
        args_keys = set(vars(args).keys())
        assert len(keys & args_keys) == 0, "Overlap between args"
        keys |= args_keys


def save_yaml(data, path):
    # Filter out non-primitive fields
    data = {
        k: v
        for k, v in data.items()
        if isinstance(v, (int, float, str, bool, list, dict, type(None)))
    }

    with open(path, "w") as file:
        yaml.dump(data, file)


def get_peft_modules(model: PeftModel, peft_config: PeftConfig) -> list[dict[str, str]]:
    return [
        {"name": name, "module": module}
        for name, module in model.named_modules()
        if name.split(".")[-1] in peft_config.target_modules
        and isinstance(module, BaseTunerLayer)
        and check_target_module_exists(peft_config, name)
    ]


def get_peft_in_out_features(
    model: PeftModel,
    peft_config: PeftConfig | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    if peft_config is None:
        return None, None
    in_features = dict()
    out_features = dict()
    for module_info in get_peft_modules(model, peft_config):
        module_name = module_info["name"]
        module = module_info["module"]
        # support just Linear layer for now
        # all modules should be a leave module that is Linear layer
        assert isinstance(module.base_layer, torch.nn.Linear), (
            "all modules should be a leave module that is Linear layer"
        )

        # this should always pass
        name = module_name.split(".")[-1]
        assert name in peft_config.target_modules

        if name not in in_features:
            in_features[name] = module.in_features
            out_features[name] = module.out_features
        else:
            # assumes each module has the same input and output features
            assert in_features[name] == module.in_features
            assert out_features[name] == module.out_features

    return in_features, out_features


def generated_lora_to_state_dict(
    lora_dict: dict,
    module_names: dict,
    target_modules: list[str],
    layer_indices: Iterable[int],
) -> dict:
    lora_state_dict = dict()
    for target_module in target_modules:
        for layer_idx in layer_indices:
            for module_name in module_names[target_module][layer_idx]:
                if "lora_A" in module_name:
                    lora_state_dict[module_name] = (
                        lora_dict[target_module]["A"][layer_idx].cpu().contiguous()
                    )
                elif "lora_B" in module_name:
                    lora_state_dict[module_name] = (
                        lora_dict[target_module]["B"][layer_idx].cpu().contiguous()
                    )
                else:
                    raise ValueError(f"Unexpected module name: {module_name}")
    return lora_state_dict


def get_lora_module_names(
    model: PeftModel,
    target_modules: list[str],
    layer_indices: Iterable[int],
) -> dict[str, list[str]]:
    module_names = {
        target_module: [[] for _ in range(len(layer_indices))]
        for target_module in target_modules
    }
    for k in get_peft_model_state_dict(model):
        if "lora" not in k:
            continue
        layer_idx = int(k.split("layers.")[-1].split(".")[0])
        if layer_idx in layer_indices:
            for target_module in target_modules:
                if target_module in k:
                    module_names[target_module][layer_idx].append(k)
                    break
    return module_names


def get_projection_specs(
    model,
    target_modules: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Directly inspect model to get per-module feature sizes.

    Unlike get_peft_in_out_features, this works on plain (non-PEFT) models
    and does NOT assert uniform dimensions across layers.

    Returns:
        (in_features, out_features) dicts keyed by module short name.
        For modules with varying sizes across layers, returns the most common size.
    """
    from collections import Counter
    in_features = {}
    out_features = {}

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        short_name = name.split(".")[-1]
        if short_name not in target_modules:
            continue

        if short_name not in in_features:
            in_features[short_name] = Counter()
            out_features[short_name] = Counter()

        in_features[short_name][module.in_features] += 1
        out_features[short_name][module.out_features] += 1

    # Return the most common size for each module
    result_in = {k: v.most_common(1)[0][0] for k, v in in_features.items()}
    result_out = {k: v.most_common(1)[0][0] for k, v in out_features.items()}
    return result_in, result_out


def get_per_layer_projection_specs(
    model,
    target_modules: list[str],
) -> dict[str, list[tuple[int, int]]]:
    """Get per-layer (in_features, out_features) for each target module.

    Returns dict mapping module_name -> list of (in_features, out_features) per layer.
    """
    from ctx_to_lora.utils import get_layers
    layers = get_layers(model)
    specs = {m: [] for m in target_modules}

    for layer in layers:
        for name, module in layer.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            short_name = name.split(".")[-1]
            if short_name in target_modules:
                specs[short_name].append((module.in_features, module.out_features))

    return specs


def compile_linear(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.compile()


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()


def concat_list(l):
    out = []
    for x in l:
        out += x
    return out


def check_is_iterable(x):
    try:
        iter(x)
    except TypeError:
        return False
    return True
