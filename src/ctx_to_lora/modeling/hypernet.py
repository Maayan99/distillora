import logging
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from math import sqrt
from typing import Any

import torch
from einops import unpack
from einops.layers.torch import EinMix as Mix
from jaxtyping import Float, Integer
from peft import (
    LoraConfig,
    LoraRuntimeConfig,
    PeftConfig,
    PeftModel,
)
from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from peft.utils import PeftType, TaskType
from torch import Tensor, nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput
from transformers.models.modernbert.modeling_modernbert import ModernBertModel

from ctx_to_lora.configs import (
    AggregatorArguments,
    BasisLoRAArguments,
    CtxEncoderArguments,
    HypernetArguments,
    TeacherArguments,
)
from ctx_to_lora.data.processing import tokenize_ctx_text
from ctx_to_lora.model_loading import (
    get_model,
    get_tokenizer,
)
from ctx_to_lora.modeling.aggregator import (
    AGGREGATOR_CLS,
    AggregatorConfig,
    get_aggregator_config,
)
from ctx_to_lora.modeling.ctx_encoder import CTX_ENCODER_CLS, CTX_ENCODER_TYPE
from ctx_to_lora.modeling.lora_layer import (
    apply_lora_to_layers,
    apply_dual_lora_to_layers,
    apply_per_layer_lora,
    lora_forward,
    lora_forward_packed,
)
from ctx_to_lora.modeling.lora_merger import combine_lora
from ctx_to_lora.utils import (
    get_layers,
    get_num_layers,
    get_peft_in_out_features,
    get_peft_modules,
)

logger = logging.getLogger()


@dataclass
class HypernetConfig:
    latent_size: int
    use_light_weight_lora: bool
    light_weight_latent_size: int
    per_rank_gen: bool
    use_per_rank_bias: bool
    use_bias: bool
    per_layer_processing: bool
    use_token_mixing: bool
    num_pre_head_layers: int
    dropout_rate: float

    lora_config: LoraConfig
    extra_modules: list[str] | None
    base_hidden_size: int

    layer_indices: Iterable[int]
    feature_sizes: tuple[dict[str, int], dict[str, int]]
    aggregator_config: AggregatorConfig


def get_hypernet_config(
    model: PreTrainedModel,
    ctx_encoder_model_config: PretrainedConfig,
    hypernet_args: HypernetArguments,
    aggregator_args: AggregatorArguments,
    ctx_encoder_args: CtxEncoderArguments,
):
    num_modules = 0
    lora_config = getattr(model, "peft_config", None)
    if lora_config is not None:
        lora_config = lora_config["default"]
        num_modules += len(lora_config.target_modules)
    num_extra_modules = len(hypernet_args.extra_modules or [])
    indices = torch.arange(get_num_layers(model), device=model.device)
    return HypernetConfig(
        **vars(hypernet_args),
        base_hidden_size=model.config.hidden_size,
        lora_config=lora_config,
        layer_indices=indices,
        feature_sizes=get_peft_in_out_features(model, peft_config=lora_config),
        aggregator_config=get_aggregator_config(
            model,
            ctx_encoder_model_config,
            ctx_encoder_args.ctx_encoder_type == CTX_ENCODER_TYPE.PER_LAYER_ACTIVATIONS,
            hypernet_args.latent_size,
            num_modules,
            num_extra_modules,
            lora_config.r,
            hypernet_args.per_rank_gen,
            aggregator_args,
        ),
    )


def get_init_peft_weights(model: PeftModel, peft_config: PeftConfig = None):
    if peft_config is None:
        peft_config = model.peft_config["default"]
    peft_weights = {module_name: dict() for module_name in peft_config.target_modules}
    adapter_name = "default"
    for module_name, module in model.named_modules():
        if not check_target_module_exists(peft_config, module_name):
            continue
        if not isinstance(module, BaseTunerLayer):
            continue
        # support just Linear layer for now
        # all modules should be a leave module that is Linear layer
        assert isinstance(module.base_layer, nn.Linear), (
            "all modules should be a leave module that is Linear layer"
        )

        # this should always pass
        name = module_name.split(".")[-1]
        assert name in peft_config.target_modules

        for submodule_name, submodule in module.named_modules():
            if not isinstance(submodule, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue

            if adapter_name not in submodule:
                continue

            if submodule_name not in peft_weights[name]:
                peft_weights[name][submodule_name] = submodule[adapter_name]
            else:
                smod1 = peft_weights[name][submodule_name]
                smod2 = submodule[adapter_name]
                assert type(smod1) == type(smod2)

    return peft_weights


class ResMLPBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_rate: float = 0,
    ):
        super().__init__()
        layers = []
        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.mlp(x)


class ResMLPBlockPerLayer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()
        layers = [
            nn.LayerNorm(input_size),
            Mix(
                "bs n_layers n_modules r d_in -> bs n_layers n_modules r d_hid",
                weight_shape="n_layers d_in d_hid",
                bias_shape="n_layers d_hid",
                n_layers=n_layers,
                d_in=input_size,
                d_hid=hidden_size,
            ),
            nn.SiLU(),
            Mix(
                "bs n_layers n_modules r d_hid -> bs n_layers n_modules r d_out",
                weight_shape="n_layers d_hid d_out",
                bias_shape="n_layers d_out",
                n_layers=n_layers,
                d_hid=hidden_size,
                d_out=output_size,
            ),
            nn.LayerNorm(output_size),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)


class HyperLoRA(nn.Module):
    def __init__(self, config: HypernetConfig):
        super().__init__()

        # aggregator output [bs, n_layers, n_modules, feature_dim]
        # by mixing the pooled features with layer embs and module embs (for pooling)
        # or via a perceiver w/ bottleneck size = n_modules * n_layers
        self.config = config
        logger.debug(f"HyperLoRA config: {self.config}")
        self.iterative_mode = False
        self._init_model()

    def _init_model(self):
        self.agg_config = self.config.aggregator_config
        self.aggregator = AGGREGATOR_CLS[self.agg_config.aggregator_type](
            **vars(self.agg_config)
        )

        self.lora_config = self.config.lora_config
        self.r = self.lora_config.r

        self.target_modules = (
            tuple(sorted(self.lora_config.target_modules)) if self.lora_config else None
        )
        self.num_modules = len(self.target_modules) if self.target_modules else 0
        self.extra_modules = (
            self.config.extra_modules if self.config.extra_modules else None
        )
        self.num_extra_modules = len(self.extra_modules) if self.extra_modules else 0
        self.layer_indices = self.config.layer_indices
        self.n_layers = len(self.layer_indices)

        self.d_in, self.d_out = self.config.feature_sizes
        self.d_latent = self.config.latent_size

        if self.target_modules:
            if self.config.per_layer_processing:
                layers = [
                    ResMLPBlockPerLayer(
                        self.n_layers,
                        self.d_latent,
                        self.d_latent * 4,
                        self.d_latent,
                    )
                    for _ in range(self.config.num_pre_head_layers)
                ]
            else:
                layers = [
                    ResMLPBlock(
                        input_size=self.config.latent_size,
                        hidden_size=self.config.latent_size * 4,
                        output_size=self.config.latent_size,
                        dropout_rate=getattr(self.config, "dropout_rate", 0),
                    )
                    for _ in range(self.config.num_pre_head_layers)
                ]

            self.layers = nn.Sequential(*layers)

            self.d_lora = max(self.d_in[m] + self.d_out[m] for m in self.target_modules)

            self.bias_A = nn.ParameterDict(
                {
                    m: nn.Parameter(
                        torch.normal(
                            0,
                            0.2 / (self.d_in[m] * self.r) ** 0.5,
                            (self.n_layers, self.r, self.d_in[m]),
                        )
                    )
                    for m in self.target_modules
                }
            )
            self.bias_B = nn.ParameterDict(
                {
                    m: nn.Parameter(torch.zeros((self.n_layers, self.r, self.d_out[m])))
                    for m in self.target_modules
                }
            )

            self.scaler_A = nn.ParameterDict(
                {
                    m: nn.Parameter(torch.ones((1, self.n_layers, self.r, 1)))
                    for m in self.target_modules
                }
            )
            self.scaler_B = nn.ParameterDict(
                {
                    m: nn.Parameter(torch.zeros((1, self.n_layers, self.r, 1)))
                    for m in self.target_modules
                }
            )

            n_modules = len(self.target_modules)
            # have to do this otherwise doesnt work with adamw_torch_fused
            # has something to do with the bias shape (n_modules r d_lora)
            # when n_modules == 1, adamw_torch_fused complains about device/layout
            # but when n_modules > 1, it works fine
            if n_modules == 1:
                self.head = Mix(
                    "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
                    weight_shape="n_layers d_latent d_lora",
                    bias_shape=None,  # no bias
                    n_layers=len(self.layer_indices),
                    d_latent=self.config.latent_size,
                    r=self.config.lora_config.r,
                    d_lora=self.d_lora,
                )
            else:
                self.head = Mix(
                    "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
                    weight_shape="n_layers n_modules d_latent d_lora",
                    bias_shape=None,  # no bias
                    n_layers=len(self.layer_indices),
                    n_modules=n_modules,
                    d_latent=self.config.latent_size,
                    r=self.config.lora_config.r,
                    d_lora=self.d_lora,
                )

    def get_head_bias(self):
        bias_dict = dict()
        for module in self.target_modules:
            bias_A = self.bias_A[module]
            bias_B = self.bias_B[module]

            bias_dict[module] = dict(A=bias_A, B=bias_B)
        return bias_dict

    def _to_lora_dict(
        self, flat_loras: Float[Tensor, "bs n_layers n_modules r max_io_dim"]
    ) -> dict[str, dict[str, Float[Tensor, "bs n_layers r _"]]]:
        if self.target_modules is None:
            return None
        # list of [bs, n_layers, r, in_d_outim]
        # and in_d_outim might vary across modules
        loras = unpack(
            flat_loras,
            [[] for _ in range(len(self.target_modules))],
            "bs n_layers * r max_io_dim",
        )

        # dict of {module:
        #   {A: [bs, n_layers, r, d_inim],
        #    B: [bs, n_layers, r, d_outim]}}
        lora_dict = dict()
        for module, lora in zip(self.target_modules, loras):
            A, B = unpack(
                lora[..., : self.d_in[module] + self.d_out[module]],
                [[self.d_in[module]], [self.d_out[module]]],
                "bs n_layers r *",
            )

            # apparently doing A * self.scaler_A is slow due to broadcasting
            A = torch.einsum("ijkl,ijkl->ijkl", A, self.scaler_A[module])
            B = torch.einsum("ijkl,ijkl->ijkl", B, self.scaler_B[module])

            lora_dict[module] = dict(A=A, B=B)

        return lora_dict

    def _to_layernorm_dict(
        self, flat_layernorms: Float[Tensor, "bs n_layers n_modules hidden_size"]
    ) -> dict[str, Float[Tensor, "bs n_layers hidden_size"]]:
        if self.extra_modules is None:
            return None
        layernorms = unpack(
            flat_layernorms,
            [[] for _ in range(len(self.extra_modules))],
            "bs n_layers * hidden_size",
        )
        return {k: v for k, v in zip(self.extra_modules, layernorms)}

    def enable_iterative_mode(self, x: bool):
        self.iterative_mode = x
        self.aggregator.enable_iterative_mode(x)

    def forward(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        position_ids: Integer[Tensor, "bs seq_len"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
    ):
        # [bs, n_layers, n_total_modules, r, feature_dim]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.aggregator.layer_to_layer and self.iterative_mode:
                # iterative inference
                # features: [bs num_layers seq_len feature_dim]
                bs, n_layers = features.shape[0:2]
                lora_emb = torch.empty(
                    (bs, n_layers, self.num_modules, self.r, self.config.latent_size),
                    device=features.device,
                )
                for i in range(n_layers):
                    lora_emb[:, i], _ = self.aggregator(
                        features[:, i], attn_mask, position_ids
                    )

            else:
                # batched inference
                lora_emb, _ = self.aggregator(features, attn_mask, position_ids)

        # [bs, n_layers, n_modules, r, max_in_d_outim]
        flat_loras = None
        if self.target_modules:
            lora_emb = self.layers(lora_emb)
            norm = torch.norm(lora_emb, dim=-1, keepdim=True)
            norm_lora_emb = lora_emb / norm
            flat_loras = self.head(norm_lora_emb)

        flat_layernorms = None

        return flat_loras, flat_layernorms

    def generate_weights(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        position_ids: Integer[Tensor, "bs seq_len"] | None = None,
    ):
        flat_loras, flat_layernorms = self.forward(features, attn_mask, position_ids)
        return self._to_lora_dict(flat_loras), self._to_layernorm_dict(flat_layernorms)


class ModulatedPretrainedModel(nn.Module):
    def __init__(
        self,
        base_model: PeftModel,
        hypernet_config: HypernetConfig,
        ctx_encoder_args: CtxEncoderArguments,
        use_base_input_as_ctx: bool = False,
        # need non-packed inputs for generation
        use_sequence_packing: bool = True,
        user_defined_scaling: float = 1,
        inp_compressor=None,
    ):
        assert not use_base_input_as_ctx
        super().__init__()
        self.device = base_model.device
        self.peft_config = base_model.peft_config["default"]
        self.hypernet_config = hypernet_config
        self.ctx_encoder_args = ctx_encoder_args
        self.use_base_input_as_ctx = use_base_input_as_ctx
        self.use_sequence_packing = use_sequence_packing
        self.user_defined_scaling = user_defined_scaling
        self.inp_compressor = inp_compressor
        self.model_accepts_loss_kwargs = True
        self.generated_loras = None

        self.register_module("base_model", base_model)
        self._init_model()
        self._bias_hyper_init()

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        train: bool = True,
        base_model_kwargs: dict = None,
        use_flash_attn: bool = True,
        **kwargs: Any,
    ):
        lora_config = state_dict["hypernet_config"].lora_config
        print(f"lora_config: {lora_config}")
        model_name_or_path = state_dict["base_model_name_or_path"]
        base_model = get_model(
            model_name_or_path,
            train=train,
            requires_grad=False,
            peft_config=lora_config,
            model_kwargs=base_model_kwargs,
            use_flash_attn=use_flash_attn,
        )
        hypernet_config = state_dict["hypernet_config"]
        if getattr(hypernet_config, "num_pre_head_layers", None) is None:
            hypernet_config.num_pre_head_layers = 4
        if getattr(hypernet_config, "use_per_rank_bias", None) is None:
            hypernet_config.use_per_rank_bias = False
        if getattr(hypernet_config, "use_bias", None) is None:
            hypernet_config.use_bias = True
        ctx_encoder_args = state_dict["ctx_encoder_args"]
        model = cls(base_model, hypernet_config, ctx_encoder_args, **kwargs)
        model.load_state_dict(state_dict)
        return model

    def patch_lora_forward(self):
        layers = get_layers(self.base_model)

        lora_forward_fn = (
            lora_forward_packed if self.use_sequence_packing else lora_forward
        )
        for layer_idx in self.hypernet.layer_indices:
            for module_info in get_peft_modules(layers[layer_idx], self.peft_config):
                name = module_info["name"]
                module = module_info["module"]
                if getattr(module, "patched_forward", False):
                    continue
                logger.debug(f"Applying LoRA forward to {name}")
                module.forward_orig = module.forward
                module.patched_forward = True
                module.forward = partial(
                    lora_forward_fn,
                    self=module,
                    lora_dropout_p=self.peft_config.lora_dropout,
                    scaling=self.peft_config.lora_alpha,
                )

    def _init_model(self):
        # disable adapter of the base model
        # this only works with LoRA(?)
        # we disable to avoid peft lora computation
        self.base_model.disable_adapter_layers()

        self.hypernet = (
            HyperLoRA(self.hypernet_config).to(self.device).to(torch.float32)
        )

        self.patch_lora_forward()

        ctx_model_name = self.ctx_encoder_args.ctx_encoder_model_name_or_path
        if ctx_model_name is None:
            ctx_model_name = self.base_model.config.name_or_path
        # use an explicit copy of the base model
        # for using with "modules_to_save"
        base_model_attn_impl = self.base_model.config._attn_implementation
        logger.debug(f"ctx_model_name: {ctx_model_name}")
        logger.debug(f"base_model.config._attn_implementation: {base_model_attn_impl}")
        encoder_model = get_model(
            ctx_model_name,
            train=self.base_model.training,
            requires_grad=False,
            use_flash_attn=base_model_attn_impl == "flash_attention_2",
            use_q_lora=self.ctx_encoder_args.quantize_ctx_encoder,
        )
        self.ctx_encoder = CTX_ENCODER_CLS[self.ctx_encoder_args.ctx_encoder_type](
            encoder_model, self.ctx_encoder_args
        )

    # delegate to base_model
    @property
    def config(self):
        return self.base_model.config

    @property
    def generation_config(self):
        return self.base_model.generation_config

    @property
    def vocab_size(self):
        return self.base_model.vocab_size

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    @torch.no_grad()
    def _bias_hyper_init(self):
        if self.hypernet.extra_modules:
            self.hypernet.extra_head.weight.data[:] = 0
            self.hypernet.extra_head.bias.data[:] = 0
        if self.hypernet.target_modules:
            peft_weights = get_init_peft_weights(
                self.base_model, self.hypernet.lora_config
            )
            logger.debug(f"peft_weights: {peft_weights}")
            r = self.hypernet_config.lora_config.r
            nn.init.normal_(
                self.hypernet.head.weight,
                mean=0,
                std=0.5
                / sqrt(self.hypernet.config.latent_size + self.hypernet.d_lora * r),
                # the head outputs per rank lora --> divide by r to scale down grad
            )

    def state_dict(self, *args, **kwargs):
        # we assume ctx_encoder and base model is frozen here
        if len([p for p in self.ctx_encoder.parameters() if p.requires_grad]):
            raise ValueError("ctx_encoder contains trainable parameters")
        if len([p for p in self.base_model.parameters() if p.requires_grad]):
            raise ValueError("base model contains trainable parameters")

        state_dict = self.hypernet.state_dict(*args, **kwargs)
        state_dict["base_model_name_or_path"] = self.base_model.name_or_path
        state_dict["hypernet_config"] = self.hypernet_config
        state_dict["ctx_encoder_args"] = self.ctx_encoder_args
        return state_dict

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self.base_model_name_or_path = state_dict.pop("base_model_name_or_path")
        self.hypernet_config = state_dict.pop("hypernet_config")
        self.ctx_encoder_args = state_dict.pop("ctx_encoder_args")
        if self.base_model_name_or_path != self.base_model.name_or_path:
            raise ValueError(
                f"Base model name or path mismatch. "
                f"The base model given is: {self.base_model.name_or_path}, "
                f"but the loaded name is: {self.base_model_name_or_path}"
            )
        self._init_model()

        def remove_compile_prefix(sd: dict[str, Tensor]) -> dict[str, Tensor]:
            COMPILED_PREFIX = "_orig_mod."
            for k in list(sd.keys()):
                if k.startswith(COMPILED_PREFIX):
                    sd[k[len(COMPILED_PREFIX) :]] = sd.pop(k)
            return sd

        load_result = self.hypernet.load_state_dict(
            remove_compile_prefix(state_dict),
            strict=True,  # , *args, **kwargs
        )
        logger.info(f"load result: {load_result}")
        return load_result

    def generate_weights(
        self,
        ctx_ids: Integer[Tensor, "bs ctx_len"],
        ctx_attn_mask: Integer[Tensor, "bs ctx_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "bs ctx_len"] | None = None,
        **kwargs: Any,
    ):
        with torch.no_grad():
            ctx_encoder_kwargs = dict(
                input_ids=ctx_ids,
                attention_mask=ctx_attn_mask,
                position_ids=ctx_position_ids,
            )
            if isinstance(self.ctx_encoder.base_model, ModernBertModel):
                position_ids = ctx_position_ids.flatten()
                indices = torch.arange(
                    position_ids.size(0), device=position_ids.device, dtype=torch.int32
                )
                # [bsz + 1]
                cu_seqlens = torch.cat(
                    (
                        indices[position_ids == 0],
                        torch.tensor(
                            position_ids.size(),
                            device=position_ids.device,
                            dtype=torch.int32,
                        ),
                    )
                )
                ctx_encoder_kwargs = dict(
                    input_ids=ctx_ids.squeeze(0),
                    cu_seqlens=cu_seqlens,
                    max_seqlen=position_ids.max() + 1,
                    attention_mask=-1,
                    seq_len=-1,
                    batch_size=-1,
                )

            ctx_features = self.ctx_encoder(**ctx_encoder_kwargs, **kwargs)

        if isinstance(self.ctx_encoder.base_model, ModernBertModel):
            ctx_features = ctx_features.unsqueeze(0)
        if self.user_defined_scaling == 1:
            return self.hypernet.generate_weights(
                ctx_features, ctx_attn_mask, ctx_position_ids
            )

        lora_dict, _ = self.hypernet.generate_weights(
            ctx_features, ctx_attn_mask, ctx_position_ids
        )
        for module in lora_dict:
            lora_dict[module]["A"] = lora_dict[module]["A"] * self.user_defined_scaling
            lora_dict[module]["B"] = lora_dict[module]["B"] * self.user_defined_scaling
        return lora_dict, None

    def enable_iterative_mode(self, x: bool):
        self.hypernet.enable_iterative_mode(x)

    def forward(
        self,
        ctx_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_attn_mask: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
        n_queries: Integer[Tensor, "n_ctx"] | None = None,
        return_generated_lora: bool | None = False,
        *model_inputs_args: Any,
        **model_inputs_kwargs: dict[str, Any],
    ) -> tuple | ModelOutput:
        """Forward pass of the modulated model."""
        generated_loras = None
        generated_layernorms = None
        if ctx_ids is None and not self.use_base_input_as_ctx:
            logger.warning(
                (
                    "*" * 100,
                    "\n\nNo ctx_features provided, using the base model for forward pass\n\n",
                    "*" * 100,
                )
            )

        else:
            if self.use_base_input_as_ctx:
                ctx_ids = (
                    model_inputs_kwargs["input_ids"]
                    if "input_ids" in model_inputs_kwargs
                    else model_inputs_args[0]
                )
                ctx_attn_mask = (
                    model_inputs_kwargs["attention_mask"]
                    if "attention_mask" in model_inputs_kwargs
                    else None
                )
                ctx_position_ids = (
                    model_inputs_kwargs["position_ids"]
                    if "position_ids" in model_inputs_kwargs
                    else None
                )
            generated_loras, generated_layernorms = self.generate_weights(
                ctx_ids, ctx_attn_mask, ctx_position_ids
            )

        if generated_loras is not None:
            generated_loras = combine_lora(
                generated_loras,
                n_ctx_chunks,
                lora_bias=self.hypernet.get_head_bias()
                if self.hypernet.config.use_bias
                else None,
            )

            # input_ids in model_inputs_kwargs contains only
            # prompt + response (for hypernet training)
            position_ids = (
                model_inputs_kwargs["position_ids"]
                if "position_ids" in model_inputs_kwargs
                else None
            )

            if n_queries is None:
                if ctx_position_ids is None:
                    n_queries = torch.ones(
                        ctx_ids.shape[0], dtype=torch.int32, device=self.device
                    )
                else:
                    # quite redundant (we do cu_seqlens many places)
                    # TODO: compute cu_seqlens here and propagate that
                    n_queries = torch.ones(
                        (ctx_position_ids == 0).sum(),
                        dtype=torch.int32,
                        device=self.device,
                    )

            apply_lora_to_layers(
                self.base_model,
                self.hypernet.layer_indices,
                generated_loras,
                n_queries,
                position_ids,
            )
        model_outputs = self.base_model(*model_inputs_args, **model_inputs_kwargs)

        if return_generated_lora:
            return model_outputs, (generated_loras, generated_layernorms)
        else:
            return model_outputs

    def combine_lora(self, *args, **kwargs):
        # for timing
        return combine_lora(*args, **kwargs)

    def apply_lora_to_layers(self, *args, **kwargs):
        # for timing
        return apply_lora_to_layers(*args, **kwargs)

    # for simple api usage
    def internalize(self, ctx_str: str):
        ctx_tokenizer = get_tokenizer(self.ctx_encoder.base_model.name_or_path)
        ctx_ids = tokenize_ctx_text(dict(context=[ctx_str]), ctx_tokenizer)["ctx_ids"]
        return self._internalize_from_ids(torch.tensor(ctx_ids, device=self.device))

    def _internalize_from_ids(
        self,
        ctx_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_attn_mask: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
    ):
        self.patch_lora_forward()
        if ctx_attn_mask is None and ctx_position_ids is None:
            assert ctx_ids.shape[0] == 1
            ctx_attn_mask = torch.ones_like(ctx_ids)
        generated_loras, generated_layernorms = self.generate_weights(
            ctx_ids, ctx_attn_mask, ctx_position_ids
        )
        self.generated_loras = generated_loras

    def reset(self):
        self.generated_loras = None
        layers = get_layers(self.base_model)
        for layer_idx in self.hypernet.layer_indices:
            for module_info in get_peft_modules(layers[layer_idx], self.peft_config):
                name = module_info["name"]
                module = module_info["module"]
                logger.debug(f"Resetting forward for {name}")
                module.forward = module.forward_orig
                module.patched_forward = False

    @torch.inference_mode()
    def generate(
        self,
        ctx_ids: Integer[Tensor, "n_chunks ctx_length"] | None = None,
        ctx_attn_mask: Integer[Tensor, "n_chunks ctx_length"] | None = None,
        ctx_position_ids: Integer[Tensor, "n_chunks ctx_length"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
        n_queries: Integer[Tensor, "n_ctx"] | None = None,
        scalers: Float[Tensor, "n_ctx"] | None = None,
        bias_scaler: float | None = None,
        *model_inputs_args: Any,
        **model_inputs_kwargs: dict[str, Any],
    ):
        generated_loras = None
        generated_layernorms = None
        if (
            ctx_ids is None
            and not self.generated_loras
            and not self.use_base_input_as_ctx
        ):
            print(
                "*" * 100
                + "\n\nNo ctx_ids provided, using the base model for generation\n\n"
                + "*" * 100
            )
        elif ctx_ids is None and self.generated_loras:
            generated_loras = self.generated_loras
            if n_ctx_chunks is None:
                n_ctx_chunks = torch.tensor((1,), device=self.device)
            print(
                "*" * 100
                + "\n\nUsing internalized LoRAs for generation\n\n"
                + "*" * 100
            )
        else:
            if self.use_base_input_as_ctx:
                ctx_ids = (
                    model_inputs_kwargs["input_ids"]
                    if "input_ids" in model_inputs_kwargs
                    else model_inputs_args[0]
                )
                ctx_attn_mask = (
                    model_inputs_kwargs["attention_mask"]
                    if "attention_mask" in model_inputs_kwargs
                    else None
                )
                ctx_position_ids = (
                    model_inputs_kwargs["position_ids"]
                    if "position_ids" in model_inputs_kwargs
                    else None
                )
            generated_loras, generated_layernorms = self.generate_weights(
                ctx_ids, ctx_attn_mask, ctx_position_ids
            )

        if generated_loras is not None:
            generated_loras = self.combine_lora(
                generated_loras,
                n_ctx_chunks,
                lora_bias=self.hypernet.get_head_bias()
                if self.hypernet.config.use_bias
                else None,
                scalers=scalers,
                bias_scaler=bias_scaler,
            )

            # apply lora hook to the base model
            # TODO: we dont this position_ids for generation?
            position_ids = (
                model_inputs_kwargs["position_ids"]
                if "position_ids" in model_inputs_kwargs
                else None
            )
            if n_queries is None:
                if ctx_position_ids is None:
                    n_queries = torch.ones(
                        model_inputs_kwargs["input_ids"].shape[0],
                        dtype=torch.int32,
                        device=self.device,
                    )
                else:
                    # quite redundant (we do cu_seqlens many places)
                    # TODO: compute cu_seqlens here and propagate that
                    n_queries = torch.ones(
                        (ctx_position_ids == 0).sum(),
                        dtype=torch.int32,
                        device=self.device,
                    )

            apply_lora_to_layers(
                self.base_model,
                self.hypernet.layer_indices,
                generated_loras,
                n_queries,
                position_ids,
            )

        model_outputs = self.base_model.generate(
            *model_inputs_args, **model_inputs_kwargs
        )
        return model_outputs


@dataclass
class HyperDistillConfig:
    """Configuration for HyperDistill model."""
    latent_size: int
    lora_rank: int

    # Basis LoRA
    n_basis: int
    basis_rank: int
    per_module_routing: bool
    n_refinement_blocks: int

    # Module specs: dict mapping virtual_name -> (n_layers, d_in, d_out)
    basis_module_specs: dict

    # Hyper-generated targets (subset of basis targets)
    hyper_target_modules: list[str]
    # All basis target modules (virtual names)
    basis_target_modules: list[str]

    # Student model info
    student_hidden_size: int
    student_num_layers: int

    # Aggregator config
    aggregator_config: AggregatorConfig

    # Encoder args
    ctx_encoder_type: str

    # LoRA scaling
    lora_alpha: float = None  # defaults to lora_rank in HyperDistillModel


class MultiHeadHyperLoRA(nn.Module):
    """Multi-head hypernetwork that generates LoRAs for heterogeneous target modules.

    Instead of a single EinMix head, uses multiple heads for different module groups,
    supporting different output dimensions per head.
    """

    def __init__(
        self,
        config: HyperDistillConfig,
    ):
        super().__init__()
        self.config = config
        self.d_latent = config.latent_size
        self.rank = config.lora_rank

        # Group hyper-generated targets by output dimension
        self.heads = nn.ModuleDict()
        self.head_assignments = {}  # virtual_name -> head_name

        # Compute d_lora for each target
        module_d_loras = {}
        for vname in config.hyper_target_modules:
            if vname in config.basis_module_specs:
                n_layers, d_in, d_out = config.basis_module_specs[vname]
                module_d_loras[vname] = d_in + d_out

        # Group modules by d_lora for shared heads
        from collections import defaultdict
        groups = defaultdict(list)
        for vname, d_lora in module_d_loras.items():
            groups[d_lora].append(vname)

        # Create per-group heads
        for d_lora, vnames in groups.items():
            head_name = f"head_{d_lora}"
            # Per-layer EinMix projection
            self.heads[head_name] = nn.Sequential(
                nn.LayerNorm(self.d_latent),
                nn.Linear(self.d_latent, d_lora),
            )
            nn.init.zeros_(self.heads[head_name][1].bias)

            for vname in vnames:
                self.head_assignments[vname] = head_name

        # Per-module scalers (learned)
        self.scaler_A = nn.ParameterDict()
        self.scaler_B = nn.ParameterDict()
        for vname in config.hyper_target_modules:
            if vname in config.basis_module_specs:
                n_layers = config.basis_module_specs[vname][0]
                self.scaler_A[vname] = nn.Parameter(
                    torch.ones(1, n_layers, self.rank, 1)
                )
                self.scaler_B[vname] = nn.Parameter(
                    torch.full((1, n_layers, self.rank, 1), 0.01)
                )

    def forward(
        self,
        latent_queries: dict[str, Float[Tensor, "batch n_layers rank d_latent"]],
    ) -> dict[str, dict[str, Float[Tensor, "batch n_layers rank d"]]]:
        """Generate LoRA A/B matrices from latent queries.

        Args:
            latent_queries: Dict mapping virtual_name -> (batch, n_layers, rank, d_latent)

        Returns:
            Dict mapping virtual_name -> {A: (batch, n_layers, rank, d_in),
                                           B: (batch, n_layers, rank, d_out)}
        """
        lora_dict = {}

        for vname, queries in latent_queries.items():
            if vname not in self.head_assignments:
                continue

            head_name = self.head_assignments[vname]
            head = self.heads[head_name]

            # queries: (batch, n_layers, rank, d_latent)
            # Normalize before head
            norm = torch.norm(queries, dim=-1, keepdim=True)
            norm_queries = queries / (norm + 1e-8)

            # Project: (batch, n_layers, rank, d_lora)
            flat_lora = head(norm_queries)

            # Split into A and B
            n_layers, d_in, d_out = self.config.basis_module_specs[vname]
            A = flat_lora[..., :d_in]
            B = flat_lora[..., d_in:d_in + d_out]

            # Apply per-module scalers
            A = torch.einsum("ijkl,ijkl->ijkl", A, self.scaler_A[vname])
            B = torch.einsum("ijkl,ijkl->ijkl", B, self.scaler_B[vname])

            lora_dict[vname] = {"A": A, "B": B}

        return lora_dict


class HyperDistillModel(nn.Module):
    """HyperDistill: Basis LoRA + Hypernetwork for cross-model distillation.

    Combines:
    1. Context encoder (frozen student model) for per-layer activations
    2. Perceiver aggregator → latent queries
    3. Coefficient head → basis mixing coefficients
    4. Basis LoRA bank → mixed basis LoRAs
    5. Refinement blocks → refined latent queries
    6. Multi-head LoRA generation → hyper-generated LoRAs
    7. Combined (basis + hyper) LoRA application to student
    8. Online teacher for distillation targets
    """

    def __init__(
        self,
        base_model: nn.Module,  # Student model (plain, no PEFT)
        config: HyperDistillConfig,
        ctx_encoder_args: CtxEncoderArguments,
        teacher: nn.Module | None = None,
        basis_lora_bank: nn.Module | None = None,
        use_sequence_packing: bool = True,
        lora_alpha: float = None,
    ):
        super().__init__()
        self.device = next(base_model.parameters()).device
        self.hyperdistill_config = config
        self.ctx_encoder_args = ctx_encoder_args
        self.use_sequence_packing = use_sequence_packing
        self.model_accepts_loss_kwargs = True
        self.lora_alpha = lora_alpha or config.lora_alpha or float(config.lora_rank)

        # Register base model (student, frozen)
        self.register_module("base_model", base_model)

        # Teacher (frozen, for online distillation)
        if teacher is not None:
            self.register_module("teacher", teacher)
        else:
            self.teacher = None

        # Basis LoRA bank
        if basis_lora_bank is not None:
            self.register_module("basis_bank", basis_lora_bank)
        else:
            self.basis_bank = None

        self._init_model()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Delegate gradient checkpointing to the base (student) model."""
        self.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()

    @property
    def vocab_size(self):
        return self.base_model.config.vocab_size

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    @property
    def generation_config(self):
        return self.base_model.generation_config

    def _init_model(self):
        """Initialize context encoder, perceiver, coefficient head, and hyper heads."""
        from ctx_to_lora.modeling.coefficient_head import (
            CoefficientHead,
            RefinementBlocks,
        )

        config = self.hyperdistill_config

        # Context encoder (frozen copy of student)
        ctx_model_name = self.ctx_encoder_args.ctx_encoder_model_name_or_path
        if ctx_model_name is None:
            ctx_model_name = self.base_model.config.name_or_path

        base_model_attn_impl = self.base_model.config._attn_implementation
        encoder_model = get_model(
            ctx_model_name,
            train=self.base_model.training,
            requires_grad=False,
            use_flash_attn=base_model_attn_impl == "flash_attention_2",
        )
        self.ctx_encoder = CTX_ENCODER_CLS[self.ctx_encoder_args.ctx_encoder_type](
            encoder_model, self.ctx_encoder_args
        )

        # Perceiver aggregator
        # Total output queries = sum over hyper targets of (n_layers * rank)
        n_output_queries = sum(
            config.basis_module_specs[v][0] * config.lora_rank
            for v in config.hyper_target_modules
            if v in config.basis_module_specs
        )

        from ctx_to_lora.modeling.idefics2 import Idefics2Perceiver, Idefics2PerceiverConfig

        feature_size = self.ctx_encoder.config.hidden_size
        encoder_config = Idefics2PerceiverConfig(
            input_size=feature_size,
            num_blocks=config.aggregator_config.num_blocks,
            num_self_attn_per_block=config.aggregator_config.num_self_attn_per_block,
            shared_weights=config.aggregator_config.shared_weights,
            n_latents=config.aggregator_config.n_latent_queries,
            intermediate_size_factor=4,
            hidden_size=config.latent_size,
            attn_implementation="sdpa",
        )
        decoder_config = Idefics2PerceiverConfig(
            input_size=config.latent_size,
            num_blocks=1,
            num_self_attn_per_block=0,
            shared_weights=False,
            n_latents=n_output_queries,
            intermediate_size_factor=4,
            hidden_size=config.latent_size,
            attn_implementation="sdpa",
        )
        self.perceiver = Idefics2Perceiver(encoder_config, decoder_config)

        # Coefficient head (for basis mixing)
        if self.basis_bank is not None:
            self.coefficient_head = CoefficientHead(
                d_latent=config.latent_size,
                n_basis=config.n_basis,
            )
            self.refinement_blocks = RefinementBlocks(
                d_latent=config.latent_size,
                n_basis=config.n_basis,
                n_blocks=config.n_refinement_blocks,
            )

        # Multi-head LoRA generation
        self.hyper_heads = MultiHeadHyperLoRA(config)

        # Patch base model with LoRA forwards
        self._patch_lora_forward()

    def _patch_lora_forward(self):
        """Monkey-patch student model layers with LoRA forward passes."""
        from ctx_to_lora.modeling.model_constants import VIRTUAL_MODULE_SPECS

        layers = get_layers(self.base_model)

        lora_forward_fn = (
            lora_forward_packed if self.use_sequence_packing else lora_forward
        )

        # Collect all real module names that need patching
        all_real_modules = set()
        for vname in list(self.hyperdistill_config.basis_module_specs.keys()):
            if vname in VIRTUAL_MODULE_SPECS:
                spec = VIRTUAL_MODULE_SPECS[vname]
                all_real_modules.add(
                    f"{spec['path_prefix']}.{spec['real_name']}"
                )

        for layer_idx in range(len(layers)):
            layer = layers[layer_idx]
            for name, module in layer.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                if name not in all_real_modules:
                    continue
                if getattr(module, "patched_forward", False):
                    continue

                module.forward_orig = module.forward
                module.patched_forward = True
                base_forward = partial(
                    lora_forward_fn,
                    self=module,
                    lora_dropout_p=0.0,
                    scaling=self.lora_alpha,
                )
                module._base_lora_forward = base_forward
                module.forward = base_forward

    @property
    def config(self):
        return self.base_model.config

    def state_dict(self, *args, **kwargs):
        """Save only trainable components (not frozen base model / teacher / ctx encoder)."""
        state = {}

        # Save perceiver
        for k, v in self.perceiver.state_dict(*args, **kwargs).items():
            state[f"perceiver.{k}"] = v

        # Save hyper heads
        for k, v in self.hyper_heads.state_dict(*args, **kwargs).items():
            state[f"hyper_heads.{k}"] = v

        # Save basis bank
        if self.basis_bank is not None:
            for k, v in self.basis_bank.state_dict(*args, **kwargs).items():
                state[f"basis_bank.{k}"] = v

        # Save coefficient head and refinement
        if hasattr(self, "coefficient_head"):
            for k, v in self.coefficient_head.state_dict(*args, **kwargs).items():
                state[f"coefficient_head.{k}"] = v
            for k, v in self.refinement_blocks.state_dict(*args, **kwargs).items():
                state[f"refinement_blocks.{k}"] = v

        # Save metadata
        state["config"] = self.hyperdistill_config
        state["ctx_encoder_args"] = self.ctx_encoder_args
        state["base_model_name_or_path"] = self.base_model.config.name_or_path

        return state

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        config = state_dict.pop("config")
        ctx_encoder_args = state_dict.pop("ctx_encoder_args")
        base_model_name = state_dict.pop("base_model_name_or_path")

        # Load component state dicts
        perceiver_sd = {k.replace("perceiver.", ""): v for k, v in state_dict.items() if k.startswith("perceiver.")}
        hyper_heads_sd = {k.replace("hyper_heads.", ""): v for k, v in state_dict.items() if k.startswith("hyper_heads.")}
        basis_sd = {k.replace("basis_bank.", ""): v for k, v in state_dict.items() if k.startswith("basis_bank.")}
        coeff_sd = {k.replace("coefficient_head.", ""): v for k, v in state_dict.items() if k.startswith("coefficient_head.")}
        refine_sd = {k.replace("refinement_blocks.", ""): v for k, v in state_dict.items() if k.startswith("refinement_blocks.")}

        self.perceiver.load_state_dict(perceiver_sd, strict=True)
        self.hyper_heads.load_state_dict(hyper_heads_sd, strict=True)
        if self.basis_bank is not None and basis_sd:
            self.basis_bank.load_state_dict(basis_sd, strict=True)
        if hasattr(self, "coefficient_head") and coeff_sd:
            self.coefficient_head.load_state_dict(coeff_sd, strict=True)
        if hasattr(self, "refinement_blocks") and refine_sd:
            self.refinement_blocks.load_state_dict(refine_sd, strict=True)

    def _encode_context(
        self,
        ctx_ids: Integer[Tensor, "n_ctx ctx_len"],
        ctx_attn_mask: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
    ):
        """Encode system prompt through frozen context encoder."""
        with torch.no_grad():
            ctx_features = self.ctx_encoder(
                input_ids=ctx_ids,
                attention_mask=ctx_attn_mask,
                position_ids=ctx_position_ids,
            )
        return ctx_features

    def generate_weights(
        self,
        ctx_ids,
        ctx_attn_mask=None,
        ctx_position_ids=None,
    ):
        """Full weight generation pipeline.

        Returns:
            (combined_lora_dict, basis_coefficients)
            where combined_lora_dict maps virtual_name -> {A, B}
        """
        from ctx_to_lora.modeling.model_constants import VIRTUAL_MODULE_SPECS
        from einops import rearrange

        # 1. Encode context
        ctx_features = self._encode_context(ctx_ids, ctx_attn_mask, ctx_position_ids)

        # 2. Handle per-layer features for perceiver
        # Flatten layers into sequence dimension so perceiver processes all layers together
        # Output: (bs, n_output_queries, d) — no per-layer batch expansion needed
        if ctx_features.dim() == 4:
            from einops import repeat
            bs, n_layers_ctx, seq_len, d = ctx_features.shape
            features_flat = rearrange(
                ctx_features,
                "bs n_layers seq_len d -> bs (n_layers seq_len) d",
            )
            attn_mask_for_perceiver = ctx_attn_mask
            if attn_mask_for_perceiver is not None:
                attn_mask_for_perceiver = repeat(
                    attn_mask_for_perceiver,
                    "bs seq_len -> bs (n_layers seq_len)",
                    n_layers=n_layers_ctx,
                )
        else:
            bs = ctx_features.shape[0]
            features_flat = ctx_features
            attn_mask_for_perceiver = ctx_attn_mask

        # 3. Perceiver: (bs, seq, d) -> (n_ctx, n_output_queries, d)
        #    The perceiver may unpack packed sequences via position_ids,
        #    so output batch size (n_ctx) can differ from input batch size.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            flat_latents = self.perceiver(features_flat, attn_mask_for_perceiver, ctx_position_ids)
        bs = flat_latents.shape[0]  # actual number of contexts (may differ from packed batch size)

        # 4. Coefficient head: pool latents → basis coefficients
        basis_lora_dict = None
        coefficients = None
        if self.basis_bank is not None:
            coefficients = self.coefficient_head(flat_latents)

            # 5. Basis mixing
            basis_lora_dict = self.basis_bank.mix(coefficients)

            # 6. Refinement: inject coefficients into latents
            flat_latents = self.refinement_blocks(flat_latents, coefficients)

        # 7. Split flat latents into per-module queries
        hconfig = self.hyperdistill_config
        latent_queries = {}
        offset = 0
        for vname in hconfig.hyper_target_modules:
            if vname not in hconfig.basis_module_specs:
                continue
            n_layers = hconfig.basis_module_specs[vname][0]
            n_queries = n_layers * hconfig.lora_rank
            queries = flat_latents[:, offset:offset + n_queries]
            queries = queries.view(bs, n_layers, hconfig.lora_rank, hconfig.latent_size)
            latent_queries[vname] = queries
            offset += n_queries

        # 8. Multi-head LoRA generation
        hyper_lora_dict = self.hyper_heads(latent_queries)

        return basis_lora_dict, hyper_lora_dict, coefficients

    def _build_real_lora_dict(
        self,
        basis_lora_dict: dict | None,
        hyper_lora_dict: dict | None,
    ) -> dict[str, dict[int, dict[str, Tensor]]]:
        """Convert virtual-name LoRA dicts to real module paths, combining basis + hyper.

        For modules targeted by both basis and hyper:
            Combined A/B = basis A/B + hyper A/B (additive)
        For modules targeted by basis only:
            Use basis A/B directly

        Returns dict mapping real_module_short_name -> layer_idx -> {
            A: (batch, rank, d_in),
            B: (batch, rank, d_out)
        }

        Uses per-layer storage to support heterogeneous dimensions across layer types
        (e.g., v_proj has d_out=1024 on DeltaNet layers but d_out=256 on FullAttn layers).
        """
        from ctx_to_lora.modeling.model_constants import VIRTUAL_MODULE_SPECS

        # Determine batch size from available data
        sample_dict = hyper_lora_dict or basis_lora_dict
        if sample_dict is None:
            return {}

        # Collect all virtual modules
        all_virtual = set()
        if basis_lora_dict:
            all_virtual.update(basis_lora_dict.keys())
        if hyper_lora_dict:
            all_virtual.update(hyper_lora_dict.keys())

        # Build per-real-module, per-layer dicts
        # real_loras[real_name][layer_idx] = {"A": (batch, r, d_in), "B": (batch, r, d_out)}
        real_loras: dict[str, dict[int, dict[str, Tensor]]] = {}

        for vname in all_virtual:
            if vname not in VIRTUAL_MODULE_SPECS:
                continue
            spec = VIRTUAL_MODULE_SPECS[vname]
            # Use full module path (e.g. "self_attn.q_proj", "linear_attn.out_proj", "mlp.down_proj")
            full_path = f"{spec['path_prefix']}.{spec['real_name']}"
            layer_indices = spec["layer_indices"]

            if full_path not in real_loras:
                real_loras[full_path] = {}

            # Add basis contribution
            if basis_lora_dict and vname in basis_lora_dict:
                basis_A = basis_lora_dict[vname]["A"]  # (batch, n_virtual_layers, rank, d_in)
                basis_B = basis_lora_dict[vname]["B"]
                for vi, li in enumerate(layer_indices):
                    if li not in real_loras[full_path]:
                        real_loras[full_path][li] = {
                            "A": basis_A[:, vi],
                            "B": basis_B[:, vi],
                        }
                    else:
                        real_loras[full_path][li]["A"] = real_loras[full_path][li]["A"] + basis_A[:, vi]
                        real_loras[full_path][li]["B"] = real_loras[full_path][li]["B"] + basis_B[:, vi]

            # Add hyper contribution
            if hyper_lora_dict and vname in hyper_lora_dict:
                hyper_A = hyper_lora_dict[vname]["A"]
                hyper_B = hyper_lora_dict[vname]["B"]
                for vi, li in enumerate(layer_indices):
                    if li not in real_loras[full_path]:
                        real_loras[full_path][li] = {
                            "A": hyper_A[:, vi],
                            "B": hyper_B[:, vi],
                        }
                    else:
                        real_loras[full_path][li]["A"] = real_loras[full_path][li]["A"] + hyper_A[:, vi]
                        real_loras[full_path][li]["B"] = real_loras[full_path][li]["B"] + hyper_B[:, vi]

        return real_loras

    def forward(
        self,
        ctx_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_attn_mask: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
        n_queries: Integer[Tensor, "n_ctx"] | None = None,
        return_generated_lora: bool | None = False,
        *model_inputs_args: Any,
        **model_inputs_kwargs: dict[str, Any],
    ) -> tuple | ModelOutput:
        """Forward pass of HyperDistill model."""
        generated_loras = None
        coefficients = None

        if ctx_ids is not None:
            # Generate LoRA weights
            basis_loras, hyper_loras, coefficients = self.generate_weights(
                ctx_ids, ctx_attn_mask, ctx_position_ids,
            )

            # Combine into real module LoRA dict
            generated_loras = self._build_real_lora_dict(basis_loras, hyper_loras)

        if generated_loras is not None:
            position_ids = model_inputs_kwargs.get("position_ids", None)

            if n_queries is None:
                if ctx_position_ids is None:
                    n_queries = torch.ones(
                        ctx_ids.shape[0], dtype=torch.int32, device=self.device
                    )
                else:
                    n_queries = torch.ones(
                        (ctx_position_ids == 0).sum(),
                        dtype=torch.int32,
                        device=self.device,
                    )

            apply_per_layer_lora(
                self.base_model,
                generated_loras,
                n_queries,
                position_ids,
            )

        model_outputs = self.base_model(*model_inputs_args, **model_inputs_kwargs)

        if return_generated_lora:
            return model_outputs, (generated_loras, coefficients)
        return model_outputs


# needed for loading model from checkpoint
# see https://github.com/huggingface/transformers/pull/34632
torch.serialization.add_safe_globals(
    [
        AggregatorConfig,
        LoraConfig,
        HypernetConfig,
        HyperDistillConfig,
        PeftType,
        TaskType,
        LoraRuntimeConfig,
        set,  # for real?
    ]
)
