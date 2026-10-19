from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from megatron.bridge.models.conversion.mapping_registry import (
    MegatronMappingRegistry,
)
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import (
    hook_hf_module_setattr_for_tp_grad_sync,
)
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule

from slime_plugins.minicpmv46_common import (
    DEFAULT_PATCH_SIZE,
    IMAGE_TOKEN_ID,
    unflatten_pixel_values,
)
from slime_plugins.models.qwen3_5 import get_qwen3_5_spec

logger = logging.getLogger(__name__)

class PaddedVocabMapping(AutoMapping):

    def __init__(
        self,
        megatron_param: str,
        hf_param: str,
        hf_vocab_size: int,
        padded_vocab_size: int,
    ):
        super().__init__(megatron_param, hf_param)
        self.hf_vocab_size = hf_vocab_size
        self.padded_vocab_size = padded_vocab_size

    def hf_to_megatron(self, hf_weights, megatron_module):
        if hf_weights.shape[0] != self.hf_vocab_size:
            raise ValueError(
                f"expected {self.hf_vocab_size} HF vocab rows, got {hf_weights.shape}"
            )
        if self.padded_vocab_size > self.hf_vocab_size:
            padding = hf_weights.new_zeros(
                (self.padded_vocab_size - self.hf_vocab_size, *hf_weights.shape[1:])
            )
            hf_weights = torch.cat([hf_weights, padding], dim=0)
        return super().hf_to_megatron(hf_weights, megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        result = super().megatron_to_hf(megatron_weights, megatron_module)
        return {
            name: weight[: self.hf_vocab_size]
            for name, weight in result.items()
        }

    def resolve(self, captures):
        megatron_param, hf_param = self._resolve_names(captures)
        return type(self)(
            megatron_param,
            hf_param,
            self.hf_vocab_size,
            self.padded_vocab_size,
        )

class MiniCPMV46VLModel(MegatronModule):

    def __init__(
        self,
        language_transformer_config,
        language_transformer_layer_spec,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
    ) -> None:
        super().__init__(config=language_transformer_config)
        self.pre_process = pre_process
        self.post_process = post_process
        cfg = language_transformer_config
        self.image_token_id = cfg.image_token_id
        self.patch_size = cfg.vision_patch_size
        self.downsample_mode = cfg.downsample_mode
        self.share_embeddings_and_output_weights = False

        self.vision_tower = None
        self.merger = None
        if self.pre_process:
            from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import (
                MiniCPMV4_6Merger,
                MiniCPMV4_6VisionModel,
            )

            vision_config = deepcopy(cfg.hf_vision_config)
            vision_config._attn_implementation = cfg.vision_attention_impl
            vision_tower = MiniCPMV4_6VisionModel._from_config(vision_config)
            merger = MiniCPMV4_6Merger(deepcopy(cfg.hf_model_config))
            hook_hf_module_setattr_for_tp_grad_sync(vision_tower)
            hook_hf_module_setattr_for_tp_grad_sync(merger)
            vision_tower = vision_tower.to(dtype=cfg.params_dtype)
            merger = merger.to(dtype=cfg.params_dtype)
            if torch.cuda.is_available():
                vision_tower = vision_tower.cuda()
                merger = merger.cuda()
            self.vision_tower = vision_tower
            self.merger = merger

        self.language_model = MCoreGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="rope",
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=(
                language_transformer_config.share_embeddings_and_output_weights
            ),
            scatter_embedding_sequence_parallel=False,
        )
        self.share_embeddings_and_output_weights = (
            self.language_model.share_embeddings_and_output_weights
        )

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        if len(input_tensor) != 1:
            raise ValueError(f"expected one pipeline input, got {len(input_tensor)}")
        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def _forward_vision(
        self,
        pixel_values_flat: torch.Tensor,
        target_sizes: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.vision_tower.parameters()).device
        dtype = next(self.vision_tower.parameters()).dtype
        target_sizes = target_sizes.to(device=device, dtype=torch.int32)
        pixel_values = unflatten_pixel_values(
            pixel_values_flat.to(device=device),
            target_sizes,
            self.patch_size,
        ).to(dtype=dtype)
        use_vit_merger = self.downsample_mode != "4x"
        vision_output = self.vision_tower(
            pixel_values,
            target_sizes=target_sizes,
            use_vit_merger=use_vit_merger,
        )
        merger_sizes = target_sizes // 2 if use_vit_merger else target_sizes
        image_features = self.merger(
            vision_output.last_hidden_state,
            merger_sizes,
        )
        return torch.cat(image_features, dim=0)

    def _dummy_vision_term(self) -> torch.Tensor:
        device = next(self.vision_tower.parameters()).device
        dtype = next(self.vision_tower.parameters()).dtype
        size = 16
        target_sizes = torch.tensor([[size, size]], dtype=torch.int32, device=device)
        length = size * size * self.patch_size
        pixels = torch.zeros(
            (1, 3, self.patch_size, length),
            dtype=dtype,
            device=device,
        )
        use_vit_merger = self.downsample_mode != "4x"
        vision_output = self.vision_tower(
            pixels,
            target_sizes=target_sizes,
            use_vit_merger=use_vit_merger,
        )
        merger_sizes = target_sizes // 2 if use_vit_merger else target_sizes
        features = self.merger(vision_output.last_hidden_state, merger_sizes)
        return sum(feature.sum() for feature in features) * 0.0

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        inference_params=None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values_flat: torch.Tensor = None,
        target_sizes: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if inference_params is not None:
            raise NotImplementedError("Megatron MiniCPM-V 4.6 inference is unsupported")
        has_images = (
            pixel_values_flat is not None
            and target_sizes is not None
            and pixel_values_flat.numel() > 0
        )
        if has_images and parallel_state.get_context_parallel_world_size() != 1:
            raise NotImplementedError(
                "MiniCPM-V 4.6 image scatter requires context parallel size 1"
            )

        combined_embeddings = None
        if self.pre_process:
            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,
            ).clone()
            if has_images:
                features = self._forward_vision(pixel_values_flat, target_sizes)
                hidden = combined_embeddings.shape[-1]
                embeddings_bsh = combined_embeddings.transpose(0, 1).contiguous()
                flat_embeddings = embeddings_bsh.reshape(-1, hidden)
                image_mask = input_ids.reshape(-1) == self.image_token_id
                image_positions = torch.where(image_mask)[0]
                if image_positions.numel() != features.shape[0]:
                    raise ValueError(
                        "MiniCPM-V 4.6 image placeholders/features mismatch: "
                        f"{image_positions.numel()} != {features.shape[0]}"
                    )
                flat_embeddings = flat_embeddings.index_copy(
                    0,
                    image_positions.to(flat_embeddings.device),
                    features.to(
                        device=flat_embeddings.device,
                        dtype=flat_embeddings.dtype,
                    ),
                )
                combined_embeddings = (
                    flat_embeddings.reshape_as(embeddings_bsh)
                    .transpose(0, 1)
                    .contiguous()
                )
            elif self.training and self.vision_tower is not None:
                combined_embeddings = combined_embeddings + self._dummy_vision_term()

            if self.config.sequence_parallel:
                combined_embeddings = (
                    tensor_parallel.scatter_to_sequence_parallel_region(
                        combined_embeddings
                    ).contiguous()
                )

        return self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

@dataclass
class MiniCPMV46ModelProvider(GPTModelProvider):
    hf_vocab_size: int = 248094
    image_token_id: int = IMAGE_TOKEN_ID
    vision_patch_size: int = DEFAULT_PATCH_SIZE
    downsample_mode: str = "16x"
    vision_attention_impl: str = "sdpa"
    hf_vision_config: object = None
    hf_model_config: object = None
    hf_checkpoint: str = None
    qwen_gdn_backend: str = "fla"
    num_experts: int | None = None
    position_embedding_type: str = "rope"
    scatter_embedding_sequence_parallel: bool = False
    language_max_sequence_length: int = 262144

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        if pre_process is None:
            pre_process = parallel_state.is_pipeline_first_stage(
                ignore_virtual=False, vp_stage=vp_stage
            )
        if post_process is None:
            post_process = parallel_state.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=vp_stage
            )
        transformer_layer_spec = get_qwen3_5_spec(self, self, vp_stage)
        return MiniCPMV46VLModel(
            language_transformer_config=self,
            language_transformer_layer_spec=transformer_layer_spec,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )

try:
    from transformers.models.minicpmv4_6 import (
        MiniCPMV4_6ForConditionalGeneration as _MiniCPMV46HF,
    )
except ImportError:
    _MiniCPMV46HF = "MiniCPMV4_6ForConditionalGeneration"

@MegatronModelBridge.register_bridge(
    source=_MiniCPMV46HF,
    target=MiniCPMV46VLModel,
)
class MiniCPMV46Bridge(MegatronModelBridge):

    strict_hf_export = False

    def provider_bridge(self, hf_pretrained):
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        model_dtype = self.dtype_from_hf(text_config, default=torch.bfloat16)
        rope_parameters = getattr(text_config, "rope_parameters", {}) or {}
        checkpoint_path = getattr(hf_config, "_name_or_path", None) or getattr(
            hf_pretrained, "model_name_or_path", None
        )
        if not checkpoint_path:
            raise RuntimeError("cannot resolve MiniCPM-V 4.6 checkpoint path")

        padded_vocab_size = (
            (text_config.vocab_size + 255) // 256
        ) * 256
        return MiniCPMV46ModelProvider(
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,
            kv_channels=text_config.head_dim,
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            normalization="RMSNorm",
            layernorm_zero_centered_gamma=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            add_qkv_bias=bool(getattr(text_config, "attention_bias", False)),
            qk_layernorm=True,
            attention_output_gate=bool(
                getattr(text_config, "attn_output_gate", True)
            ),
            hidden_dropout=0.0,
            attention_dropout=text_config.attention_dropout,
            autocast_dtype=model_dtype,
            make_vocab_size_divisible_by=128,
            rotary_percent=float(
                rope_parameters.get(
                    "partial_rotary_factor",
                    getattr(text_config, "partial_rotary_factor", 0.25),
                )
            ),
            rotary_base=int(rope_parameters.get("rope_theta", 10_000_000)),
            share_embeddings_and_output_weights=bool(
                getattr(text_config, "tie_word_embeddings", True)
            ),
            vocab_size=padded_vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            hf_vocab_size=text_config.vocab_size,
            image_token_id=hf_config.image_token_id,
            vision_patch_size=hf_config.vision_config.patch_size,
            downsample_mode=hf_config.downsample_mode,
            hf_vision_config=deepcopy(hf_config.vision_config),
            hf_model_config=deepcopy(hf_config),
            hf_checkpoint=str(checkpoint_path),
            qwen_gdn_backend="fla",
            language_max_sequence_length=text_config.max_position_embeddings,
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        direct = {
            "language_model.decoder.final_layernorm.weight": (
                "model.language_model.norm.weight"
            ),
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": (
                "model.language_model.layers.*.input_layernorm.weight"
            ),
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": (
                "model.language_model.layers.*.post_attention_layernorm.weight"
            ),
            "language_model.decoder.layers.*.input_layernorm.weight": (
                "model.language_model.layers.*.input_layernorm.weight"
            ),
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": (
                "model.language_model.layers.*.post_attention_layernorm.weight"
            ),
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": (
                "model.language_model.layers.*.self_attn.o_proj.weight"
            ),
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": (
                "model.language_model.layers.*.self_attn.q_norm.weight"
            ),
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": (
                "model.language_model.layers.*.self_attn.k_norm.weight"
            ),
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": (
                "model.language_model.layers.*.mlp.down_proj.weight"
            ),
        }
        gdn_names = (
            "A_log",
            "conv1d.weight",
            "dt_bias",
            "in_proj_a.weight",
            "in_proj_b.weight",
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "norm.weight",
            "out_proj.weight",
        )
        mappings = [
            AutoMapping(megatron_param=megatron, hf_param=hf)
            for megatron, hf in direct.items()
        ]
        mappings.extend(
            [
                PaddedVocabMapping(
                    "language_model.embedding.word_embeddings.weight",
                    "model.language_model.embed_tokens.weight",
                    hf_vocab_size=248094,
                    padded_vocab_size=248320,
                ),
                PaddedVocabMapping(
                    "language_model.output_layer.weight",
                    "lm_head.weight",
                    hf_vocab_size=248094,
                    padded_vocab_size=248320,
                ),
            ]
        )
        mappings.extend(
            [
                ReplicatedMapping(
                    megatron_param="vision_tower.**",
                    hf_param="model.vision_tower.**",
                ),
                ReplicatedMapping(
                    megatron_param="merger.**",
                    hf_param="model.merger.**",
                ),
                ReplicatedMapping(
                    megatron_param=(
                        "language_model.decoder.layers.*."
                        "self_attention.input_layernorm.weight"
                    ),
                    hf_param="model.language_model.layers.*.input_layernorm.weight",
                ),
                *[
                    ReplicatedMapping(
                        megatron_param=(
                            "language_model.decoder.layers.*."
                            f"self_attention.linear_attn.{gdn_name}"
                        ),
                        hf_param=(
                            "model.language_model.layers.*."
                            f"linear_attn.{gdn_name}"
                        ),
                    )
                    for gdn_name in gdn_names
                ],
                QKVMapping(
                    megatron_param=(
                        "language_model.decoder.layers.*."
                        "self_attention.linear_qkv.weight"
                    ),
                    q="model.language_model.layers.*.self_attn.q_proj.weight",
                    k="model.language_model.layers.*.self_attn.k_proj.weight",
                    v="model.language_model.layers.*.self_attn.v_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param=(
                        "language_model.decoder.layers.*.mlp.linear_fc1.weight"
                    ),
                    gate="model.language_model.layers.*.mlp.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )
        return MegatronMappingRegistry(*mappings)
