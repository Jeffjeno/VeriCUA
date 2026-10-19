from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule

from slime_plugins.minicpmv_common import (
    DEFAULT_PATCH_SIZE,
    DEFAULT_QUERY_NUM,
    scan_image_bounds,
    unflatten_pixel_values,
)

logger = logging.getLogger(__name__)

def _load_remote_vision_classes(checkpoint_path: str):

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    siglip_cls = get_class_from_dynamic_module(
        "modeling_navit_siglip.SiglipVisionTransformer", checkpoint_path
    )
    resampler_cls = get_class_from_dynamic_module("resampler.Resampler", checkpoint_path)
    return siglip_cls, resampler_cls

class MiniCPMVVLModel(MegatronModule):

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
        self.query_num = cfg.query_num
        self.patch_size = cfg.vision_patch_size
        self.vision_batch_size = cfg.vision_batch_size

        self.share_embeddings_and_output_weights = False

        self.vpm = None
        self.resampler = None
        if self.pre_process:
            siglip_cls, resampler_cls = _load_remote_vision_classes(cfg.hf_checkpoint_path)
            vision_config = cfg.hf_vision_config

            vision_config._attn_implementation = "eager"
            vpm = siglip_cls(vision_config)
            if cfg.drop_vision_last_layer:
                vpm.encoder.layers = vpm.encoder.layers[:-1]
            resampler = resampler_cls(
                num_queries=cfg.query_num,
                embed_dim=cfg.hidden_size,
                num_heads=cfg.hidden_size // 128,
                kv_dim=vision_config.hidden_size,
                adaptive=True,
            )
            hook_hf_module_setattr_for_tp_grad_sync(vpm)
            hook_hf_module_setattr_for_tp_grad_sync(resampler)
            vpm = vpm.to(dtype=cfg.params_dtype)
            resampler = resampler.to(dtype=cfg.params_dtype)
            if torch.cuda.is_available():
                vpm = vpm.to("cuda")
                resampler = resampler.to("cuda")
            self.vpm = vpm
            self.resampler = resampler

        self.language_model = MCoreGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="rope",
            rotary_percent=1.0,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=False,
            scatter_embedding_sequence_parallel=False,
        )
        self._apply_longrope_ext_factors(cfg)

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    def _apply_longrope_ext_factors(self, cfg) -> None:

        ext = getattr(cfg, "rope_ext_factors", None)
        if not ext:
            return
        rope = getattr(self.language_model, "rotary_pos_emb", None)
        if rope is None or not hasattr(rope, "inv_freq"):
            raise RuntimeError("longrope ext factors set but rotary_pos_emb.inv_freq not found")
        factors = torch.as_tensor(ext, dtype=rope.inv_freq.dtype, device=rope.inv_freq.device)
        assert factors.numel() == rope.inv_freq.numel(), (
            f"longrope factor count {factors.numel()} != inv_freq dims {rope.inv_freq.numel()}"
        )
        rope.inv_freq = rope.inv_freq / factors
        logger.info("Applied MiniCPM-V longrope ext factors to Megatron RoPE inv_freq")

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1
        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def _forward_vision(self, pixel_values_flat: torch.Tensor, tgt_sizes: torch.Tensor) -> torch.Tensor:

        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        device = self.vpm.embeddings.position_embedding.weight.device
        tgt_sizes = tgt_sizes.to(device=device, dtype=torch.int32)

        slices = unflatten_pixel_values(pixel_values_flat, tgt_sizes, self.patch_size)
        max_patches = int((tgt_sizes[:, 0].long() * tgt_sizes[:, 1].long()).max().item())

        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            slices, batch_first=True, padding_value=0.0
        )
        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

        patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
        valid = tgt_sizes[:, 0].long() * tgt_sizes[:, 1].long()
        patch_attn_mask[:, 0, :] = (
            torch.arange(max_patches, device=device).unsqueeze(0) < valid.unsqueeze(1)
        )

        all_pixel_values = all_pixel_values.to(device=device, dtype=dtype)
        if B > self.vision_batch_size:
            hs = []
            for i in range(0, B, self.vision_batch_size):
                hs.append(
                    self.vpm(
                        all_pixel_values[i : i + self.vision_batch_size],
                        patch_attention_mask=patch_attn_mask[i : i + self.vision_batch_size],
                        tgt_sizes=tgt_sizes[i : i + self.vision_batch_size],
                    ).last_hidden_state
                )
            vision_embedding = torch.cat(hs, dim=0)
        else:
            vision_embedding = self.vpm(
                all_pixel_values,
                patch_attention_mask=patch_attn_mask,
                tgt_sizes=tgt_sizes,
            ).last_hidden_state

        return self.resampler(vision_embedding, tgt_sizes)

    def _dummy_vision_term(self, device, dtype) -> torch.Tensor:

        dummy = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype)
        grid = 224 // self.patch_size
        tgt = torch.tensor([[grid, grid]], dtype=torch.int32, device=device)
        feats = self.resampler(self.vpm(dummy).last_hidden_state, tgt)
        return feats.sum() * 0.0

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
        tgt_sizes: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        assert inference_params is None, "Inference not supported"

        has_images = pixel_values_flat is not None and pixel_values_flat.numel() > 0
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size > 1 and has_images:
            raise NotImplementedError(
                "MiniCPM-V bridge does not support context parallel with images "
                "(image <unk> spans would straddle zigzag CP chunks); "
                "run with --context-parallel-size 1"
            )

        combined_embeddings = None
        if self.pre_process:

            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,
            ).clone()

            if has_images:
                feats = self._forward_vision(pixel_values_flat, tgt_sizes)
                hidden = combined_embeddings.shape[-1]
                bounds = scan_image_bounds(input_ids, self.query_num)
                if bounds.shape[0] != feats.shape[0]:

                    logger.error(
                        "MiniCPM-V vision scatter skipped: %d token spans != %d "
                        "pixel slices (stray sampled image tags?)",
                        bounds.shape[0],
                        feats.shape[0],
                    )
                    combined_embeddings = combined_embeddings + (
                        feats.sum() * 0.0
                    )
                elif bounds.shape[0] > 0:

                    emb = combined_embeddings.transpose(0, 1).contiguous()
                    bs = emb.shape[0]
                    flat = emb.reshape(-1, hidden)
                    idx = torch.cat(
                        [
                            torch.arange(int(s), int(e), device=flat.device)
                            for s, e in bounds.tolist()
                        ]
                    )
                    flat = flat.index_copy(
                        0, idx, feats.reshape(-1, hidden).to(flat.dtype)
                    )
                    combined_embeddings = (
                        flat.reshape(bs, -1, hidden).transpose(0, 1).contiguous()
                    )
            elif self.training and self.vpm is not None:
                combined_embeddings = combined_embeddings + self._dummy_vision_term(
                    combined_embeddings.device, combined_embeddings.dtype
                )

            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                    combined_embeddings
                ).contiguous()

        output = self.language_model(
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

        return output

@dataclass
class MiniCPMVModelProvider(GPTModelProvider):

    query_num: int = DEFAULT_QUERY_NUM
    vision_patch_size: int = DEFAULT_PATCH_SIZE
    vision_batch_size: int = 16
    drop_vision_last_layer: bool = False
    hf_vision_config: object = None
    hf_checkpoint_path: str = None

    rope_ext_factors: list = None

    position_embedding_type: str = "rope"
    scatter_embedding_sequence_parallel: bool = False
    language_max_sequence_length: int = 32768

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        if pre_process is None:
            pre_process = parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
        if post_process is None:
            post_process = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)

        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=False,
            multi_latent_attention=False,
        )

        return MiniCPMVVLModel(
            language_transformer_config=self,
            language_transformer_layer_spec=transformer_layer_spec,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )

@MegatronModelBridge.register_bridge(source="MiniCPMV", target=MiniCPMVVLModel)
class MiniCPMVBridge(MegatronModelBridge):

    def provider_bridge(self, hf_pretrained):
        hf_config = hf_pretrained.config
        model_dtype = self.dtype_from_hf(hf_config, default=torch.bfloat16)

        rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
        rope_ext_factors = None
        if rope_scaling.get("rope_type") == "longrope":
            short = rope_scaling.get("short_factor")
            long = rope_scaling.get("long_factor")
            assert short == long, (
                "MiniCPM-V bridge assumes short_factor == long_factor "
                "(static per-dim rescale); got differing factors"
            )
            rope_ext_factors = list(long)

        checkpoint_path = getattr(hf_config, "_name_or_path", None) or getattr(
            hf_pretrained, "model_name_or_path", None
        )
        assert checkpoint_path, "cannot resolve checkpoint path for remote vision code"

        provider = MiniCPMVModelProvider(

            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            kv_channels=getattr(hf_config, "head_dim", 128),
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
            add_qkv_bias=bool(getattr(hf_config, "attention_bias", False)),
            qk_layernorm=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            autocast_dtype=model_dtype,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            rotary_base=hf_config.rope_theta,
            share_embeddings_and_output_weights=bool(getattr(hf_config, "tie_word_embeddings", False)),
            vocab_size=hf_config.vocab_size,
            seq_length=hf_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,

            query_num=getattr(hf_config, "query_num", DEFAULT_QUERY_NUM),
            vision_patch_size=hf_config.vision_config.patch_size,
            vision_batch_size=getattr(hf_config, "vision_batch_size", 16),
            drop_vision_last_layer=bool(getattr(hf_config, "drop_vision_last_layer", False)),
            hf_vision_config=hf_config.vision_config,
            hf_checkpoint_path=checkpoint_path,

            rope_ext_factors=rope_ext_factors,
            language_max_sequence_length=hf_config.max_position_embeddings,
        )
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:

        param_mappings = {
            "language_model.embedding.word_embeddings.weight": "llm.model.embed_tokens.weight",
            "language_model.output_layer.weight": "llm.lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "llm.model.norm.weight",

            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "llm.model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "llm.model.layers.*.post_attention_layernorm.weight",

            "language_model.decoder.layers.*.input_layernorm.weight": "llm.model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "llm.model.layers.*.post_attention_layernorm.weight",

            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "llm.model.layers.*.self_attn.o_proj.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "llm.model.layers.*.mlp.down_proj.weight",
        }

        mapping_list = [
            AutoMapping(megatron_param=m, hf_param=h) for m, h in param_mappings.items()
        ]
        mapping_list.extend(
            [

                ReplicatedMapping(megatron_param="vpm.**", hf_param="vpm.**"),
                ReplicatedMapping(megatron_param="resampler.**", hf_param="resampler.**"),
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="llm.model.layers.*.self_attn.q_proj.weight",
                    k="llm.model.layers.*.self_attn.k_proj.weight",
                    v="llm.model.layers.*.self_attn.v_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="llm.model.layers.*.mlp.gate_proj.weight",
                    up="llm.model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)
