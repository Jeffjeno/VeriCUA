from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import build_module

class MiniMaxM2SelfAttention(SelfAttention):

    def __init__(self, config, submodules, *args, **kwargs):

        q_layernorm = submodules.q_layernorm
        k_layernorm = submodules.k_layernorm
        submodules.q_layernorm = IdentityOp
        submodules.k_layernorm = IdentityOp

        super().__init__(config, submodules, *args, **kwargs)

        submodules.q_layernorm = q_layernorm
        submodules.k_layernorm = k_layernorm

        self.q_norm = build_module(
            q_layernorm,
            hidden_size=self.hidden_size_per_attention_head * config.num_attention_heads,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.k_norm = build_module(
            k_layernorm,
            hidden_size=self.hidden_size_per_attention_head * config.num_query_groups,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, *args, **kwargs):
        query, key, value = super().get_query_key_value_tensors(hidden_states, key_value_states, *args, **kwargs)

        query = query.reshape(*query.shape[:-2], -1)
        key = key.reshape(*key.shape[:-2], -1)

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            query = gather_from_tensor_model_parallel_region(query)
            key = gather_from_tensor_model_parallel_region(key)

        query = self.q_norm(query)
        key = self.k_norm(key)

        if tp_size > 1:
            query = scatter_to_tensor_model_parallel_region(query)
            key = scatter_to_tensor_model_parallel_region(key)

        query = query.view(*query.shape[:2], -1, self.hidden_size_per_attention_head)
        key = key.view(*key.shape[:2], -1, self.hidden_size_per_attention_head)

        return query, key, value

def get_minimax_m2_layer_spec(args, config, vp_stage=None):

    kwargs = {"use_transformer_engine": args.transformer_impl == "transformer_engine"}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    spec = get_gpt_decoder_block_spec(config, **kwargs)

    for layer_spec in spec.layer_specs:
        layer_spec.submodules.self_attention.module = MiniMaxM2SelfAttention

    return spec
