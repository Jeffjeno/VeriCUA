import torch

from slime.backends.megatron_utils.misc_utils import strip_param_name_prefix

def remove_padding(name: str, param: torch.Tensor, vocab_size: int) -> torch.Tensor:

    stripped = strip_param_name_prefix(name)
    if stripped in {"embedding.word_embeddings.weight", "output_layer.weight"} or stripped.endswith(
        (".embedding.word_embeddings.weight", ".output_layer.weight")
    ):
        return param[:vocab_size]
    return param
