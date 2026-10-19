import inspect
import re
from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from slime.backends.megatron_utils.misc_utils import strip_param_name_prefix
from slime.utils.types import ParamInfo

def all_gather_param(name: str, param: torch.nn.Parameter) -> torch.Tensor:

    if "expert_bias" in name:
        return param

    assert hasattr(param, "tensor_model_parallel"), f"{name} does not have tensor_model_parallel attribute"
    if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
        return param.data

    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()

    param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
    dist.all_gather(param_partitions, param.data, group=tp_group)
    partition_dim = param.partition_dim
    assert param.partition_stride == 1 or (
        param.partition_stride == 2 and "linear_fc1" in name
    ), "partition_stride != 1 is not supported"

    if "linear_fc1.weight" in name or "linear_fc1.bias" in name:
        param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
        param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]

    if "linear_fc2.weight" in name:
        if partition_dim == 0:
            partition_dim = 1
    param = torch.cat(param_partitions, dim=partition_dim)
    return param

def all_gather_params_async(
    param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]],
) -> list[torch.Tensor]:

    gather_tasks = []
    handles = []

    for info, param in param_infos_and_params:

        if "expert_bias" in info.name:
            gather_tasks.append((info, param, None, None, None))
            handles.append(None)
        elif not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
            gather_tasks.append((info, param.data, None, None, None))
            handles.append(None)
        else:

            if ".experts." in info.name:
                tp_size = mpu.get_expert_tensor_parallel_world_size()
                tp_group = mpu.get_expert_tensor_parallel_group()
            else:
                tp_size = mpu.get_tensor_model_parallel_world_size()
                tp_group = mpu.get_tensor_model_parallel_group()

            param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
            handle = dist.all_gather(param_partitions, param.data, group=tp_group, async_op=True)
            gather_tasks.append((info, None, handle, param_partitions, param.partition_dim))
            handles.append(handle)

    for handle in handles:
        if handle is not None:
            handle.wait()

    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim in gather_tasks:
        if handle is None:

            param = direct_param
        else:

            assert partition_dim is not None, "partition_stride != 1 is not supported"

            if "linear_fc1.weight" in info.name or "linear_fc1.bias" in info.name:
                param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
                param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]

            if "linear_fc2.weight" in info.name:
                if partition_dim == 0:
                    partition_dim = 1
            param = torch.cat(param_partitions, dim=partition_dim)

        gathered_params.append(param)

    return gathered_params

def named_params_and_buffers(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    convert_to_global_name: bool = True,
    translate_gpu_to_cpu: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    if convert_to_global_name:
        ans = _named_params_and_buffers_global(args, model)
    else:
        ans = _named_params_and_buffers_vanilla(model)

    if translate_gpu_to_cpu:
        ans = ((name, _maybe_get_cpu_backup(tensor)) for name, tensor in ans)

    return ans

def _maybe_get_cpu_backup(x: torch.Tensor):
    from torch_memory_saver import torch_memory_saver

    if (cpu_tensor := torch_memory_saver.get_cpu_backup(x, zero_copy=True)) is not None:
        return cpu_tensor

    return x

def _named_params_and_buffers_vanilla(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    for vp_stage, model_module in enumerate(model):

        def _compute_fqn(name, vp_stage=vp_stage):
            return f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}"

        for name, param in model_module.named_parameters():
            yield _compute_fqn(name), param

        for name, buffer in model_module.named_buffers():

            if "expert_bias" not in name:
                continue
            yield _compute_fqn(name), buffer

def _named_params_and_buffers_global(
    args: Namespace, model: Sequence[torch.nn.Module]
) -> Iterator[tuple[str, torch.Tensor]]:

    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    if args.num_experts:
        expert_offset = ep_rank * args.num_experts // ep_size

    sig = inspect.signature(get_transformer_layer_offset)
    need_vp_stage = "vp_stage" in sig.parameters

    for vp_stage, model_module in enumerate(model):
        if need_vp_stage:
            layer_offset = get_transformer_layer_offset(model_module.config, vp_stage)
        else:
            layer_offset = get_transformer_layer_offset(model_module.config)
        for name, param in model_module.named_parameters():

            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:

                mtp_layers_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
                match = re.match(mtp_layers_pattern, name)
                if not match:
                    yield name, param
                    continue

                layer_idx, rest = match.groups()
                expert_pattern = r"transformer_layer\.mlp\.experts\.(.+)\.(weight|bias)(\d+)"
                match = re.match(expert_pattern, rest)
                if not match:
                    yield name, param
                    continue

                rest, param_type, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.mtp.layers.{layer_idx}.transformer_layer.mlp.experts.{rest}.{param_type}{expert_idx}", param
                continue

            layer_idx, rest = match.groups()
            layer_idx = int(layer_idx) + layer_offset

            expert_pattern = r"mlp\.experts\.(.+)\.(weight|bias)(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, param_type, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.decoder.layers.{layer_idx}.mlp.experts.{rest}.{param_type}{expert_idx}", param
            else:
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", param

        for name, buffer in model_module.named_buffers():

            if "expert_bias" not in name:
                continue

            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
            else:
                layer_idx, rest = match.groups()
                layer_idx = int(layer_idx) + layer_offset
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", buffer
