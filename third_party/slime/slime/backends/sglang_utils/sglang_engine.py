import dataclasses
import ipaddress
import logging
import multiprocessing
import os
import time
from urllib.parse import quote

import requests
import sglang_router
from packaging.version import parse
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from urllib3.exceptions import NewConnectionError

from slime.ray.ray_actor import RayActor
from slime.utils.http_utils import get_host_info

logger = logging.getLogger(__name__)

_ROUTER_REGISTER_TIMEOUT_SECS = float(os.environ.get("SLIME_ROUTER_REGISTER_TIMEOUT_SECS", "120"))
_ROUTER_REGISTER_MAX_ATTEMPTS = int(os.environ.get("SLIME_ROUTER_REGISTER_MAX_ATTEMPTS", "3"))
_HEALTH_PROBE_TIMEOUT_SECS = float(os.environ.get("SLIME_HEALTH_PROBE_TIMEOUT_SECS", "30"))
_HEALTH_PROBE_LOG_EVERY = 15

def _local_session() -> requests.Session:

    session = requests.Session()
    session.trust_env = False
    return session

def _local_request(method: str, url: str, **kwargs) -> requests.Response:

    with _local_session() as session:
        return session.request(method, url, **kwargs)

def get_base_gpu_id(args, rank):
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
    return start_index

def _to_local_gpu_id(physical_gpu_id: int) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return physical_gpu_id

    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]

    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)

    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible)-1} (local)."
    )

def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    if getattr(server_args, "encoder_only", False):
        from sglang.srt.disaggregation.encode_server import launch_server_process as sglang_launch_server_process

        return sglang_launch_server_process(
            server_args,
            start_method="spawn",
            wait_for_server=True,
        )

    from sglang.srt.entrypoints.http_server import launch_server

    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()

    if getattr(server_args, "node_rank", 0) != 0:
        return p

    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        is_process_alive=lambda: p.is_alive(),
    )

    return p

def _wait_server_healthy(base_url, api_key, is_process_alive):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }

    with _local_session() as session:
        attempt = 0
        while True:
            attempt += 1
            try:
                response = session.get(
                    f"{base_url}/health_generate", headers=headers, timeout=_HEALTH_PROBE_TIMEOUT_SECS
                )
                if response.status_code == 200:
                    break
                last_error = f"HTTP {response.status_code} {response.text[:200]!r}"
            except requests.RequestException as e:
                last_error = repr(e)

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            if attempt % _HEALTH_PROBE_LOG_EVERY == 0:
                logger.warning(
                    f"Still waiting for {base_url}/health_generate to return 200 "
                    f"(attempt {attempt}, last result: {last_error})"
                )
            time.sleep(2)

class SGLangEngine(RayActor):
    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        sglang_overrides: dict | None = None,
        num_gpus_per_engine: int | None = None,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine

    def init(
        self,
        dist_init_addr,
        port,
        nccl_port,
        host=None,
        disaggregation_bootstrap_port=None,
        router_ip=None,
        router_port=None,
    ):
        self.router_ip = router_ip if router_ip is not None else self.args.sglang_router_ip
        self.router_port = router_port if router_port is not None else self.args.sglang_router_port

        host = host or get_host_info()[1]

        def _format_v6_uri(addr):
            if not addr or addr.startswith("["):
                return addr
            try:
                if ipaddress.ip_address(addr).version == 6:
                    return f"[{addr}]"
            except ValueError:
                pass
            return addr

        host = _format_v6_uri(host)
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"

        server_args_dict, external_engine_need_check_fields = _compute_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
            sglang_overrides=self.sglang_overrides,
            num_gpus_per_engine=self.num_gpus_per_engine,
        )

        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]
        self.server_port = server_args_dict["port"]

        if self.args.rollout_external:
            self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
        else:
            self._init_normal(server_args_dict)

    def _init_external(self, expect_server_args, external_engine_need_check_fields):
        logger.info(f"Use external SGLang engine (rank={self.rank}, expect_server_args={expect_server_args})")

        def _get_actual_server_args():
            response = _local_request(
                "GET",
                f"http://{self.server_host}:{self.server_port}/get_server_info",
                timeout=_HEALTH_PROBE_TIMEOUT_SECS,
            )
            response.raise_for_status()
            return response.json()

        def _sanity_check_server_args(actual_server_args, expect_server_args):
            for name in external_engine_need_check_fields:
                expect_value = expect_server_args.get(name)
                actual_value = actual_server_args.get(name)
                assert (
                    actual_value == expect_value
                ), f"{name=} {expect_value=} {actual_value=} {expect_server_args=} {actual_server_args=}"

        _wait_server_healthy(
            base_url=f"http://{self.server_host}:{self.server_port}",
            api_key=None,
            is_process_alive=lambda: True,
        )
        actual_server_args = _get_actual_server_args()
        _sanity_check_server_args(actual_server_args, expect_server_args)

    def _init_normal(self, server_args_dict):
        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        self.process = launch_server_process(ServerArgs(**server_args_dict))

        if self.worker_type == "encoder":
            return

        if self.node_rank == 0 and self.router_ip and self.router_port:
            router_base = f"http://{self.router_ip}:{self.router_port}"
            worker_url = f"http://{self.server_host}:{self.server_port}"
            if parse(sglang_router.__version__) <= parse("0.2.1"):
                assert self.worker_type == "regular", "pd disaggregation is not supported in old router."
                response = self._register_with_router(f"{router_base}/add_worker?url={worker_url}")
            else:
                payload = {"url": worker_url, "worker_type": self.worker_type}
                if self.worker_type == "prefill":
                    payload["bootstrap_port"] = server_args_dict["disaggregation_bootstrap_port"]
                response = self._register_with_router(f"{router_base}/workers", json=payload)
            response.raise_for_status()

    def _register_with_router(self, url: str, **post_kwargs) -> requests.Response:

        worker_url = f"http://{self.server_host}:{self.server_port}"
        last_error = None
        with _local_session() as session:
            for attempt in range(1, _ROUTER_REGISTER_MAX_ATTEMPTS + 1):
                start = time.time()
                try:
                    response = session.post(url, timeout=_ROUTER_REGISTER_TIMEOUT_SECS, **post_kwargs)
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    last_error = e
                    logger.warning(
                        f"Router registration attempt {attempt}/{_ROUTER_REGISTER_MAX_ATTEMPTS} for worker "
                        f"{worker_url} failed after {time.time() - start:.1f}s: {e!r}"
                    )
                    if attempt < _ROUTER_REGISTER_MAX_ATTEMPTS:
                        time.sleep(min(5.0 * attempt, 15.0))
                    continue
                logger.info(
                    f"Router registration for worker {worker_url} -> {url} returned HTTP {response.status_code} "
                    f"in {time.time() - start:.1f}s: {response.text[:300]!r}"
                )
                return response
        raise RuntimeError(
            f"Failed to register SGLang worker {worker_url} with the router via {url} after "
            f"{_ROUTER_REGISTER_MAX_ATTEMPTS} attempts ({_ROUTER_REGISTER_TIMEOUT_SECS:.0f}s timeout each). "
            f"Last error: {last_error!r}. Check that the router process is alive and that no http_proxy "
            f"intercepts intra-cluster traffic (add the node IP to no_proxy)."
        ) from last_error

    def _make_request(self, endpoint: str, payload: dict | None = None):

        if self.node_rank != 0:
            return

        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        response = _local_request("POST", url, json=payload or {})
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    def health_generate(self, timeout: float = 5.0) -> bool:

        if self.node_rank != 0:
            return True

        response = _local_request(
            "GET",
            f"http://{self.server_host}:{self.server_port}/health_generate",
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):

        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_tensor",
            payload,
        )

    def flush_cache(self):

        if self.node_rank != 0:
            return

        for _ in range(60):
            try:
                response = _local_request("GET", f"http://{self.server_host}:{self.server_port}/flush_cache")
                if response.status_code == 200:
                    break
                logger.info(f"Error flushing cache: HTTP {response.status_code} {response.text!r}")
                time.sleep(1)
            except NewConnectionError as e:
                raise e
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
                time.sleep(1)
                continue
        else:
            raise TimeoutError("Timeout while flushing cache.")

    def get_url(self):
        if self.node_rank != 0:
            return None
        return f"http://{self.server_host}:{self.server_port}"

    def shutdown(self):
        if self.args.rollout_external:
            return

        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        if self.worker_type != "encoder" and self.node_rank == 0:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            router_base = f"http://{self.router_ip}:{self.router_port}"
            response = None
            if parse(sglang_router.__version__) <= parse("0.2.1"):
                response = _local_request(
                    "POST", f"{router_base}/remove_worker?url={worker_url}", timeout=_ROUTER_REGISTER_TIMEOUT_SECS
                )
            elif parse(sglang_router.__version__) < parse("0.3.0"):
                worker_url = quote(worker_url, safe="")
                response = _local_request(
                    "DELETE", f"{router_base}/workers/{worker_url}", timeout=_ROUTER_REGISTER_TIMEOUT_SECS
                )
            else:
                try:
                    all_workers = _local_request(
                        "GET", f"{router_base}/workers", timeout=_ROUTER_REGISTER_TIMEOUT_SECS
                    ).json()["workers"]
                    for worker in all_workers:
                        if worker["url"] == worker_url:
                            worker_id = worker["id"]
                            response = _local_request(
                                "DELETE", f"{router_base}/workers/{worker_id}", timeout=_ROUTER_REGISTER_TIMEOUT_SECS
                            )
                            break
                    else:
                        logger.warning(f"Worker {worker_url} not found in router during shutdown.")
                except Exception as e:
                    logger.warning(f"Failed to fetch workers list or remove worker: {e}")

            if response is not None:
                response.raise_for_status()
        kill_process_tree(self.process.pid)

    def get_weight_version(self):
        if self.node_rank != 0:
            return
        url = f"http://{self.server_host}:{self.server_port}/get_weight_version"
        response = _local_request("GET", url)
        response.raise_for_status()
        return response.json()["weight_version"]

    def set_weight_version(self, new_version: str):

        return self._make_request("update_weight_version", {"new_version": str(new_version)})

    def release_memory_occupation(self):
        self.flush_cache()
        return self._make_request("release_memory_occupation")

    def resume_memory_occupation(self, tags: list[str] = None):

        return self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
        )

    def check_weights(self, action: str):
        return self._make_request("weights_checker", {"action": action})

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str | None = None,
        weight_version: str | None = None,
        files: list[str] | None = None,
    ):

        payload: dict = {"model_path": model_path}
        if load_format is not None:
            payload["load_format"] = load_format
        if weight_version is not None:
            payload["weight_version"] = weight_version
        if files is not None:
            payload["files"] = files
        return self._make_request("update_weights_from_disk", payload)

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def destroy_weights_update_group(self, group_name):
        try:
            return self._make_request(
                "destroy_weights_update_group",
                {
                    "group_name": group_name,
                },
            )
        except requests.exceptions.RequestException:

            pass

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        flush_cache=False,
        weight_version: str | None = None,
        load_format: str | None = None,
        delta=None,
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        if load_format is not None:
            payload["load_format"] = load_format
        if delta is not None:

            import json
            from dataclasses import asdict

            payload["delta"] = json.dumps(
                {
                    "encoding": delta.encoding.value,
                    "params": [asdict(p) for p in delta.params],
                    "checksum": delta.checksum,
                }
            )
        return self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    def pause_generation(self):
        response = _local_request("POST", f"http://{self.server_host}:{self.server_port}/pause_generation", json={})
        response.raise_for_status()
        return response

    def continue_generation(self):
        response = _local_request("POST", f"http://{self.server_host}:{self.server_port}/continue_generation", json={})
        response.raise_for_status()
        return response

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
    ):

        return self._make_request(
            "post_process_weights",
            {
                "restore_weights_before_load": restore_weights_before_load,
                "post_process_quantization": post_process_quantization,
            },
        )

    def start_profile(
        self,

        output_dir: str | None = None,

        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        profile_by_stage: bool = False,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ):
        response = _local_request(
            "POST",
            f"http://{self.server_host}:{self.server_port}/start_profile",
            json={
                "output_dir": output_dir,
                "start_step": start_step,
                "num_steps": num_steps,
                "activities": activities,
                "profile_by_stage": profile_by_stage,
                "with_stack": with_stack,
                "record_shapes": record_shapes,
            },
        )
        response.raise_for_status()
        return response

    def stop_profile(self):
        response = _local_request("POST", f"http://{self.server_host}:{self.server_port}/stop_profile", json={})
        response.raise_for_status()
        return response

    def simulate_crash(self):
        if self.args.rollout_external or not getattr(self, "process", None):
            logger.info(
                "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
                self.args.rollout_external,
            )
            return

        logger.info(f"Simulating crash on engine {self.server_host}:{self.server_port}...")
        self.shutdown()

def _compute_server_args(
    args,
    rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int | None = None,
    sglang_overrides: dict | None = None,
    num_gpus_per_engine: int | None = None,
):
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    nnodes = max(1, _gpus_per_engine // args.num_gpus_per_node)
    node_rank = rank % nnodes
    base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)
    base = _to_local_gpu_id(base)
    kwargs = {

        "model_path": (
            getattr(args, "sglang_model_path", None)
            or os.environ.get("SLIME_SGLANG_MODEL_PATH")
            or args.hf_checkpoint
        ),
        "trust_remote_code": True,
        "random_seed": args.seed + rank,

        "enable_memory_saver": args.offload_rollout,

        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base,

        "tp_size": _gpus_per_engine // args.sglang_pp_size,
        "dp_size": args.sglang_dp_size,
        "pp_size": args.sglang_pp_size,
        "ep_size": args.sglang_ep_size,

        "skip_server_warmup": True,

        "enable_draft_weights_cpu_backup": True,

        "enable_metrics": True,
    }

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        kwargs["load_balance_method"] = "follow_bootstrap_room"
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True
    elif worker_type == "encoder":
        kwargs["encoder_only"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    server_arg_fields = dataclasses.fields(ServerArgs)
    server_arg_field_names = {attr.name for attr in server_arg_fields}
    unused_keys = set(kwargs.keys())
    for attr in server_arg_fields:
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
        unused_keys.discard(attr.name)

    if sglang_overrides:
        for key, value in sglang_overrides.items():
            normalized_key = key.replace("-", "_")
            if normalized_key != key:
                logger.warning(
                    f"sglang_overrides key '{key}' normalized to '{normalized_key}' (rank={rank}). "
                    "Please use underscore style in YAML overrides."
                )
            if normalized_key in kwargs:
                logger.info(
                    f"sglang_overrides: overriding {normalized_key}={kwargs[normalized_key]} -> {value} (rank={rank})"
                )
            kwargs[normalized_key] = value
            if normalized_key in server_arg_field_names:
                unused_keys.discard(normalized_key)
            else:
                unused_keys.add(normalized_key)

    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    return kwargs, external_engine_need_check_fields

_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = [
    "model_path",
    "trust_remote_code",
    "random_seed",
    "nccl_port",
    "dist_init_addr",
    "skip_server_warmup",
    "enable_draft_weights_cpu_backup",
    "enable_metrics",
    "mem_fraction_static",
]
