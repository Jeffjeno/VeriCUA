# Online Judge Training

`run.sh` uses REINFORCE++ through the bundled Slime runtime. It preserves
the multimodal preprocessing, per-round fanout and judge-only loss mask.

Use separate Python environments for environment services and GPU training.
The GPU environment must provide compatible CUDA, PyTorch, SGLang,
Megatron-LM and Megatron Bridge installations. The bundled Slime sources
include local changes; replacing them with the latest upstream is unsupported.

```bash
pip install -r train/rft/requirements-env.txt
```

In the GPU environment:

```bash
pip install -r third_party/slime/requirements.txt -r train/rft/requirements-train.txt
pip install --no-deps --no-build-isolation -e third_party/slime
```

Start the four frozen actor endpoints independently, then start environment
services. Set endpoint URLs and paths for your installation:

```bash
export ACTOR_EVOCUA_API_BASE=http://localhost:30001/v1
export ACTOR_QWEN3VL4B_API_BASE=http://localhost:30002/v1
export ACTOR_HOLO3_API_BASE=http://localhost:30003/v1
export ACTOR_QWEN36_API_BASE=http://localhost:30004/v1
export AGENT_RUNTIME_VENV=/path/to/environment-venv
export OSWORLD_VM_PATH=/path/to/Ubuntu.qcow2
export GUIRL_RUN_DIR="$PWD/outputs/shaped"
bash train/rft/start_env.sh
```

Start the trainer on the Ray head, with the rollout node joined to the
same cluster and this checkout mounted at the same location on both nodes:

```bash
export HF_CHECKPOINT=/path/to/cold-start-checkpoint
export MEGATRON_DIR=/path/to/Megatron-LM
export HEAD_NODE_IP=127.0.0.1
export WORKER_NODE_IP=127.0.0.2
export PROBE_API_BASE=http://localhost:30005/v1
export WANDB_MODE=offline
export GUIRL_REWARD_MODE=shaped
export GUIRL_RUN_DIR="$PWD/outputs/shaped"
bash train/rft/run.sh
```

Replace loopback addresses with reachable addresses for multiple nodes.
Use dedicated Ray nodes: the launcher manages the local Ray instance.
The default trainer uses 8 training GPUs and 8 rollout GPUs; frozen actors
and the probe need additional inference capacity. The environment default
is 16 workers, independently configurable with `NUM_ENV_WORKERS`.

The 280-task manifest produces 18 updates per epoch: each update contains
16 tasks evaluated by each of four actors, for a global batch of 64.
The last task group is deterministically padded. Defaults are two epochs,
learning rate 1e-6, weight decay 0.1, clipping 0.2/0.28, and no KL loss.
Judge sampling uses temperature 1, a 24576-token limit and three rounds;
the frozen Holo3.1 probe uses temperature 0 and at most 15 steps.

For the binary reward ablation, use `GUIRL_REWARD_MODE=binary`, a fresh
`RUN_TAG`, output directory and `GUIRL_LOCAL_STATE_ROOT`. Use the same
local state root for the environment services and trainer of each run.
Keep all other settings identical.
Binary reward is 1 when the judge's success/failure verdict matches the
actor's original environment GT and 0 otherwise, including uncertain or
unparseable verdicts and missing GT. Probe use does not change binary reward.
The shaped mode retains the correctness-dependent probe penalty.
`FULL_EPOCHS=1` runs one epoch. Durable state must not be reused across
reward modes. `GUIRL_RENDER_ONLY=1` writes configuration and argv without
starting Ray, models or environment services.

Model weights, VM images, teacher outputs and downloaded benchmark files
are not bundled.
