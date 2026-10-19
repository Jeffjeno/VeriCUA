# WebArena Evaluation

The `webarena` suite evaluates 122 recorded trajectories with GUIRL. It
restores trajectories in a live BrowserGym environment and uses the native
`WebArenaHolo_Probe` browser action surface. Upstream environments, website
services and downloaded trajectories are not bundled.

Set `paths.manifest` in `config/eval_datasets/webarena.yaml` to the downloaded
JSONL manifest. Each row must include the action and screenshot paths,
task metadata, runtime settings and original environment score.

Set `PYTHON` to an environment containing WebArena-Verified and BrowserGym
dependencies. Configure the Qwen3-VL-4B judge and Holo probe endpoints:

```bash
export PYTHON=/path/to/environment-venv/bin/python
export WA_GUIRL_JUDGE_BASE_URL=http://127.0.0.1:8030/v1
export WA_GUIRL_JUDGE_MODEL_NAME=Qwen3-VL-4B-Thinking
export WA_GUIRL_JUDGE_API_KEY=EMPTY
export WA_GUIRL_PROBE_BASE_URL=http://127.0.0.1:30000/v1
export WA_GUIRL_PROBE_MODEL_NAME=holo3-1-35b-a3b
export WA_GUIRL_PROBE_API_KEY=EMPTY

python suites/webarena/doctor.py --skip-llm
bash suites/webarena/run.sh guirl
```

Website URLs use `WA_EXTERNAL_HOST`, `WA_INSTANCE`, `WA_PORT_STRIDE` and
`WA_SHARE_READONLY`. Run each shard against an isolated website instance:

```bash
WA_INSTANCE=0 NUM_SHARDS=4 SHARD_INDEX=0 bash suites/webarena/run.sh guirl
SMOKE_CANDIDATE=holo__task-00015__seed-1__rev-2 bash suites/webarena/smoke.sh
python suites/webarena/score.py --result-root results/webarena_eval/guirl_qwen4b_holo
bash suites/webarena/recover.sh guirl
```

Use `RESULT_ROOT` to override output locations. Full website reset is the
default; `--skip-full-reset` requires a separately restored state.
