# MobileWorld Evaluation

The `mobileworld` suite evaluates recorded Android trajectories with GUIRL.
It replays native Android actions and probes the live Android state with
GUI-Owl. MobileWorld source, services and downloaded trajectories are not
bundled; provision them with the dataset in `config/mobile_world_datasets/`.

Set `PYTHON` to an environment containing the project and MobileWorld
dependencies, then configure the MobileWorld backend and Qwen judge and
GUI-Owl probe endpoints:

```bash
export PYTHON=/path/to/environment-venv/bin/python
export MOBILE_WORLD_BACKENDS=http://127.0.0.1:6800
export MW_GUIRL_JUDGE_BASE_URL=http://127.0.0.1:8030/v1
export MW_GUIRL_PROBE_BASE_URL=http://127.0.0.1:8060/v1
export MW_GUIRL_PROBE_MODEL_NAME=GUI-Owl-1.5-32B-Think

python suites/mobileworld/doctor.py --method guirl_qwen4b_gui_owl --skip-llm
bash suites/mobileworld/run.sh guirl
```

Run a smoke episode, score results, or recover failed episodes with the same
method:

```bash
SMOKE_TASK=AcceptMeetingTask bash suites/mobileworld/smoke.sh
python suites/mobileworld/score.py --result-root results/mobileworld/guirl_qwen4b_gui_owl
bash suites/mobileworld/recover.sh
```

Use `RESULT_ROOT` to override the output location. `--offline-passive` is an
approximate fallback when native replay is unavailable; do not mix its results
with replay results.
