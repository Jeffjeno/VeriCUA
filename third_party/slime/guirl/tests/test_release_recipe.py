import itertools
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from guirl.judge.evaluator import compute_reward

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('status,score,probe', itertools.product(
    ['success', 'failure', 'uncertain', 'parse_failure'], [None, 0.0, 0.5, 1.0], [False, True]))
def test_binary_reward(status, score, probe):
    expected = float(score is not None and status == ('success' if score >= 1 else 'failure'))
    assert compute_reward(status, score, used_probe=probe, reward_mode='binary') == expected


def test_ablation_changes_only_reward_and_state_identity(tmp_path):
    values = []
    for mode in ['shaped', 'binary']:
        output = tmp_path / mode
        env = {**os.environ, 'MEGATRON_DIR': '/path/to/megatron', 'HF_CHECKPOINT': '/path/to/model',
               'HEAD_NODE_IP': '127.0.0.1', 'WORKER_NODE_IP': '127.0.0.2', 'NET_IFACES': 'lo',
               'PROBE_API_BASE': 'http://127.0.0.1:30005/v1', 'GUIRL_RENDER_ONLY': '1',
               'GUIRL_REWARD_MODE': mode, 'GUIRL_RUN_DIR': str(output), 'RUN_TAG': mode,
               'GUIRL_V2_TRAINER_STATE_DIR': str(output / 'state'), 'GUIRL_V2_RUN_ID': mode,
               'TMPDIR': str(output / 'tmp'), 'RAY_TMPDIR': str(output / 'ray')}
        subprocess.run(['bash', str(ROOT / 'guirl/scripts/train_judge.sh')], env=env,
                       capture_output=True, check=True)
        argv = (output / 'train_argv.nul').read_bytes().decode().rstrip('\0').split('\0')
        for flag in ['--save', '--save-hf', '--custom-config-path', '--wandb-dir']:
            argv[argv.index(flag) + 1] = '<output>'
        expected = {'--global-batch-size': '64', '--rollout-batch-size': '64',
                    '--num-rollout': '36', '--advantage-estimator': 'reinforce_plus_plus',
                    '--lr': '1e-6', '--eps-clip': '0.2', '--eps-clip-high': '0.28'}
        for flag, value in expected.items():
            assert argv[argv.index(flag) + 1] == value
        cfg = yaml.safe_load((output / 'guirl_full_config.yaml').read_text())
        assert cfg.pop('guirl_reward_mode') == mode
        for key in ['guirl_v2_run_id', 'guirl_v2_trainer_state_dir']:
            cfg.pop(key)
        values.append((argv, cfg))
    assert values[0] == values[1]
