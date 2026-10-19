from argparse import Namespace
from pathlib import Path

from guirl.env_service import rollout_master_server as server

def _configure_root(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(
        server,
        "_ARGS",
        Namespace(guirl_probe_output_root=str(root)),
    )

def test_probe_artifacts_stay_under_server_root(tmp_path, monkeypatch):
    output_root = tmp_path / "server-owned"
    client_result_dir = tmp_path / "root-owned-client-result"
    _configure_root(monkeypatch, output_root)

    artifact_dir = server._probe_artifact_dir(
        {
            "artifact_key": str(client_result_dir / "../../escape"),
            "example_result_dir": str(client_result_dir),
        },
        lease_id="lease:worker:task",
        sampler_round=1,
    )

    assert artifact_dir.is_dir()
    assert artifact_dir.relative_to(output_root.resolve())
    assert artifact_dir.name == "round_01"
    assert not client_result_dir.exists()

def test_legacy_client_path_is_only_an_opaque_key(tmp_path, monkeypatch):
    output_root = tmp_path / "server-owned"
    legacy_client_dir = tmp_path / "legacy-client-result"
    _configure_root(monkeypatch, output_root)

    first = server._probe_artifact_dir(
        {"example_result_dir": str(legacy_client_dir)},
        lease_id="lease-a",
        sampler_round=1,
    )
    second = server._probe_artifact_dir(
        {"example_result_dir": str(legacy_client_dir)},
        lease_id="lease-a",
        sampler_round=2,
    )
    other_lease = server._probe_artifact_dir(
        {"example_result_dir": str(legacy_client_dir)},
        lease_id="lease-b",
        sampler_round=1,
    )

    assert first.parent == second.parent
    assert first != second
    assert first.parent != other_lease.parent
    assert not legacy_client_dir.exists()
