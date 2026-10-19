import asyncio

import pytest

from guirl.env_service.rpc_env_client import AsyncRpcEnvClient

@pytest.mark.parametrize("endpoint", [
    "/reset", "/step", "/get_obs", "/prepare_for_probe", "/evaluate", "/close",
])
@pytest.mark.parametrize("operation_id", [None, "explicit-v2-id"])
def test_retries_share_id_but_new_calls_get_new_ids(endpoint, operation_id, monkeypatch):
    from guirl.env_service import rpc_env_client

    monkeypatch.setattr(rpc_env_client, "DEFAULT_RETRY_DELAY", 0)
    sent = []

    class Response:
        status = 200

        async def __aenter__(self):
            if len(sent) % 2 == 1:
                raise asyncio.TimeoutError("response lost after execution")
            return self

        async def __aexit__(self, *_args):
            pass

        async def json(self):
            return {"ok": True}

    class Session:
        closed = False

        def post(self, url, json):
            sent.append(dict(json))
            return Response()

    async def run():
        client = AsyncRpcEnvClient("http://worker", max_retries=2)
        client._session = Session()
        payload = {"lease_id": "legacy"}
        if operation_id is not None:
            payload["operation_id"] = operation_id
        await client._post(endpoint, payload)
        await client._post(endpoint, payload)
        assert payload == ({"lease_id": "legacy"} if operation_id is None else {
            "lease_id": "legacy", "operation_id": operation_id,
        })

    asyncio.run(run())
    assert sent[0] == sent[1]
    assert sent[2] == sent[3]
    assert sent[0]["operation_id"]
    if operation_id is None:
        assert sent[0]["operation_id"] != sent[2]["operation_id"]
    else:
        assert sent[0]["operation_id"] == operation_id

def test_health_probes_have_short_timeout():
    from guirl.env_service.rpc_env_client import HEALTH_TIMEOUT
    timeouts = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def json(self):
            return {"ok": True}

    class Session:
        closed = False

        def get(self, url, *, timeout):
            timeouts.append(timeout.total)
            return Response()

    async def run():
        client = AsyncRpcEnvClient("http://worker")
        client._session = Session()
        assert await client.healthcheck()
        assert await client.status() == {"ok": True}
    asyncio.run(run())
    assert timeouts == [10, 10]
    assert HEALTH_TIMEOUT.connect == 5
