import asyncio
import json

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from gateway.realtime.openai_sideband import (
    OpenAIRealtimeCallClient,
    OpenAIRealtimeError,
    OpenAIRealtimeSideband,
)


@pytest.mark.asyncio
async def test_call_creation_keeps_standard_key_in_authorization_header():
    captured = {}

    async def create_call(request):
        captured["authorization"] = request.headers.get("Authorization")
        captured["safety_identifier"] = request.headers.get(
            "OpenAI-Safety-Identifier"
        )
        reader = await request.multipart()
        fields = {}
        async for part in reader:
            fields[part.name] = await part.text()
        captured["fields"] = fields
        return web.Response(
            text="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            content_type="application/sdp",
            headers={"Location": "/v1/realtime/calls/call_test"},
        )

    app = web.Application()
    app.router.add_post("/v1/realtime/calls", create_call)
    server = TestServer(app)
    await server.start_server()
    try:
        client = OpenAIRealtimeCallClient(
            "sk-server-only",
            base_url=str(server.make_url("/v1")).rstrip("/"),
            safety_identifier="stable-server-derived-id",
        )
        result = await client.create_call(
            "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            {"type": "realtime", "model": "gpt-realtime"},
        )
    finally:
        await server.close()

    assert result.call_id == "call_test"
    assert result.answer_sdp.startswith("v=0")
    assert captured["safety_identifier"] == "stable-server-derived-id"
    assert captured["authorization"] == "Bearer sk-server-only"
    assert "sk-server-only" not in json.dumps(captured["fields"])
    assert json.loads(captured["fields"]["session"])["model"] == "gpt-realtime"


@pytest.mark.asyncio
async def test_call_creation_surfaces_provider_failure_without_key():
    async def fail(_request):
        return web.Response(status=429, text="rate limited")

    app = web.Application()
    app.router.add_post("/v1/realtime/calls", fail)
    server = TestServer(app)
    await server.start_server()
    try:
        client = OpenAIRealtimeCallClient(
            "sk-secret",
            base_url=str(server.make_url("/v1")).rstrip("/"),
        )
        with pytest.raises(OpenAIRealtimeError) as caught:
            await client.create_call(
                "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
                {"type": "realtime"},
            )
    finally:
        await server.close()

    assert caught.value.status == 429
    assert caught.value.retryable is True
    assert "sk-secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_call_creation_uses_exact_custom_endpoint_and_query():
    captured = {}

    async def create_call(request):
        captured["path"] = request.path
        captured["route"] = request.query.get("route")
        return web.Response(
            text="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            content_type="application/sdp",
            headers={"Location": "/v1/realtime/calls/call_proxy"},
        )

    app = web.Application()
    app.router.add_post("/proxy/realtime/calls", create_call)
    server = TestServer(app)
    await server.start_server()
    try:
        call_url = str(server.make_url("/proxy/realtime/calls?route=stock"))
        client = OpenAIRealtimeCallClient("proxy-key", call_url=call_url)
        result = await client.create_call(
            "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            {"type": "realtime"},
        )
    finally:
        await server.close()

    assert result.call_id == "call_proxy"
    assert captured == {"path": "/proxy/realtime/calls", "route": "stock"}


class _FakeSocket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.incoming.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        self.closed = True
        await self.incoming.put(None)


@pytest.mark.asyncio
async def test_sideband_serializes_events_and_reconnects_same_call():
    sockets = []
    connects = []
    received = []

    async def connect(url, **kwargs):
        connects.append((url, kwargs))
        socket = _FakeSocket()
        sockets.append(socket)
        return socket

    async def on_event(event):
        received.append(event)

    sideband = OpenAIRealtimeSideband(
        "sk-secret",
        "call_123",
        on_event,
        connect_fn=connect,
    )
    await sideband.connect()
    await sideband.send({"type": "response.create"})
    await sockets[0].incoming.put(json.dumps({"type": "session.updated"}))
    await sockets[0].incoming.put(None)
    await sideband.receive_loop()
    await sideband.reconnect()
    await sideband.close()

    assert json.loads(sockets[0].sent[0])["type"] == "response.create"
    assert received == [{"type": "session.updated"}]
    assert len(connects) == 2
    assert all("call_id=call_123" in url for url, _kwargs in connects)
    assert all(
        details["additional_headers"]["Authorization"] == "Bearer sk-secret"
        for _url, details in connects
    )


@pytest.mark.asyncio
async def test_sideband_preserves_proxy_query_and_replaces_stale_call_id():
    connects = []

    async def connect(url, **kwargs):
        connects.append((url, kwargs))
        return _FakeSocket()

    sideband = OpenAIRealtimeSideband(
        "proxy-key",
        "call new/value",
        lambda _event: None,
        websocket_url=(
            "wss://proxy.example/v1/realtime"
            "?route=a%20b&flag&opaque=%FF&call_id=stale"
        ),
        connect_fn=connect,
    )
    await sideband.connect()
    await sideband.close()

    assert connects[0][0] == (
        "wss://proxy.example/v1/realtime"
        "?route=a%20b&flag&opaque=%FF&call_id=call%20new%2Fvalue"
    )
