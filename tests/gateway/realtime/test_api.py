import asyncio
import json
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.realtime.controls import ControlEventBroker
from gateway.realtime.manager import RealtimeSessionManager
from gateway.realtime.openai_sideband import (
    OpenAIRealtimeCallClient,
    OpenAIRealtimeSideband,
)
from gateway.realtime.protocol import RealtimeVoiceConfig
from gateway.realtime.session import REALTIME_SYSTEM_GUIDANCE
from hermes_state import SessionDB


class _FakeSession:
    def __init__(self, session_id="voice_1"):
        self.session_id = session_id
        self.broker = ControlEventBroker(session_id)
        self.closed = False

    async def handle_control_command(self, command):
        if command["type"] == "session.ping":
            self.broker.publish(
                "session.pong", {"request_id": command.get("request_id")}
            )


class _FakeManager:
    def __init__(self):
        self.config = RealtimeVoiceConfig(enabled=True)
        self.session = _FakeSession()
        self.session.config = self.config
        self.created_offer = ""

    async def create_session(self, offer_sdp, requested_session_id=None):
        self.created_offer = offer_sdp
        if requested_session_id:
            self.session.session_id = requested_session_id
        return SimpleNamespace(
            session=self.session,
            answer_sdp="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            call_id="call_1",
        )

    def require(self, session_id):
        assert session_id == self.session.session_id
        return self.session

    async def require_active(self, session_id):
        return self.require(session_id)

    async def close_session(self, session_id, reason="client_closed"):
        if session_id != self.session.session_id:
            return False
        self.session.closed = True
        return True


def _adapter_and_app():
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "gateway-secret"})
    )
    manager = _FakeManager()
    adapter._realtime_runtime_status = lambda: (
        manager.config,
        "sk-openai-server-only",
        True,
    )
    adapter._ensure_realtime_manager = lambda: manager
    app = web.Application()
    app.router.add_post(
        "/v1/realtime/sessions", adapter._handle_realtime_session_create
    )
    app.router.add_get(
        "/v1/realtime/sessions/{session_id}/control",
        adapter._handle_realtime_control,
    )
    app.router.add_delete(
        "/v1/realtime/sessions/{session_id}",
        adapter._handle_realtime_session_delete,
    )
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return adapter, manager, app


@pytest.mark.asyncio
async def test_create_returns_sdp_metadata_without_standard_openai_key():
    adapter, manager, app = _adapter_and_app()
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/v1/realtime/sessions",
            data="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            headers={
                "Authorization": "Bearer gateway-secret",
                "Content-Type": "application/sdp",
            },
        )
        payload = await response.json()
    finally:
        await client.close()
        adapter._response_store.close()

    assert response.status == 200
    assert payload["session_id"] == "voice_1"
    assert payload["call_id"] == "call_1"
    assert payload["model"] == "gpt-realtime"
    assert payload["voice"] == "marin"
    assert manager.created_offer.startswith("v=0")
    assert "sk-openai-server-only" not in str(payload)


@pytest.mark.asyncio
async def test_create_rejects_client_model_override():
    adapter, _manager, app = _adapter_and_app()
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/v1/realtime/sessions",
            json={
                "sdp": "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
                "model": "unapproved-model",
            },
            headers={"Authorization": "Bearer gateway-secret"},
        )
        payload = await response.json()
    finally:
        await client.close()
        adapter._response_store.close()

    assert response.status == 400
    assert payload["error"]["code"] == "model_override_not_allowed"


@pytest.mark.asyncio
async def test_control_socket_replays_state_and_accepts_structured_ping():
    adapter, manager, app = _adapter_and_app()
    manager.session.broker.publish("session.state", {"state": "ready"})
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        socket = await client.ws_connect(
            "/v1/realtime/sessions/voice_1/control",
            headers={"Authorization": "Bearer gateway-secret"},
        )
        replay = await socket.receive_json()
        await socket.send_json(
            {
                "version": 1,
                "type": "session.ping",
                "request_id": "request_1",
                "data": {},
            }
        )
        pong = await socket.receive_json()
        await socket.close()
    finally:
        await client.close()
        adapter._response_store.close()

    assert replay["type"] == "session.state"
    assert pong["type"] == "session.pong"
    assert pong["data"]["request_id"] == "request_1"


@pytest.mark.asyncio
async def test_capabilities_report_transport_and_authorization_semantics():
    adapter, _manager, app = _adapter_and_app()
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get(
            "/v1/capabilities",
            headers={"Authorization": "Bearer gateway-secret"},
        )
        payload = await response.json()
    finally:
        await client.close()
        adapter._response_store.close()

    details = payload["features"]["realtime_voice_details"]
    assert payload["features"]["realtime_voice"] is True
    assert details["transport"] == "webrtc_sideband"
    assert details["structured_approvals"] is True
    assert details["voice_authorization"] is False
    assert details["barge_in"]["cancels_started_tools"] is False


async def _receive_matching(queue, event_type):
    while True:
        event = await asyncio.wait_for(queue.get(), timeout=5)
        if event.get("type") == event_type:
            return event


async def _receive_control_matching(socket, event_type):
    while True:
        event = await asyncio.wait_for(socket.receive_json(), timeout=5)
        if event.get("type") == event_type:
            return event


@pytest.mark.asyncio
async def test_api_real_manager_round_trip_with_fake_openai_peers(tmp_path):
    provider_events = asyncio.Queue()
    provider_connected = asyncio.Event()
    provider = {}

    async def create_call(request):
        reader = await request.multipart()
        fields = {}
        async for part in reader:
            fields[part.name] = await part.text()
        provider["authorization"] = request.headers.get("Authorization")
        provider["session"] = json.loads(fields["session"])
        return web.Response(
            text="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            content_type="application/sdp",
            headers={"Location": "/v1/realtime/calls/call_integration"},
        )

    async def sideband(request):
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        provider["call_id"] = request.query.get("call_id")
        provider["socket"] = socket
        provider_connected.set()
        async for message in socket:
            if message.type == web.WSMsgType.TEXT:
                await provider_events.put(json.loads(message.data))
        return socket

    provider_app = web.Application()
    provider_app.router.add_post("/v1/realtime/calls", create_call)
    provider_app.router.add_get("/v1/realtime", sideband)
    provider_server = TestServer(provider_app)
    await provider_server.start_server()

    db = SessionDB(db_path=tmp_path / "state.db")
    config = RealtimeVoiceConfig(
        enabled=True,
        intermediate_speech_enabled=False,
        provider_call_max_seconds=3600,
    )
    provider_base_url = str(provider_server.make_url("/v1")).rstrip("/")
    websocket_url = str(provider_server.make_url("/v1/realtime")).replace(
        "http://", "ws://", 1
    )

    def agent_factory(session_id):
        from run_agent import AIAgent

        return AIAgent(
            model=config.model,
            api_key="test-realtime-key",
            base_url=provider_base_url,
            provider="openai-api",
            api_mode="chat_completions",
            enabled_toolsets=["skills"],
            session_id=session_id,
            platform="realtime_voice",
            session_db=db,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            ephemeral_system_prompt=REALTIME_SYSTEM_GUIDANCE,
        )

    manager = RealtimeSessionManager(
        config=config,
        api_key="test-realtime-key",
        agent_factory=agent_factory,
        call_client=OpenAIRealtimeCallClient(
            "test-realtime-key",
            base_url=provider_base_url,
        ),
        sideband_factory=lambda call_id, handler: OpenAIRealtimeSideband(
            "test-realtime-key",
            call_id,
            handler,
            websocket_url=websocket_url,
        ),
    )
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "gateway-secret"})
    )
    adapter._realtime_runtime_status = lambda: (
        config,
        "test-realtime-key",
        True,
    )
    adapter._ensure_realtime_manager = lambda: manager
    api_app = web.Application()
    api_app.router.add_post(
        "/v1/realtime/sessions", adapter._handle_realtime_session_create
    )
    api_app.router.add_get(
        "/v1/realtime/sessions/{session_id}/control",
        adapter._handle_realtime_control,
    )
    api_app.router.add_delete(
        "/v1/realtime/sessions/{session_id}",
        adapter._handle_realtime_session_delete,
    )
    client = TestClient(TestServer(api_app))
    await client.start_server()
    authorization = {
        "Authorization": f"Bearer {adapter._expected_api_key()}"
    }
    control = None
    try:
        response = await client.post(
            "/v1/realtime/sessions",
            json={
                "session_id": "voice-integration",
                "sdp": "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            },
            headers=authorization,
        )
        payload = await response.json()
        assert response.status == 200
        assert payload["session_id"] == "voice-integration"
        assert payload["call_id"] == "call_integration"
        assert "test-realtime-key" not in json.dumps(payload)
        await asyncio.wait_for(provider_connected.wait(), timeout=5)

        session_config = provider["session"]
        assert REALTIME_SYSTEM_GUIDANCE.strip() in session_config["instructions"]
        assert "skill_view" in {tool["name"] for tool in session_config["tools"]}
        assert session_config["audio"]["input"]["turn_detection"][
            "create_response"
        ] is False
        assert provider["call_id"] == "call_integration"

        control = await client.ws_connect(
            "/v1/realtime/sessions/voice-integration/control",
            headers=authorization,
        )
        await provider["socket"].send_json(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "input_integration",
                "transcript": "Say hello.",
            }
        )
        response_create = await _receive_matching(
            provider_events, "response.create"
        )
        assert response_create["response"].get("conversation", "auto") == "auto"

        await provider["socket"].send_json(
            {
                "type": "response.output_audio_transcript.done",
                "response_id": "response_integration",
                "transcript": "Hello from the voice agent.",
            }
        )
        await provider["socket"].send_json(
            {
                "type": "response.done",
                "response": {
                    "id": "response_integration",
                    "status": "completed",
                    "output": [],
                },
            }
        )
        completed = await _receive_control_matching(control, "turn.completed")
        assert completed["data"]["provider_response_id"] == "response_integration"

        messages = db.get_messages("voice-integration")
        assert [message["role"] for message in messages[-2:]] == [
            "user",
            "assistant",
        ]
        assert messages[-2]["content"] == "Say hello."
        assert messages[-1]["content"] == "Hello from the voice agent."
        persisted = db.get_realtime_session_state("voice-integration")
        assert persisted["provider_call_id"] == "call_integration"
        assert persisted["frozen_instructions"] == session_config["instructions"]

        await control.close()
        control = None
        closed = await client.delete(
            "/v1/realtime/sessions/voice-integration",
            headers=authorization,
        )
        assert closed.status == 204
    finally:
        if control is not None:
            await control.close()
        await manager.close_all()
        await client.close()
        adapter._response_store.close()
        await provider_server.close()
        db.close()
