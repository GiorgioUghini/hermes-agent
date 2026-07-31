import asyncio
import threading
import time
from unittest.mock import patch

import pytest

from gateway.realtime.manager import RealtimeSessionError, RealtimeSessionManager
from gateway.realtime.protocol import RealtimeCall, RealtimeVoiceConfig
from hermes_state import SessionDB


class _FakeAgent:
    def __init__(self, db, session_id, tools=None):
        self._session_db = db
        self.session_id = session_id
        self.tools = tools or []
        self._cached_system_prompt = ""

    def _ensure_db_session(self):
        return None


class _FakeSideband:
    def __init__(self, call_id):
        self.call_id = call_id
        self.sent = []
        self.connected = False
        self.closed = False
        self._stop = asyncio.Event()

    async def connect(self):
        self.connected = True

    async def send(self, event):
        self.sent.append(event)

    async def receive_loop(self):
        await self._stop.wait()

    async def reconnect(self):
        self.connected = True

    async def close(self):
        self.closed = True
        self._stop.set()


class _FakeCallClient:
    def __init__(self, *calls):
        self._calls = list(calls)
        self.requests = []

    async def create_call(self, offer_sdp, session_config):
        self.requests.append((offer_sdp, session_config))
        return self._calls.pop(0)


def _manager(db, *, call_client=None):
    sidebands = []

    def sideband_factory(call_id, _handler):
        sideband = _FakeSideband(call_id)
        sidebands.append(sideband)
        return sideband

    manager = RealtimeSessionManager(
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        api_key="sk-test",
        agent_factory=lambda session_id: _FakeAgent(
            db,
            session_id,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "current_tool",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        ),
        call_client=call_client,
        sideband_factory=sideband_factory,
    )
    return manager, sidebands


def test_manager_wires_configured_provider_endpoints(db):
    config = RealtimeVoiceConfig.from_config(
        {
            "realtime_voice": {
                "enabled": True,
                "transport": {
                    "call_url": "https://proxy.example/v1/realtime/calls",
                    "sideband_url": "wss://proxy.example/v1/realtime",
                },
            }
        }
    )
    manager = RealtimeSessionManager(
        config=config,
        api_key="proxy-key",
        agent_factory=lambda session_id: _FakeAgent(db, session_id),
    )

    assert manager._call_client._call_url == config.call_url
    sideband = manager._sideband_factory("call_proxy", lambda _event: None)
    assert sideband._websocket_url == config.sideband_url


def _persist_call(db, *, state="ready", started_at=None):
    db.save_realtime_session_state(
        "voice-recovery",
        provider_call_id="call_original",
        provider_call_started_at=started_at or time.time(),
        state=state,
        model="gpt-realtime-snapshot",
        voice="cedar",
        frozen_instructions="original frozen prompt",
        frozen_tools=[
            {
                "type": "function",
                "function": {
                    "name": "frozen_tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )


@pytest.fixture()
def db(tmp_path):
    state = SessionDB(db_path=tmp_path / "state.db")
    state.create_session(session_id="voice-recovery", source="realtime_voice")
    yield state
    state.close()


@pytest.mark.asyncio
async def test_require_active_reattaches_safe_call_with_frozen_snapshot(db):
    _persist_call(db)
    manager, sidebands = _manager(db)
    try:
        session = await manager.require_active("voice-recovery")
        assert session.call_id == "call_original"
        assert session.frozen_instructions == "original frozen prompt"
        assert session.config.model == "gpt-realtime-snapshot"
        assert session.config.voice == "cedar"
        assert [tool["function"]["name"] for tool in session.frozen_tools] == [
            "frozen_tool"
        ]
        assert sidebands[0].connected is True
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_require_active_demands_renewal_for_non_idle_projection(db):
    _persist_call(db, state="turn_active")
    manager, _sidebands = _manager(db)
    try:
        with pytest.raises(RealtimeSessionError) as caught:
            await manager.require_active("voice-recovery")
        assert caught.value.code == "session_renewal_required"
        assert caught.value.status == 409
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_persisted_renewal_reuses_prompt_tools_and_replays_tool_chain(db):
    _persist_call(db, state="renewal_required")
    db.append_message("voice-recovery", role="user", content="Find the news.")
    db.append_message(
        "voice-recovery",
        role="assistant",
        content="I will check.",
        tool_calls=[
            {
                "id": "tool_call_1",
                "type": "function",
                "function": {
                    "name": "frozen_tool",
                    "arguments": '{"query":"news"}',
                },
            }
        ],
    )
    db.append_message(
        "voice-recovery",
        role="tool",
        tool_name="frozen_tool",
        tool_call_id="tool_call_1",
        content='{"result":"latest"}',
    )
    db.append_message(
        "voice-recovery",
        role="assistant",
        content="Here is the latest.",
    )
    call_client = _FakeCallClient(
        RealtimeCall(answer_sdp="answer", call_id="call_replacement")
    )
    manager, sidebands = _manager(db, call_client=call_client)
    try:
        created = await manager.renew_session("voice-recovery", "offer")
        assert created.call_id == "call_replacement"
        sent_config = call_client.requests[0][1]
        assert sent_config["instructions"] == "original frozen prompt"
        assert sent_config["model"] == "gpt-realtime-snapshot"
        assert sent_config["audio"]["output"]["voice"] == "cedar"
        assert [tool["name"] for tool in sent_config["tools"]] == ["frozen_tool"]

        item_types = [
            event["item"]["type"]
            for event in sidebands[0].sent
            if event.get("type") == "conversation.item.create"
        ]
        assert item_types == [
            "message",
            "message",
            "function_call",
            "function_call_output",
            "message",
        ]
        output = next(
            event
            for event in sidebands[0].sent
            if event.get("item", {}).get("type") == "function_call_output"
        )
        assert output["item"]["call_id"] == "tool_call_1"
        assert "latest" in output["item"]["output"]
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_restart_requeues_running_review_from_durable_boundary(db):
    _persist_call(db)
    boundary = db.append_message(
        "voice-recovery",
        role="user",
        content="Remember this preference.",
    )
    db.mark_realtime_review_due(
        "voice-recovery",
        boundary_message_id=boundary,
        review_memory=True,
        review_skills=False,
    )
    assert db.mark_realtime_review_running(
        "voice-recovery", boundary_message_id=boundary
    )
    completed = threading.Event()
    snapshots = []

    def fake_spawn(_agent, snapshot, *, review_memory, review_skills):
        def target():
            snapshots.append((snapshot, review_memory, review_skills))
            completed.set()
            return True

        return target, "review"

    manager, _sidebands = _manager(db)
    try:
        with patch(
            "agent.background_review.spawn_background_review_thread",
            side_effect=fake_spawn,
        ):
            session = await manager.require_active("voice-recovery")
            assert await asyncio.to_thread(completed.wait, 2)
            await asyncio.to_thread(session._review_thread.join, 2)

        assert snapshots[0][0][-1]["content"] == "Remember this preference."
        assert snapshots[0][1:] == (True, False)
        state = db.get_realtime_session_state("voice-recovery")
        assert state["review_state"] == "completed"
        assert state["review_boundary_message_id"] == boundary
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_idle_prune_does_not_end_an_active_turn(db):
    _persist_call(db)
    manager, _sidebands = _manager(db)
    session = await manager.require_active("voice-recovery")
    session.last_activity_at = time.time() - manager.config.idle_timeout_seconds - 1
    session._turn = object()
    try:
        await manager.prune()
        assert manager.get("voice-recovery") is session

        session._turn = None
        await manager.prune()
        assert manager.get("voice-recovery") is None
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_review_drain_can_restart_after_database_claim_failure(db):
    _persist_call(db)
    manager, _sidebands = _manager(db)
    session = await manager.require_active("voice-recovery")
    boundary = db.append_message(
        "voice-recovery",
        role="user",
        content="Review this durable turn.",
    )
    snapshot = [{"role": "user", "content": "Review this durable turn."}]
    try:
        with patch.object(
            db,
            "mark_realtime_review_running",
            side_effect=RuntimeError("temporary database failure"),
        ):
            session._schedule_background_review_sync(
                messages_snapshot=snapshot,
                review_memory=True,
            )
            failed_thread = session._review_thread
            assert failed_thread is not None
            await asyncio.to_thread(failed_thread.join, 2)

        assert session._review_running is False
        assert session._review_pending is not None

        with patch(
            "agent.background_review.spawn_background_review_thread",
            return_value=(lambda: True, "review"),
        ):
            session._schedule_background_review_sync(
                messages_snapshot=snapshot,
                review_memory=True,
            )
            retry_thread = session._review_thread
            assert retry_thread is not None
            await asyncio.to_thread(retry_thread.join, 2)

        assert retry_thread.is_alive() is False
        assert session._review_running is False
        state = db.get_realtime_session_state("voice-recovery")
        assert state["review_state"] == "completed"
        assert state["review_boundary_message_id"] == boundary
    finally:
        await manager.close_all()
