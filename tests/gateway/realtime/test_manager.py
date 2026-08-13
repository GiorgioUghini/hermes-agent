import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.realtime.manager import RealtimeSessionError, RealtimeSessionManager
from gateway.realtime.protocol import (
    RealtimeCall,
    RealtimeProtocolError,
    RealtimeVoiceConfig,
    validate_device_advertisement,
)
from hermes_state import SessionDB


DEVICE_ID = "22222222-2222-4222-8222-222222222222"


class _FakeAgent:
    def __init__(self, db, session_id, tools=None):
        self._session_db = db
        self.session_id = session_id
        self.tools = tools or []
        self._cached_system_prompt = ""
        self.ephemeral_system_prompt = None
        self.iteration_budget = SimpleNamespace(consume=lambda: True)
        self._skill_nudge_interval = 0
        self.valid_tool_names = set()

    def _ensure_db_session(self):
        return None

    def _build_system_prompt(self, _system_message):
        return "fresh frozen prompt"


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


class _BlockingCallClient:
    def __init__(self, call):
        self.call = call
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def create_call(self, _offer_sdp, _session_config):
        self.started.set()
        await self.release.wait()
        return self.call


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
    config = RealtimeVoiceConfig.from_config({
        "realtime_voice": {
            "enabled": True,
            "transport": {
                "call_url": "https://proxy.example/v1/realtime/calls",
                "sideband_url": "wss://proxy.example/v1/realtime",
            },
        }
    })
    manager = RealtimeSessionManager(
        config=config,
        api_key="proxy-key",
        agent_factory=lambda session_id: _FakeAgent(db, session_id),
    )

    assert manager._call_client._call_url == config.call_url
    sideband = manager._sideband_factory("call_proxy", lambda _event: None)
    assert sideband._websocket_url == config.sideband_url


def _persist_call(db, *, state="ready", started_at=None, frozen_tools=None):
    if frozen_tools is None:
        frozen_tools = [
            {
                "type": "function",
                "function": {
                    "name": "frozen_tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    db.save_realtime_session_state(
        "voice-recovery",
        provider_call_id="call_original",
        provider_call_started_at=started_at or time.time(),
        state=state,
        model="gpt-realtime-snapshot",
        voice="cedar",
        frozen_instructions="original frozen prompt",
        frozen_tools=frozen_tools,
    )


def _device(*capabilities):
    return validate_device_advertisement({
        "protocol_version": 1,
        "instance_id": DEVICE_ID,
        "capabilities": [
            {"name": name, "version": version}
            for name, version in capabilities
        ],
    })


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
async def test_create_without_device_metadata_preserves_base_tools(db):
    call_client = _FakeCallClient(
        RealtimeCall(answer_sdp="answer", call_id="call_without_device")
    )
    manager, _sidebands = _manager(db, call_client=call_client)
    try:
        created = await manager.create_session(
            "offer",
            requested_session_id="voice-recovery",
        )

        assert created.session.device_response() is None
        assert [tool["name"] for tool in call_client.requests[0][1]["tools"]] == [
            "current_tool"
        ]
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_negotiation_adds_only_server_catalog_timer_tools(db):
    call_client = _FakeCallClient(
        RealtimeCall(answer_sdp="answer", call_id="call_with_device")
    )
    manager, _sidebands = _manager(db, call_client=call_client)
    try:
        created = await manager.create_session(
            "offer",
            requested_session_id="voice-recovery",
            device=_device(("timer", 1), ("unsupported_future", 9)),
        )

        tool_names = [
            tool["name"] for tool in call_client.requests[0][1]["tools"]
        ]
        assert tool_names == [
            "current_tool",
            "device_timer_create",
            "device_timer_list",
            "device_timer_cancel",
        ]
        assert created.session.device_response() == {
            "protocol_version": 1,
            "instance_id": DEVICE_ID,
            "capabilities": [{"name": "timer", "version": 1}],
        }
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_device_tools_require_durable_session_storage():
    call_client = _FakeCallClient(
        RealtimeCall(answer_sdp="answer", call_id="call-should-not-start")
    )
    manager = RealtimeSessionManager(
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        api_key="sk-test",
        agent_factory=lambda session_id: _FakeAgent(None, session_id),
        call_client=call_client,
        sideband_factory=lambda *_args: _FakeSideband("unused"),
    )

    with pytest.raises(RealtimeSessionError) as exc:
        await manager.create_session(
            "offer",
            requested_session_id="device-no-storage",
            device=_device(("timer", 1)),
        )

    assert exc.value.code == "device_persistence_unavailable"
    assert call_client.requests == []


@pytest.mark.asyncio
async def test_renewal_without_device_advertisement_removes_device_tools(db):
    call_client = _FakeCallClient(
        RealtimeCall(answer_sdp="answer-1", call_id="call_with_device"),
        RealtimeCall(answer_sdp="answer-2", call_id="call_without_device"),
    )
    manager, _sidebands = _manager(db, call_client=call_client)
    try:
        created = await manager.create_session(
            "offer-1",
            requested_session_id="voice-recovery",
            device=_device(("timer", 1)),
        )
        assert created.session.device_response() is not None

        renewed = await manager.renew_session("voice-recovery", "offer-2")

        assert renewed.session.device_response() is None
        assert [tool["name"] for tool in call_client.requests[1][1]["tools"]] == [
            "current_tool"
        ]
        persisted = db.get_realtime_session_state("voice-recovery")
        assert persisted["device_context"] is None
        assert [
            tool["function"]["name"] for tool in persisted["frozen_base_tools"]
        ] == ["current_tool"]
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_restart_recovers_dispatched_timer_as_outcome_unknown(db):
    _persist_call(db, state="turn_active")
    db.append_message("voice-recovery", role="user", content="Set a timer.")
    db.append_message(
        "voice-recovery",
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "server-call-before-device",
                "type": "function",
                "function": {
                    "name": "frozen_tool",
                    "arguments": "{}",
                },
            },
            {
                "id": "device-call-crashed",
                "type": "function",
                "function": {
                    "name": "device_timer_create",
                    "arguments": '{"duration_seconds":300}',
                },
            },
            {
                "id": "server-call-after-device",
                "type": "function",
                "function": {
                    "name": "frozen_tool",
                    "arguments": "{}",
                },
            },
        ],
    )
    request = {
        "invocation_id": "device-call-crashed",
        "device_instance_id": DEVICE_ID,
        "tool_name": "device_timer_create",
        "capability": "timer",
        "operation": "create",
        "arguments": {"duration_seconds": 300, "label": None},
        "expires_at_epoch_ms": 1_786_450_200_000,
    }
    assert db.create_realtime_device_invocation(
        "voice-recovery",
        invocation_id="device-call-crashed",
        tool_name="device_timer_create",
        side_effecting=True,
        request_data=request,
        expires_at_epoch_ms=request["expires_at_epoch_ms"],
    )
    assert db.mark_realtime_device_invocation_dispatch(
        "voice-recovery",
        "device-call-crashed",
        dispatched=False,
    )
    assert db.mark_realtime_device_invocation_dispatch(
        "voice-recovery",
        "device-call-crashed",
        dispatched=True,
    )
    call_client = _FakeCallClient(
        RealtimeCall(answer_sdp="answer", call_id="call_after_crash"),
        RealtimeCall(
            answer_sdp="continued-answer",
            call_id="call_after_rejection",
        ),
        RealtimeCall(
            answer_sdp="recovered-answer",
            call_id="call_after_accepted_crash",
        ),
    )
    manager, sidebands = _manager(db, call_client=call_client)
    try:
        with pytest.raises(RealtimeSessionError) as recovery:
            await manager.require_active("voice-recovery")
        assert recovery.value.code == "session_renewal_required"

        renewed = await manager.renew_session("voice-recovery", "offer")

        durable = db.get_tool_result_by_call_id(
            "voice-recovery",
            "device-call-crashed",
        )
        assert durable is not None
        assert '"code":"outcome_unknown"' in durable["content"]
        assert durable["effect_disposition"] == "unknown"
        skipped_server = db.get_tool_result_by_call_id(
            "voice-recovery",
            "server-call-after-device",
        )
        assert skipped_server is not None
        assert "Tool execution skipped" in skipped_server["content"]
        assert skipped_server["effect_disposition"] == "none"
        uncertain_server = db.get_tool_result_by_call_id(
            "voice-recovery",
            "server-call-before-device",
        )
        assert uncertain_server is not None
        assert '"code":"outcome_unknown"' in uncertain_server["content"]
        assert uncertain_server["effect_disposition"] == "unknown"
        invocation = db.get_realtime_device_invocation(
            "voice-recovery",
            "device-call-crashed",
        )
        assert invocation["status"] == "completed"
        assert invocation["result_source"] == "recovery"
        assert invocation["continuation_required"] is True
        output = next(
            event
            for event in sidebands[0].sent
            if event.get("item", {}).get("type") == "function_call_output"
            and event["item"].get("call_id") == "device-call-crashed"
        )
        assert output["item"]["call_id"] == "device-call-crashed"
        assert "outcome_unknown" in output["item"]["output"]
        projected_call_ids = [
            event["item"]["call_id"]
            for event in sidebands[0].sent
            if event.get("item", {}).get("type") == "function_call_output"
        ]
        assert projected_call_ids == [
            "server-call-before-device",
            "device-call-crashed",
            "server-call-after-device",
        ]
        output_index = sidebands[0].sent.index(output)
        continuation_index = next(
            index
            for index, event in enumerate(sidebands[0].sent)
            if index > output_index and event.get("type") == "response.create"
        )
        assert output_index < continuation_index
        assert renewed.session._turn is not None
        continuation = sidebands[0].sent[continuation_index]
        await renewed.session.handle_provider_event({
            "type": "error",
            "error": {
                "event_id": continuation["event_id"],
                "code": "temporary_provider_error",
                "message": "Retry the continuation.",
            },
        })
        assert db.get_realtime_device_invocation(
            "voice-recovery",
            "device-call-crashed",
        )["continuation_required"] is True
        assert renewed.session.can_renew
        await renewed.session.handle_provider_event({
            "type": "response.created",
            "response": {"id": "unrelated-response"},
        })
        assert db.get_realtime_device_invocation(
            "voice-recovery",
            "device-call-crashed",
        )["continuation_required"] is True
        continued = await manager.renew_session(
            "voice-recovery",
            "offer-recovery-continued",
        )
        assert len(sidebands) == 2
        renewed_continuations = [
            event
            for event in sidebands[1].sent
            if event.get("type") == "response.create"
            and not event["response"].get("metadata")
        ]
        assert len(renewed_continuations) == 1
        await continued.session.handle_provider_event({
            "type": "response.created",
            "response": {"id": "accepted-recovery"},
        })
        assert db.get_realtime_device_invocation(
            "voice-recovery",
            "device-call-crashed",
        )["continuation_required"] is False
        assert db.get_realtime_session_state("voice-recovery")["state"] == (
            "turn_active"
        )
        await continued.session.close(
            reason="crash_after_continuation_acceptance",
            end_session=False,
        )
        recovered = await manager.renew_session(
            "voice-recovery",
            "offer-after-accepted-crash",
        )
        assert len(sidebands) == 3
        assert db.get_realtime_device_invocation(
            "voice-recovery",
            "device-call-crashed",
        )["continuation_required"] is True
        recovered_call_ids = [
            event["item"]["call_id"]
            for event in sidebands[2].sent
            if event.get("item", {}).get("type") == "function_call_output"
        ]
        assert recovered_call_ids == [
            "server-call-before-device",
            "device-call-crashed",
            "server-call-after-device",
        ]
        assert any(
            event.get("type") == "response.create"
            and not event["response"].get("metadata")
            for event in sidebands[2].sent
        )
        await recovered.session.handle_provider_event({
            "type": "response.created",
            "response": {"id": "accepted-after-crash"},
        })
        assert db.get_realtime_device_invocation(
            "voice-recovery",
            "device-call-crashed",
        )["continuation_required"] is False
        assert not any(
            event["type"] == "device.action.request"
            for event in recovered.session.broker.snapshot()
        )
        with pytest.raises(RealtimeProtocolError) as late_result:
            await recovered.session._handle_device_action_result({
                "invocation_id": "device-call-crashed",
                "status": "ok",
            })
        assert late_result.value.code == "expired_device_invocation"
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_restart_preserves_device_result_captured_before_ack(db):
    _persist_call(db, state="renewal_required")
    db.append_message("voice-recovery", role="user", content="Set a timer.")
    db.append_message(
        "voice-recovery",
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "device-call-result-captured",
                "type": "function",
                "function": {
                    "name": "device_timer_create",
                    "arguments": '{"duration_seconds":300}',
                },
            }
        ],
    )
    request = {
        "invocation_id": "device-call-result-captured",
        "device_instance_id": DEVICE_ID,
        "tool_name": "device_timer_create",
        "capability": "timer",
        "operation": "create",
        "arguments": {"duration_seconds": 300, "label": None},
        "expires_at_epoch_ms": 1_786_450_200_000,
    }
    assert db.create_realtime_device_invocation(
        "voice-recovery",
        invocation_id="device-call-result-captured",
        tool_name="device_timer_create",
        side_effecting=True,
        request_data=request,
        expires_at_epoch_ms=request["expires_at_epoch_ms"],
    )
    assert db.mark_realtime_device_invocation_dispatch(
        "voice-recovery",
        "device-call-result-captured",
        dispatched=False,
    )
    assert db.mark_realtime_device_invocation_dispatch(
        "voice-recovery",
        "device-call-result-captured",
        dispatched=True,
    )
    output = '{"success":true,"result":{"timer_id":"timer-42"}}'
    assert db.record_realtime_device_invocation_result(
        "voice-recovery",
        "device-call-result-captured",
        output=output,
        effect_disposition=None,
        result_source="device",
    )
    call_client = _FakeCallClient(
        RealtimeCall(answer_sdp="answer", call_id="call_after_crash")
    )
    manager, sidebands = _manager(db, call_client=call_client)
    try:
        renewed = await manager.renew_session("voice-recovery", "offer")

        durable = db.get_tool_result_by_call_id(
            "voice-recovery",
            "device-call-result-captured",
        )
        assert durable["content"] == output
        invocation = db.get_realtime_device_invocation(
            "voice-recovery",
            "device-call-result-captured",
        )
        assert invocation["status"] == "completed"
        assert invocation["result_source"] == "device"
        projected = next(
            event
            for event in sidebands[0].sent
            if event.get("item", {}).get("type") == "function_call_output"
            and event["item"].get("call_id") == "device-call-result-captured"
        )
        assert projected["item"]["output"] == output
        assert any(
            event.get("type") == "response.create"
            for event in sidebands[0].sent[
                sidebands[0].sent.index(projected) + 1 :
            ]
        )
        assert renewed.session._turn is not None
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_restart_reconciles_device_call_not_yet_dispatched(db):
    _persist_call(db, state="renewal_required")
    db.append_message("voice-recovery", role="user", content="Set a timer.")
    db.append_message(
        "voice-recovery",
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "device-call-not-dispatched",
                "type": "function",
                "function": {
                    "name": "device_timer_create",
                    "arguments": '{"duration_seconds":300}',
                },
            }
        ],
    )
    request = {
        "invocation_id": "device-call-not-dispatched",
        "device_instance_id": DEVICE_ID,
        "tool_name": "device_timer_create",
        "capability": "timer",
        "operation": "create",
        "arguments": {"duration_seconds": 300, "label": None},
        "expires_at_epoch_ms": 1_786_450_200_000,
    }
    assert db.create_realtime_device_invocation(
        "voice-recovery",
        invocation_id="device-call-not-dispatched",
        tool_name="device_timer_create",
        side_effecting=True,
        request_data=request,
        expires_at_epoch_ms=request["expires_at_epoch_ms"],
    )
    call_client = _FakeCallClient(
        RealtimeCall(answer_sdp="answer", call_id="call_after_crash")
    )
    manager, sidebands = _manager(db, call_client=call_client)
    try:
        renewed = await manager.renew_session("voice-recovery", "offer")

        durable = db.get_tool_result_by_call_id(
            "voice-recovery",
            "device-call-not-dispatched",
        )
        assert durable is not None
        assert '"code":"device_action_not_dispatched"' in durable["content"]
        assert durable["effect_disposition"] == "none"
        invocation = db.get_realtime_device_invocation(
            "voice-recovery",
            "device-call-not-dispatched",
        )
        assert invocation["status"] == "completed"
        assert invocation["result_source"] == "recovery"
        output = next(
            event
            for event in sidebands[0].sent
            if (
                event.get("item", {}).get("type") == "function_call_output"
                and event["item"].get("call_id")
                == "device-call-not-dispatched"
            )
        )
        assert "device_action_not_dispatched" in output["item"]["output"]
        output_index = sidebands[0].sent.index(output)
        assert any(
            event.get("type") == "response.create"
            for event in sidebands[0].sent[output_index + 1 :]
        )
        assert renewed.session._turn is not None
        assert not any(
            event["type"] == "device.action.request"
            for event in renewed.session.broker.snapshot()
        )
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
async def test_persisted_renewal_preserves_empty_frozen_tool_snapshot(db):
    _persist_call(db, state="suspended", frozen_tools=[])
    call_client = _FakeCallClient(
        RealtimeCall(answer_sdp="answer", call_id="call_without_tools")
    )
    manager, _sidebands = _manager(db, call_client=call_client)
    try:
        created = await manager.renew_session("voice-recovery", "offer")

        assert created.session.frozen_tools == []
        assert call_client.requests[0][1]["tools"] == []
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_suspend_closes_provider_call_and_later_renew_preserves_session(db):
    _persist_call(db)
    call_client = _FakeCallClient(
        RealtimeCall(answer_sdp="answer", call_id="call_after_suspend")
    )
    manager, sidebands = _manager(db, call_client=call_client)
    try:
        session = await manager.require_active("voice-recovery")
        assert await manager.suspend_session("voice-recovery") is True

        assert manager.get("voice-recovery") is None
        assert session.closed is True
        assert sidebands[0].closed is True
        assert db.get_realtime_session_state("voice-recovery")["state"] == "suspended"

        renewed = await manager.renew_session("voice-recovery", "new-offer")
        assert renewed.call_id == "call_after_suspend"
        assert renewed.session.frozen_instructions == "original frozen prompt"
        assert renewed.session.config.voice == "cedar"
        assert db.get_realtime_session_state("voice-recovery")["state"] == "ready"
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_suspend_rejects_live_session_before_turn_settles(db):
    _persist_call(db)
    manager, _sidebands = _manager(db)
    session = await manager.require_active("voice-recovery")
    session._turn = object()
    try:
        with pytest.raises(RealtimeSessionError) as caught:
            await manager.suspend_session("voice-recovery")
        assert caught.value.code == "session_busy"
        assert manager.get("voice-recovery") is session
    finally:
        session._turn = None
        await manager.close_all()


@pytest.mark.asyncio
async def test_suspend_serializes_with_in_progress_provider_renewal(db):
    _persist_call(db)
    call_client = _BlockingCallClient(
        RealtimeCall(answer_sdp="answer", call_id="call_racing_renew")
    )
    manager, sidebands = _manager(db, call_client=call_client)
    try:
        await manager.require_active("voice-recovery")
        renew_task = asyncio.create_task(
            manager.renew_session("voice-recovery", "new-offer")
        )
        await call_client.started.wait()
        suspend_task = asyncio.create_task(manager.suspend_session("voice-recovery"))
        await asyncio.sleep(0)
        assert suspend_task.done() is False

        call_client.release.set()
        renewed = await renew_task
        assert renewed.call_id == "call_racing_renew"
        assert await suspend_task is True

        assert manager.get("voice-recovery") is None
        assert db.get_realtime_session_state("voice-recovery")["state"] == "suspended"
        assert all(sideband.closed for sideband in sidebands)
    finally:
        call_client.release.set()
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
