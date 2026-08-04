import asyncio
import base64
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.realtime.openai_sideband import OpenAIRealtimeError
from gateway.realtime.protocol import RealtimeProtocolError, RealtimeVoiceConfig
from gateway.realtime.session import RealtimeVoiceSession, prepare_realtime_agent
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from run_agent import AIAgent


async def _wait_until(predicate, *, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Timed out waiting for realtime session state")
        await asyncio.sleep(0.01)


class _FakeSideband:
    def __init__(self):
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


class _ResponsivePrerollSideband(_FakeSideband):
    def __init__(self):
        super().__init__()
        self.handler = None

    async def send(self, event):
        await super().send(event)
        if self.handler is None:
            return
        if event["type"] == "session.update":
            await self.handler({"type": "session.updated", "session": event["session"]})
        elif event["type"] == "input_audio_buffer.clear":
            await self.handler({"type": "input_audio_buffer.cleared"})
        elif event["type"] == "input_audio_buffer.commit":
            await self.handler({
                "type": "input_audio_buffer.committed",
                "item_id": "input_preroll",
            })


class _DeferredRestoreAckSideband(_ResponsivePrerollSideband):
    def __init__(self):
        super().__init__()
        self.update_count = 0
        self.restore_sent = asyncio.Event()
        self.restore_event = None

    async def send(self, event):
        if event["type"] == "session.update":
            self.update_count += 1
            if self.update_count == 2:
                await _FakeSideband.send(self, event)
                self.restore_event = event
                self.restore_sent.set()
                return
        await super().send(event)


class _FailingCommitPrerollSideband(_ResponsivePrerollSideband):
    async def send(self, event):
        if event["type"] == "input_audio_buffer.commit":
            await _FakeSideband.send(self, event)
            raise OpenAIRealtimeError(
                "provider commit failed",
                code="commit_failed",
                status=502,
            )
        await super().send(event)


class _FailingInitialClearSideband(_ResponsivePrerollSideband):
    def __init__(self):
        super().__init__()
        self.clear_count = 0

    async def send(self, event):
        if event["type"] == "input_audio_buffer.clear":
            self.clear_count += 1
            if self.clear_count == 1:
                await _FakeSideband.send(self, event)
                raise OpenAIRealtimeError(
                    "initial clear failed",
                    code="clear_failed",
                    status=502,
                )
        await super().send(event)


class _StalledPrerollSideband(_FakeSideband):
    async def send(self, event):
        self.sent.append(event)
        await asyncio.Event().wait()


class _RecordingRealtimeStateDB:
    def __init__(self):
        self.states = []

    def save_realtime_session_state(self, _session_id, **values):
        self.states.append(values["state"])


@pytest.mark.asyncio
async def test_preroll_commits_complete_utterance_once_and_restores_vad():
    sideband = _ResponsivePrerollSideband()
    session = RealtimeVoiceSession(
        session_id="preroll",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_preroll",
    )
    sideband.handler = session.handle_provider_event
    session.state = "ready"
    audio = b"\x01\x00" * 7200

    try:
        first = await session.ingest_preroll_audio(
            audio,
            idempotency_key="wake:1",
        )
        repeated = await session.ingest_preroll_audio(
            audio,
            idempotency_key="wake:1",
        )

        assert first == repeated
        assert first["status"] == "committed"
        assert first["item_id"] == "input_preroll"
        assert first["duration_ms"] == 300
        assert [event["type"] for event in sideband.sent].count(
            "input_audio_buffer.commit"
        ) == 1
        appended = b"".join(
            base64.b64decode(event["audio"])
            for event in sideband.sent
            if event["type"] == "input_audio_buffer.append"
        )
        assert appended == audio
        updates = [
            event for event in sideband.sent if event["type"] == "session.update"
        ]
        assert updates[0]["session"]["audio"]["input"]["turn_detection"] is None
        assert (
            updates[-1]["session"]["audio"]["input"]["turn_detection"]
            == session.config.turn_detection_config()
        )
        assert session.can_renew is False

        with pytest.raises(RealtimeProtocolError) as conflict:
            await session.ingest_preroll_audio(
                b"\x02\x00" * 7200,
                idempotency_key="wake:1",
            )
        assert conflict.value.code == "preroll_idempotency_conflict"
    finally:
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_preroll_ignores_stale_vad_acknowledgment():
    sideband = _DeferredRestoreAckSideband()
    session = RealtimeVoiceSession(
        session_id="preroll-stale-ack",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_preroll_stale_ack",
    )
    sideband.handler = session.handle_provider_event
    session.state = "ready"
    upload = asyncio.create_task(
        session.ingest_preroll_audio(
            b"\x01\x00" * 7200,
            idempotency_key="wake:stale-ack",
        )
    )
    try:
        await asyncio.wait_for(sideband.restore_sent.wait(), timeout=1)
        await session.handle_provider_event({
            "type": "session.updated",
            "session": {"audio": {"input": {"turn_detection": None}}},
        })
        await asyncio.sleep(0)
        assert upload.done() is False

        await session.handle_provider_event({
            "type": "session.updated",
            "session": sideband.restore_event["session"],
        })
        result = await upload
        assert result["status"] == "committed"
    finally:
        if not upload.done():
            upload.cancel()
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_preroll_failure_clears_audio_and_requires_sticky_renewal():
    sideband = _FailingCommitPrerollSideband()
    db = _RecordingRealtimeStateDB()
    session = RealtimeVoiceSession(
        session_id="preroll-cleanup",
        agent=SimpleNamespace(tools=[], _session_db=db),
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_preroll_cleanup",
    )
    sideband.handler = session.handle_provider_event
    session.state = "ready"
    audio = b"\x01\x00" * 7200

    with pytest.raises(OpenAIRealtimeError, match="provider commit failed"):
        await session.ingest_preroll_audio(
            audio,
            idempotency_key="wake:cleanup",
        )
    await asyncio.sleep(0)

    event_types = [event["type"] for event in sideband.sent]
    commit_index = event_types.index("input_audio_buffer.commit")
    cleanup_clear_index = event_types.index(
        "input_audio_buffer.clear",
        commit_index + 1,
    )
    restore_index = event_types.index("session.update", cleanup_clear_index + 1)
    assert commit_index < cleanup_clear_index < restore_index
    assert session._renewal_required is True
    assert session.state == "degraded"
    assert session.can_renew is True
    assert db.states[-1] == "renewal_required"

    record = session._preroll_requests["wake:cleanup"]
    assert record.task is None
    assert record.failure is not None
    sent_count = len(sideband.sent)
    with pytest.raises(OpenAIRealtimeError, match="provider commit failed"):
        await session.ingest_preroll_audio(
            audio,
            idempotency_key="wake:cleanup",
        )
    assert len(sideband.sent) == sent_count

    await session._persist_runtime_state("ready")
    assert db.states[-1] == "renewal_required"
    await session.close(reason="test_complete", end_session=False)
    assert db.states[-1] == "renewal_required"


@pytest.mark.asyncio
async def test_preroll_initial_clear_failure_retries_clear_before_vad_restore():
    sideband = _FailingInitialClearSideband()
    session = RealtimeVoiceSession(
        session_id="preroll-initial-clear",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_preroll_initial_clear",
    )
    sideband.handler = session.handle_provider_event
    session.state = "ready"

    with pytest.raises(OpenAIRealtimeError, match="initial clear failed"):
        await session.ingest_preroll_audio(
            b"\x01\x00" * 7200,
            idempotency_key="wake:initial-clear",
        )

    event_types = [event["type"] for event in sideband.sent]
    first_clear = event_types.index("input_audio_buffer.clear")
    cleanup_clear = event_types.index("input_audio_buffer.clear", first_clear + 1)
    restore = event_types.index("session.update", cleanup_clear + 1)
    assert first_clear < cleanup_clear < restore
    assert session._renewal_required is True
    await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_preroll_stalled_send_honors_operation_timeout():
    sideband = _StalledPrerollSideband()
    session = RealtimeVoiceSession(
        session_id="preroll-send-timeout",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
            preroll_timeout_seconds=0.05,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_preroll_send_timeout",
    )
    session.state = "ready"

    with pytest.raises(RealtimeProtocolError) as timeout:
        await session.ingest_preroll_audio(
            b"\x01\x00" * 7200,
            idempotency_key="wake:send-timeout",
        )
    assert timeout.value.code == "preroll_provider_timeout"
    assert session._preroll_active is False
    assert session._renewal_required is True
    assert session.can_renew is True
    await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_preroll_blocks_rotation_and_parallel_upload_until_teardown():
    sideband = _FakeSideband()
    db = _RecordingRealtimeStateDB()
    session = RealtimeVoiceSession(
        session_id="preroll-blocked",
        agent=SimpleNamespace(tools=[], _session_db=db),
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_preroll_blocked",
    )
    session.state = "ready"
    upload = asyncio.create_task(
        session.ingest_preroll_audio(
            b"\x01\x00" * 7200,
            idempotency_key="wake:blocked",
        )
    )
    await _wait_until(lambda: session._preroll_active)
    try:
        assert session.can_renew is False
        assert session.idle_timeout_eligible is False
        with pytest.raises(RealtimeProtocolError) as concurrent:
            await session.ingest_preroll_audio(
                b"\x01\x00" * 7200,
                idempotency_key="wake:other",
            )
        assert concurrent.value.code == "preroll_in_progress"
    finally:
        await session.close(reason="test_complete", end_session=False)
    with pytest.raises(asyncio.CancelledError):
        await upload
    assert db.states[-1] == "renewal_required"


@pytest.mark.asyncio
async def test_active_provider_input_blocks_suspend_and_renew():
    session = RealtimeVoiceSession(
        session_id="active-input",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(
            enabled=True,
            provider_call_max_seconds=3600,
            transcription_timeout_seconds=30,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_active_input",
    )
    session.state = "ready"

    await session.handle_provider_event({"type": "input_audio_buffer.speech_started"})
    await session.handle_provider_event({
        "type": "input_audio_buffer.committed",
        "item_id": "input_pending",
    })

    assert session.can_suspend is False
    assert session.can_renew is False
    with pytest.raises(RealtimeProtocolError) as busy:
        await session.ingest_preroll_audio(
            b"\x01\x00" * 7200,
            idempotency_key="wake:while-input-active",
        )
    assert busy.value.code == "session_busy"
    await session.close(reason="test_complete", end_session=False)


def test_new_tool_batch_clears_stale_skip_without_losing_active_interrupt():
    observed = []
    agent = SimpleNamespace(
        tools=[],
        _session_db=None,
        _skip_unstarted_tool_calls=True,
    )

    def execute(*_args):
        observed.append(agent._skip_unstarted_tool_calls)

    agent._execute_tool_calls_sequential = execute
    session = RealtimeVoiceSession(
        session_id="stale-tool-skip",
        agent=agent,
        config=RealtimeVoiceConfig(enabled=True),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_stale_tool_skip",
    )
    assistant = SimpleNamespace(content="", tool_calls=[])

    session._execute_tools_sync(assistant, [], "task", 1, skip_unstarted=False)
    session._client_interrupt_pending = True
    session._execute_tools_sync(assistant, [], "task", 2, skip_unstarted=False)

    assert observed == [False, True]


@pytest.mark.asyncio
async def test_status_interruption_with_pending_continuation_starts_handoff_timeout():
    sideband = _FakeSideband()
    agent = SimpleNamespace(
        tools=[],
        _session_db=None,
        _skip_unstarted_tool_calls=False,
    )
    session = RealtimeVoiceSession(
        session_id="status-interrupt",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            preroll_timeout_seconds=0.01,
            transcription_timeout_seconds=0.01,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_status_interrupt",
    )
    session.state = "tool_wait"
    session._turn = SimpleNamespace(messages=[])
    session._continuation_pending = True
    session._status_response_active = True
    session._active_status_response_id = "status_1"

    await session.interrupt_response(
        request_id="wake-status",
        audio_end_ms=0,
    )

    assert session._interrupt_handoff_task is not None
    await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_recovered_call_rotation_uses_remaining_provider_lifetime():
    session = RealtimeVoiceSession(
        session_id="rotation-age",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(
            enabled=True,
            provider_call_max_seconds=120,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_rotation",
        call_started_at=1000,
    )

    with (
        patch("gateway.realtime.session.time.time", return_value=1110),
        patch(
            "gateway.realtime.session.asyncio.sleep", new_callable=AsyncMock
        ) as sleep,
    ):
        await session._rotation_watch()

    sleep.assert_awaited_once_with(0.0)
    event = session.broker.snapshot()[-1]
    assert event["type"] == "session.rotation_required"
    assert event["data"]["renew_within_seconds"] == 10


@pytest.mark.asyncio
async def test_real_skill_view_pipeline_persists_before_function_output(tmp_path):
    skill_dir = get_hermes_home() / "skills" / "voice-research"
    skill_dir.mkdir(parents=True)
    marker = "VOICE_SKILL_MARKER"
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: voice-research\n"
        "description: Reads a realtime research playbook.\n"
        "version: 1.0.0\n"
        "author: Test\n"
        "---\n\n"
        "# Voice Research Skill\n\n"
        f"{marker}\n",
        encoding="utf-8",
    )
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["skills"],
        session_id="voice-session",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    frozen_prompt = prepare_realtime_agent(agent, [])
    frozen_tools = json.dumps(agent.tools, sort_keys=True)
    sideband = _FakeSideband()
    config = RealtimeVoiceConfig(
        enabled=True,
        intermediate_speech_enabled=False,
        provider_call_max_seconds=3600,
    )
    session = RealtimeVoiceSession(
        session_id="voice-session",
        agent=agent,
        config=config,
        frozen_instructions=frozen_prompt,
        conversation_history=[],
        sideband=sideband,
        call_id="call_1",
    )
    await session.start()
    try:
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "Use my voice research skill.",
        })
        assert agent._cached_system_prompt == frozen_prompt
        assert json.dumps(agent.tools, sort_keys=True) == frozen_tools
        assert sideband.sent[-1]["type"] == "response.create"

        await session.handle_provider_event({
            "type": "response.done",
            "response": {
                "id": "response_1",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "item_1",
                        "call_id": "tool_call_1",
                        "name": "skill_view",
                        "arguments": '{"name":"voice-research"}',
                    }
                ],
            },
        })
        await session._tool_task

        durable = db.get_tool_result_by_call_id("voice-session", "tool_call_1")
        assert durable is not None
        assert marker in durable["content"]
        output_events = [
            event
            for event in sideband.sent
            if event.get("type") == "conversation.item.create"
            and event.get("item", {}).get("type") == "function_call_output"
        ]
        assert marker in output_events[-1]["item"]["output"]
        messages = db.get_messages("voice-session")
        roles = [message["role"] for message in messages]
        assert roles[:3] == ["user", "assistant", "tool"]
        assert messages[1]["tool_calls"][0]["id"] == "tool_call_1"

        await session.handle_provider_event({
            "type": "response.done",
            "response": {
                "id": "response_2",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_audio",
                                "transcript": "I used the saved playbook.",
                            }
                        ],
                    }
                ],
            },
        })

        final_messages = db.get_messages("voice-session")
        assert final_messages[-1]["role"] == "assistant"
        assert final_messages[-1]["content"] == "I used the saved playbook."
        assert session.state == "ready"
    finally:
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_duplicate_provider_response_does_not_repeat_tool_side_effect(
    tmp_path,
):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["todo"],
        session_id="dedupe-session",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    frozen_prompt = prepare_realtime_agent(agent, [])
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="dedupe-session",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions=frozen_prompt,
        conversation_history=[],
        sideband=sideband,
        call_id="call_dedupe",
    )
    await session.start()
    try:
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "Track one item.",
        })
        event = {
            "type": "response.done",
            "response": {
                "id": "response_tool",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "item_todo",
                        "call_id": "call_todo",
                        "name": "todo",
                        "arguments": (
                            '{"todos":[{"id":"one","content":"one",'
                            '"status":"pending"}]}'
                        ),
                    }
                ],
            },
        }
        await session.handle_provider_event(event)
        await session._tool_task
        await session.handle_provider_event(event)

        tool_rows = [
            message
            for message in db.get_messages("dedupe-session")
            if message.get("tool_call_id") == "call_todo"
        ]
        assert len(tool_rows) == 1
    finally:
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_transcript_timeout_continues_once_and_ignores_late_transcript(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["todo"],
        session_id="transcript-timeout",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="transcript-timeout",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
            transcription_timeout_seconds=0.01,
        ),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=sideband,
        call_id="call_transcript",
    )
    await session.start()
    try:
        await session.handle_provider_event({
            "type": "input_audio_buffer.committed",
            "item_id": "input_slow",
        })
        await _wait_until(lambda: bool(session.sideband.sent))

        user_rows = [
            message
            for message in db.get_messages("transcript-timeout")
            if message["role"] == "user"
        ]
        assert len(user_rows) == 1
        assert "transcript unavailable" in user_rows[0]["content"].lower()
        assert session.sideband.sent[-1]["type"] == "response.create"

        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_slow",
            "transcript": "This arrived too late.",
        })
        user_rows = [
            message
            for message in db.get_messages("transcript-timeout")
            if message["role"] == "user"
        ]
        assert len(user_rows) == 1
        events = session.broker.subscribe(after_sequence=0).backlog
        assert any(
            event["type"] == "warning"
            and event["data"]["code"] == "transcription_timeout"
            for event in events
        )
    finally:
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_structured_interrupt_cancels_truncates_and_accepts_next_preroll(
    tmp_path,
):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["todo"],
        session_id="structured-barge-in",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    sideband = _ResponsivePrerollSideband()
    session = RealtimeVoiceSession(
        session_id="structured-barge-in",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=sideband,
        call_id="call_structured_barge_in",
    )
    sideband.handler = session.handle_provider_event

    await session.start()
    try:
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "Give me a long answer.",
        })
        await session.handle_provider_event({
            "type": "response.created",
            "response": {
                "id": "response_1",
                "status": "in_progress",
                "metadata": {},
            },
        })
        await session.handle_provider_event({
            "type": "response.output_item.added",
            "response_id": "response_1",
            "item": {
                "id": "assistant_1",
                "type": "message",
                "role": "assistant",
            },
        })

        first = await session.interrupt_response(
            request_id="wake-2",
            audio_end_ms=875,
        )
        repeated = await session.interrupt_response(
            request_id="wake-2",
            audio_end_ms=875,
        )
        assert first == repeated
        with pytest.raises(RealtimeProtocolError) as conflict:
            await session.interrupt_response(
                request_id="wake-2",
                audio_end_ms=900,
            )
        assert conflict.value.code == "interrupt_request_conflict"
        assert first["status"] == "accepted"
        assert first["truncation_requested"] is True
        assert [event["type"] for event in sideband.sent].count("response.cancel") == 1
        truncate = next(
            event
            for event in sideband.sent
            if event["type"] == "conversation.item.truncate"
        )
        assert truncate["item_id"] == "assistant_1"
        assert truncate["audio_end_ms"] == 875

        preroll = await session.ingest_preroll_audio(
            b"\x01\x00" * 7200,
            idempotency_key="wake:2",
        )
        assert preroll["status"] == "committed"
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_preroll",
            "transcript": "Actually, answer briefly.",
        })
        await session.handle_provider_event({
            "type": "response.output_audio_transcript.done",
            "response_id": "response_1",
            "transcript": "This tail was generated but not heard.",
        })
        await session.handle_provider_event({
            "type": "response.done",
            "response": {
                "id": "response_1",
                "status": "cancelled",
                "output": [],
            },
        })

        messages = db.get_messages("structured-barge-in")
        assert [message["role"] for message in messages[-3:]] == [
            "user",
            "assistant",
            "user",
        ]
        assert messages[-2]["content"] == "[Assistant response interrupted by user.]"
        assert messages[-2]["finish_reason"] == "interrupted"
        assert messages[-1]["content"] == "Actually, answer briefly."
        assert session.state == "responding"
    finally:
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_structured_interrupt_clears_active_webrtc_output_buffer():
    sideband = _FakeSideband()
    agent = SimpleNamespace(
        tools=[],
        _session_db=None,
        _skip_unstarted_tool_calls=False,
    )
    session = RealtimeVoiceSession(
        session_id="active-output-interrupt",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_active_output",
    )
    session.state = "responding"
    session._turn = SimpleNamespace(messages=[])
    session._active_response_id = "response_1"
    session._generation_active_response_ids.add("response_1")
    session._active_audio_item_id = "assistant_1"
    session._active_output_audio_response_id = "response_1"

    try:
        result = await session.interrupt_response(
            request_id="wake-active-output",
            audio_end_ms=420,
        )

        assert result["output_clear_requested"] is True
        assert result["truncation_requested"] is True
        assert [event["type"] for event in sideband.sent] == [
            "response.cancel",
            "output_audio_buffer.clear",
        ]
    finally:
        session._turn = None
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_late_playback_start_is_cleared_after_interrupt():
    sideband = _FakeSideband()
    agent = SimpleNamespace(
        tools=[],
        _session_db=None,
        _skip_unstarted_tool_calls=False,
    )
    session = RealtimeVoiceSession(
        session_id="late-output-start",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_late_output",
    )
    session.state = "responding"
    session._turn = SimpleNamespace(messages=[])
    session._active_response_id = "response_1"
    session._generation_active_response_ids.add("response_1")
    session._active_audio_item_id = "assistant_1"

    try:
        await session.interrupt_response(
            request_id="wake-before-playback",
            audio_end_ms=0,
        )
        await session.handle_provider_event({
            "type": "output_audio_buffer.started",
            "response_id": "response_1",
        })

        assert [event["type"] for event in sideband.sent] == [
            "response.cancel",
            "conversation.item.truncate",
            "output_audio_buffer.clear",
        ]
    finally:
        session._turn = None
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_completed_generation_waits_for_client_playback_before_turn_finalizes(
    tmp_path,
):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["todo"],
        session_id="playback-boundary",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="playback-boundary",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=sideband,
        call_id="call_playback",
    )

    await session.start()
    try:
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "Tell me something.",
        })
        await session.handle_provider_event({
            "type": "response.created",
            "response": {"id": "response_1", "status": "in_progress"},
        })
        await session.handle_provider_event({
            "type": "response.output_item.added",
            "response_id": "response_1",
            "item": {
                "id": "assistant_1",
                "type": "message",
                "role": "assistant",
            },
        })
        await session.handle_provider_event({
            "type": "output_audio_buffer.started",
            "response_id": "response_1",
        })
        await session.handle_provider_event({
            "type": "response.done",
            "response": {
                "id": "response_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_audio",
                                "transcript": "The complete spoken answer.",
                            }
                        ],
                    }
                ],
            },
        })

        assert session.state == "responding"
        assert session._turn is not None
        assert session._playback_finalize_task is None
        assert [row["role"] for row in db.get_messages("playback-boundary")] == ["user"]

        first = await session.complete_playback(
            request_id="played-1",
            response_id="response_1",
        )
        assert first["provider_output_drained"] is False
        assert session.state == "responding"

        await session.handle_provider_event({
            "type": "output_audio_buffer.stopped",
            "response_id": "response_1",
        })
        repeated = await session.complete_playback(
            request_id="played-1",
            response_id="response_1",
        )

        assert first == repeated
        assert session.state == "ready"
        messages = db.get_messages("playback-boundary")
        assert messages[-1]["content"] == "The complete spoken answer."
        events = session.broker.subscribe(after_sequence=0).backlog
        assert any(event["type"] == "response.generated" for event in events)
        assert any(event["type"] == "response.output_drained" for event in events)
        assert any(event["type"] == "turn.completed" for event in events)
    finally:
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_failed_response_waits_for_partial_audio_playback_boundary(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["todo"],
        session_id="failed-playback-boundary",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    session = RealtimeVoiceSession(
        session_id="failed-playback-boundary",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_failed_playback",
    )

    await session.start()
    try:
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "Start an answer.",
        })
        await session.handle_provider_event({
            "type": "response.created",
            "response": {"id": "response_1", "status": "in_progress"},
        })
        await session.handle_provider_event({
            "type": "response.output_item.added",
            "response_id": "response_1",
            "item": {
                "id": "assistant_1",
                "type": "message",
                "role": "assistant",
            },
        })
        await session.handle_provider_event({
            "type": "output_audio_buffer.started",
            "response_id": "response_1",
        })
        await session.handle_provider_event({
            "type": "response.done",
            "response": {
                "id": "response_1",
                "status": "failed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_audio",
                                "transcript": "This partial audio was rendered.",
                            }
                        ],
                    }
                ],
            },
        })

        assert session.state == "responding"
        assert session._turn is not None
        await session.complete_playback(
            request_id="played-failed",
            response_id="response_1",
        )
        await session.handle_provider_event({
            "type": "output_audio_buffer.stopped",
            "response_id": "response_1",
        })

        assert session.state == "ready"
        completed = [
            event
            for event in session.broker.subscribe(after_sequence=0).backlog
            if event["type"] == "turn.completed"
        ][-1]
        assert completed["data"]["failed"] is True
    finally:
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_turn_completion_requires_durable_sessiondb_flush(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["todo"],
        session_id="persistence-boundary",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    session = RealtimeVoiceSession(
        session_id="persistence-boundary",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_persistence_boundary",
    )

    await session.start()
    try:
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "Answer this.",
        })
        with (
            patch.object(agent, "_flush_messages_to_session_db", return_value=False),
            pytest.raises(RealtimeProtocolError) as persistence_error,
        ):
            await session.handle_provider_event({
                "type": "response.done",
                "response": {
                    "id": "response_1",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "text", "text": "The answer."}],
                        }
                    ],
                },
            })

        assert persistence_error.value.code == "turn_persistence_failed"
        assert session._turn is not None
        assert session.state == "degraded"
        assert session.can_suspend is False
        assert not any(
            event["type"] == "turn.completed" for event in session.broker.snapshot()
        )
    finally:
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_tool_batch_does_not_mask_turn_persistence_failure(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["todo"],
        session_id="tool-persistence-boundary",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    session = RealtimeVoiceSession(
        session_id="tool-persistence-boundary",
        agent=agent,
        config=RealtimeVoiceConfig(enabled=True),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_tool_persistence",
    )

    await session.start()
    try:
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "Run the tool.",
        })
        with patch.object(agent, "_flush_messages_to_session_db", return_value=False):
            await session._execute_tool_batch(
                [],
                response_id="response_tools",
                spoken_preamble="Interrupted preamble.",
                skip_unstarted=True,
            )

        assert session.state == "degraded"
        assert session._turn is not None
        assistants = [
            message
            for message in session._turn.messages
            if message.get("role") == "assistant"
        ]
        assert [message["content"] for message in assistants] == [
            "Interrupted preamble."
        ]
    finally:
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_review_enqueue_failure_does_not_block_durable_turn_completion(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["todo"],
        session_id="review-enqueue-failure",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    session = RealtimeVoiceSession(
        session_id="review-enqueue-failure",
        agent=agent,
        config=RealtimeVoiceConfig(enabled=True),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_review_enqueue_failure",
    )

    await session.start()
    try:
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "Remember this.",
        })
        assert session._turn is not None
        session._turn.should_review_memory = True

        def fail_review_enqueue(**_kwargs):
            raise RuntimeError("review queue unavailable")

        agent._background_review_dispatch = fail_review_enqueue
        await session.handle_provider_event({
            "type": "response.done",
            "response": {
                "id": "response_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "text", "text": "Remembered."}],
                    }
                ],
            },
        })

        assert session.state == "ready"
        assert session._turn is None
        events = session.broker.snapshot()
        assert any(event["type"] == "turn.completed" for event in events)
        assert any(
            event["type"] == "warning"
            and event["data"]["code"] == "background_review_enqueue_failed"
            for event in events
        )
    finally:
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_interrupt_after_generation_done_truncates_unplayed_tail(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["todo"],
        session_id="late-barge-in",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="late-barge-in",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=sideband,
        call_id="call_late_barge_in",
    )

    await session.start()
    try:
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "Give me a long answer.",
        })
        await session.handle_provider_event({
            "type": "response.created",
            "response": {"id": "response_1", "status": "in_progress"},
        })
        await session.handle_provider_event({
            "type": "response.output_item.added",
            "response_id": "response_1",
            "item": {
                "id": "assistant_1",
                "type": "message",
                "role": "assistant",
            },
        })
        await session.handle_provider_event({
            "type": "output_audio_buffer.started",
            "response_id": "response_1",
        })
        await session.handle_provider_event({
            "type": "response.done",
            "response": {
                "id": "response_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_audio",
                                "transcript": "Generated words the user did not hear.",
                            }
                        ],
                    }
                ],
            },
        })
        await session.handle_provider_event({
            "type": "output_audio_buffer.stopped",
            "response_id": "response_1",
        })

        result = await session.interrupt_response(
            request_id="wake-after-done",
            audio_end_ms=640,
        )

        assert result["status"] == "accepted"
        assert result["truncation_requested"] is True
        assert not any(event["type"] == "response.cancel" for event in sideband.sent)
        truncate = next(
            event
            for event in sideband.sent
            if event["type"] == "conversation.item.truncate"
        )
        assert truncate["item_id"] == "assistant_1"
        assert truncate["audio_end_ms"] == 640
        messages = db.get_messages("late-barge-in")
        assert messages[-1]["content"] == "[Assistant response interrupted by user.]"
        assert messages[-1]["finish_reason"] == "interrupted"
        assert session.state == "ready"
    finally:
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_structured_interrupt_before_tool_worker_start_skips_side_effects(
    tmp_path,
):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["todo"],
        session_id="pre-worker-interrupt",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="pre-worker-interrupt",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=sideband,
        call_id="call_pre_worker",
    )
    worker_waiting = threading.Event()
    release_worker = threading.Event()
    executed = []
    original_execute = session._execute_tools_sync

    def delayed_execute(*args, **kwargs):
        worker_waiting.set()
        release_worker.wait(timeout=5)
        return original_execute(*args, **kwargs)

    def fake_handle(_name, _args, _task_id, **kwargs):
        executed.append(kwargs["tool_call_id"])
        return json.dumps({"success": True})

    session._execute_tools_sync = delayed_execute
    await session.start()
    try:
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "Create a todo.",
        })
        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            await session.handle_provider_event({
                "type": "response.created",
                "response": {"id": "response_tools", "status": "in_progress"},
            })
            await session.handle_provider_event({
                "type": "response.output_item.added",
                "response_id": "response_tools",
                "item": {
                    "id": "assistant_tools",
                    "type": "message",
                    "role": "assistant",
                },
            })
            await session.handle_provider_event({
                "type": "response.done",
                "response": {
                    "id": "response_tools",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "todo_1",
                            "name": "todo",
                            "arguments": '{"todos":[{"id":"one","content":"one"}]}',
                        }
                    ],
                },
            })
            tool_task = session._tool_task
            assert tool_task is not None
            assert await asyncio.to_thread(worker_waiting.wait, 2)
            await session.interrupt_response(
                request_id="wake-before-worker",
                audio_end_ms=0,
            )
            await session.handle_provider_event({
                "type": "output_audio_buffer.started",
                "response_id": "response_tools",
            })
            release_worker.set()
            await tool_task

        assert executed == []
        assert any(
            event["type"] == "output_audio_buffer.clear" for event in sideband.sent
        )
        result = db.get_tool_result_by_call_id("pre-worker-interrupt", "todo_1")
        assert result is not None
        assert "skipped" in result["content"].lower()
    finally:
        release_worker.set()
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_barge_in_before_tool_start_skips_entire_batch(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["todo"],
        session_id="pre-tool-barge-in",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="pre-tool-barge-in",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=sideband,
        call_id="call_pre_tool",
    )
    await session.start()
    try:
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "Create both todos.",
        })
        await session.handle_provider_event({
            "type": "input_audio_buffer.speech_started"
        })
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_2",
            "transcript": "Actually, do neither.",
        })
        await session.handle_provider_event({
            "type": "response.done",
            "response": {
                "id": "response_tools",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "todo_1",
                        "name": "todo",
                        "arguments": '{"todos":[{"id":"one","content":"one"}]}',
                    },
                    {
                        "type": "function_call",
                        "call_id": "todo_2",
                        "name": "todo",
                        "arguments": '{"todos":[{"id":"two","content":"two"}]}',
                    },
                ],
            },
        })
        tool_task = session._tool_task
        assert tool_task is not None
        await tool_task

        messages = db.get_messages("pre-tool-barge-in")
        tool_rows = [message for message in messages if message["role"] == "tool"]
        assert [message["tool_call_id"] for message in tool_rows] == [
            "todo_1",
            "todo_2",
        ]
        assert all("skipped" in message["content"].lower() for message in tool_rows)
        assert [
            message["content"] for message in messages if message["role"] == "user"
        ] == [
            "Create both todos.",
            "Actually, do neither.",
        ]
        assert session._turn is not None
        assert session.state == "responding"
    finally:
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_barge_in_during_tool_preserves_started_call_and_skips_later_calls(
    tmp_path,
):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["skills"],
        session_id="in-tool-barge-in",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="in-tool-barge-in",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=sideband,
        call_id="call_in_tool",
    )
    started = threading.Event()
    release = threading.Event()
    executed = []

    def fake_handle(_name, _args, _task_id, **kwargs):
        executed.append(kwargs["tool_call_id"])
        if kwargs["tool_call_id"] == "skill_1":
            started.set()
            release.wait(timeout=5)
        return json.dumps({"success": True})

    await session.start()
    try:
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "Read both skills.",
        })
        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            await session.handle_provider_event({
                "type": "response.done",
                "response": {
                    "id": "response_tools",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "skill_1",
                            "name": "skill_view",
                            "arguments": '{"name":"one"}',
                        },
                        {
                            "type": "function_call",
                            "call_id": "skill_2",
                            "name": "skill_view",
                            "arguments": '{"name":"two"}',
                        },
                    ],
                },
            })
            tool_task = session._tool_task
            assert tool_task is not None
            assert await asyncio.to_thread(started.wait, 2)
            await session.handle_provider_event({
                "type": "input_audio_buffer.speech_started"
            })
            release.set()
            await tool_task
            assert executed == ["skill_1"]
            assert session._continuation_pending is True
            await session.handle_provider_event({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "input_2",
                "transcript": "Only use the first result.",
            })

        assert executed == ["skill_1"]
        messages = db.get_messages("in-tool-barge-in")
        second = next(
            message for message in messages if message.get("tool_call_id") == "skill_2"
        )
        assert "skipped" in second["content"].lower()
        assert [
            message["content"] for message in messages if message["role"] == "user"
        ] == ["Read both skills.", "Only use the first result."]
        assert sideband.sent[-1]["type"] == "response.create"
    finally:
        release.set()
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_replacement_speech_during_last_tool_is_durable_and_allows_next_tool(
    tmp_path,
):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["skills"],
        session_id="durable-tool-steer",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="durable-tool-steer",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=sideband,
        call_id="call_durable_tool_steer",
    )
    started = threading.Event()
    release = threading.Event()
    executed = []

    def fake_handle(_name, _args, _task_id, **kwargs):
        call_id = kwargs["tool_call_id"]
        executed.append(call_id)
        if call_id == "skill_1":
            started.set()
            release.wait(timeout=5)
        return json.dumps({"success": True, "call_id": call_id})

    await session.start()
    try:
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "Read the first skill.",
        })
        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            await session.handle_provider_event({
                "type": "response.created",
                "response": {"id": "response_tools_1", "status": "in_progress"},
            })
            await session.handle_provider_event({
                "type": "response.done",
                "response": {
                    "id": "response_tools_1",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "skill_1",
                            "name": "skill_view",
                            "arguments": '{"name":"one"}',
                        }
                    ],
                },
            })
            first_task = session._tool_task
            assert first_task is not None
            assert await asyncio.to_thread(started.wait, 2)
            await session.interrupt_response(
                request_id="wake-durable-steer",
                audio_end_ms=0,
            )
            await session.handle_provider_event({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "input_2",
                "transcript": "Use the second skill next.",
            })
            assert session._barge_in_during_response is False
            release.set()
            await first_task

            durable = db.get_tool_result_by_call_id(
                "durable-tool-steer",
                "skill_1",
            )
            assert durable is not None
            assert "Use the second skill next." in durable["content"]

            await session.handle_provider_event({
                "type": "response.created",
                "response": {"id": "response_tools_2", "status": "in_progress"},
            })
            await session.handle_provider_event({
                "type": "response.done",
                "response": {
                    "id": "response_tools_2",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "skill_2",
                            "name": "skill_view",
                            "arguments": '{"name":"two"}',
                        }
                    ],
                },
            })
            second_task = session._tool_task
            assert second_task is not None
            await second_task

        assert executed == ["skill_1", "skill_2"]
    finally:
        release.set()
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_interrupted_tool_turn_expires_when_replacement_audio_never_arrives(
    tmp_path,
):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["skills"],
        session_id="tool-handoff-timeout",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    session = RealtimeVoiceSession(
        session_id="tool-handoff-timeout",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
            preroll_timeout_seconds=0.01,
            transcription_timeout_seconds=0.01,
        ),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_tool_handoff",
    )
    started = threading.Event()
    release = threading.Event()

    def fake_handle(_name, _args, _task_id, **_kwargs):
        started.set()
        release.wait(timeout=5)
        return json.dumps({"success": True})

    await session.start()
    try:
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "Read the skill.",
        })
        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            await session.handle_provider_event({
                "type": "response.done",
                "response": {
                    "id": "response_tools",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "skill_1",
                            "name": "skill_view",
                            "arguments": '{"name":"one"}',
                        }
                    ],
                },
            })
            tool_task = session._tool_task
            assert tool_task is not None
            assert await asyncio.to_thread(started.wait, 2)
            await session.interrupt_response(
                request_id="wake-without-handoff",
                audio_end_ms=0,
            )
            release.set()
            await tool_task

        await _wait_until(lambda: session.state == "ready", timeout=2)
        assert session.can_suspend is True
        messages = db.get_messages("tool-handoff-timeout")
        assert messages[-1]["finish_reason"] == "interrupted"
        assert "replacement audio did not arrive" in messages[-1]["content"]
    finally:
        release.set()
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_background_reviews_are_single_flight_and_coalesce_boundaries(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["skills"],
        session_id="review-coalesce",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    session = RealtimeVoiceSession(
        session_id="review-coalesce",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
        ),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_review",
    )
    first_started = threading.Event()
    release = threading.Event()
    calls = []

    def fake_spawn(_agent, snapshot, *, review_memory, review_skills):
        def target():
            calls.append({
                "last": snapshot[-1]["content"],
                "memory": review_memory,
                "skills": review_skills,
            })
            if len(calls) == 1:
                first_started.set()
                release.wait(timeout=5)
                return False
            return True

        return target, "review"

    await session.start()
    try:
        db.append_message("review-coalesce", role="user", content="first")
        first_snapshot = db.get_messages("review-coalesce")
        with patch(
            "agent.background_review.spawn_background_review_thread",
            side_effect=fake_spawn,
        ):
            session._schedule_background_review_sync(
                messages_snapshot=first_snapshot,
                review_memory=True,
            )
            assert await asyncio.to_thread(first_started.wait, 2)
            first_thread = session._review_thread
            assert first_thread is not None

            recovered_session = RealtimeVoiceSession(
                session_id="review-coalesce",
                agent=agent,
                config=session.config,
                frozen_instructions=session.frozen_instructions,
                conversation_history=[],
                sideband=_FakeSideband(),
                call_id="call_review_recovered",
            )
            await recovered_session._recover_due_review()
            recovered_thread = recovered_session._review_thread
            assert recovered_thread is not None
            assert recovered_thread.is_alive()

            second_boundary = db.append_message(
                "review-coalesce", role="assistant", content="second"
            )
            session._schedule_background_review_sync(
                messages_snapshot=db.get_messages("review-coalesce"),
                review_skills=True,
            )
            release.set()
            await asyncio.to_thread(first_thread.join, 5)
            await asyncio.to_thread(recovered_thread.join, 5)

        assert calls == [
            {"last": "first", "memory": True, "skills": False},
            {"last": "second", "memory": True, "skills": True},
        ]
        state = db.get_realtime_session_state("review-coalesce")
        assert state["review_state"] == "completed"
        assert state["review_boundary_message_id"] == second_boundary
        assert state["review_memory"] is True
        assert state["review_skills"] is True
        assert not any(
            event.get("type") == "response.create" for event in session.sideband.sent
        )
    finally:
        release.set()
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_review_recovery_preserves_stored_snapshot_boundary(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["skills"],
        session_id="review-boundary",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    session = RealtimeVoiceSession(
        session_id="review-boundary",
        agent=agent,
        config=RealtimeVoiceConfig(enabled=True),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_review_boundary",
    )
    snapshots = []

    def fake_spawn(_agent, snapshot, *, review_memory, review_skills):
        def target():
            snapshots.append((
                [message["content"] for message in snapshot],
                review_memory,
                review_skills,
            ))
            return True

        return target, "review"

    await session.start()
    try:
        boundary = db.append_message(
            "review-boundary",
            role="user",
            content="included",
        )
        db.mark_realtime_review_due(
            "review-boundary",
            boundary_message_id=boundary,
            review_memory=True,
            review_skills=False,
        )
        db.append_message(
            "review-boundary",
            role="assistant",
            content="not included",
        )

        with patch(
            "agent.background_review.spawn_background_review_thread",
            side_effect=fake_spawn,
        ):
            await session._recover_due_review()
            thread = session._review_thread
            assert thread is not None
            await asyncio.to_thread(thread.join, 5)

        assert snapshots == [(["included"], True, False)]
        state = db.get_realtime_session_state("review-boundary")
        assert state["review_state"] == "completed"
        assert state["review_boundary_message_id"] == boundary
    finally:
        await session.close(reason="test_complete", end_session=False)
        db.close()
