import asyncio
import base64
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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


class _FailingSideband(_FakeSideband):
    async def send(self, event):
        self.sent.append(event)
        raise OpenAIRealtimeError("provider send failed", code="send_failed")


class _CommitDuringAppendSideband(_FakeSideband):
    def __init__(self):
        super().__init__()
        self.handler = None
        self.append_count = 0

    async def send(self, event):
        await super().send(event)
        if event["type"] != "input_audio_buffer.append":
            return
        self.append_count += 1
        if self.append_count == 1 and self.handler is not None:
            await self.handler({
                "type": "input_audio_buffer.committed",
                "item_id": "input_early",
            })
            await self.handler({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "input_early",
                "transcript": "early",
            })


class _AckFallbackCommitSideband(_FakeSideband):
    def __init__(self):
        super().__init__()
        self.handler = None

    async def send(self, event):
        await super().send(event)
        if (
            event["type"] == "input_audio_buffer.commit"
            and self.handler is not None
        ):
            await self.handler({
                "type": "input_audio_buffer.committed",
                "item_id": "input_fallback",
            })


class _CommitThenFailAppendSideband(_FakeSideband):
    def __init__(self):
        super().__init__()
        self.handler = None
        self.append_count = 0

    async def send(self, event):
        await super().send(event)
        if event["type"] != "input_audio_buffer.append":
            return
        self.append_count += 1
        if self.append_count == 2 and self.handler is not None:
            await self.handler({
                "type": "input_audio_buffer.committed",
                "item_id": "input_partial",
            })
            await self.handler({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "input_partial",
                "transcript": "partial",
            })
            raise OpenAIRealtimeError("append failed", code="append_failed")


STREAM_ID = "11111111-1111-4111-8111-111111111111"


def _begin_command(stream_id=STREAM_ID, *, preroll_ms=0):
    return {
        "type": "turn.begin",
        "stream_id": stream_id,
        "barge_in": False,
        "preroll_ms": preroll_ms,
        "interrupted": None,
    }


@pytest.mark.asyncio
async def test_audio_uplink_forwards_burst_without_vad_updates_or_manual_commit():
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="uplink",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
            audio_uplink_append_chunk_milliseconds=80,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_uplink",
    )
    session.state = "ready"
    audio = b"\x01\x00" * 48_000

    try:
        await session.begin_audio_uplink(_begin_command())
        await session.append_audio_uplink(STREAM_ID, audio)
        result = await session.end_audio_uplink(STREAM_ID)

        appended = b"".join(
            base64.b64decode(event["audio"])
            for event in sideband.sent
            if event["type"] == "input_audio_buffer.append"
        )
        assert appended == audio
        assert result["status"] == "ending"
        assert len(sideband.sent) == 25
        assert not any(
            event["type"]
            in {
                "session.update",
                "input_audio_buffer.clear",
                "input_audio_buffer.commit",
            }
            for event in sideband.sent
        )
        assert session.can_renew is False
    finally:
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_dispatches_before_turn_end_and_transcription():
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="uplink-vad",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_uplink_vad",
    )
    session.state = "ready"
    session._handle_input = AsyncMock()

    try:
        await session.begin_audio_uplink(_begin_command())
        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 7200)
        await session.handle_provider_event(
            {"type": "input_audio_buffer.speech_started"}
        )
        await session.handle_provider_event({
            "type": "input_audio_buffer.committed",
            "item_id": "input_1",
        })
        session._handle_input.assert_awaited_once_with(
            "[Voice transcript pending; use the committed audio input.]",
            "input_1",
        )
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "latest news",
        })
        assert session._handle_input.await_count == 1

        result = await session.end_audio_uplink(STREAM_ID)

        assert result["status"] == "committed"
        assert session._audio_uplink is None
    finally:
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_turn_end_authorizes_commit_before_declared_burst():
    session = RealtimeVoiceSession(
        session_id="uplink-short-burst",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_uplink_short_burst",
    )
    session.state = "ready"
    session._handle_input = AsyncMock()

    try:
        await session.begin_audio_uplink(_begin_command(preroll_ms=2000))
        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 4800)
        await session.handle_provider_event({
            "type": "input_audio_buffer.committed",
            "item_id": "input_short",
        })
        session._handle_input.assert_not_awaited()

        result = await session.end_audio_uplink(STREAM_ID)

        assert result["status"] == "committed"
        assert session._audio_uplink is None
        session._handle_input.assert_awaited_once_with(
            "[Voice transcript pending; use the committed audio input.]",
            "input_short",
        )
    finally:
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_combines_vad_commits_inside_the_burst():
    session = RealtimeVoiceSession(
        session_id="uplink-multiple-commits",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_uplink_multiple",
    )
    session.state = "ready"
    session._handle_input = AsyncMock()

    try:
        await session.begin_audio_uplink(_begin_command(preroll_ms=200))
        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 1920)
        await session.handle_provider_event(
            {"type": "input_audio_buffer.speech_started"}
        )
        await session.handle_provider_event({
            "type": "input_audio_buffer.speech_stopped",
            "audio_end_ms": 80,
        })
        await session.handle_provider_event({
            "type": "input_audio_buffer.committed",
            "item_id": "input_1",
        })
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_1",
            "transcript": "tell me the",
        })
        session._handle_input.assert_not_awaited()

        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 2880)
        await session.handle_provider_event(
            {"type": "input_audio_buffer.speech_started"}
        )
        await session.handle_provider_event({
            "type": "input_audio_buffer.speech_stopped",
            "audio_end_ms": 200,
        })
        await session.handle_provider_event({
            "type": "input_audio_buffer.committed",
            "item_id": "input_2",
        })
        session._handle_input.assert_awaited_once_with(
            (
                "tell me the\n"
                "[Voice transcript pending; use the committed audio input.]"
            ),
            "input_2",
        )
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_2",
            "transcript": "latest news",
        })
        await session.end_audio_uplink(STREAM_ID)
        assert session._handle_input.await_count == 1
    finally:
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_commit_during_append_keeps_later_chunks_pending():
    sideband = _CommitDuringAppendSideband()
    session = RealtimeVoiceSession(
        session_id="uplink-commit-race",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(
            enabled=True,
            provider_call_max_seconds=3600,
            audio_uplink_append_chunk_milliseconds=80,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_uplink_commit_race",
    )
    sideband.handler = session.handle_provider_event
    session.state = "ready"
    session._handle_input = AsyncMock()

    try:
        await session.begin_audio_uplink(_begin_command())
        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 3840)
        result = await session.end_audio_uplink(STREAM_ID)

        assert result["status"] == "ending"
        assert session._audio_uplink is not None
        assert session._audio_uplink.uncommitted_bytes > 0
        session._handle_input.assert_not_awaited()

        await session.handle_provider_event({
            "type": "input_audio_buffer.speech_stopped",
            "audio_end_ms": 160,
        })
        await session.handle_provider_event({
            "type": "input_audio_buffer.committed",
            "item_id": "input_tail",
        })
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_tail",
            "transcript": "tail",
        })

        session._handle_input.assert_awaited_once_with(
            (
                "early\n"
                "[Voice transcript pending; use the committed audio input.]"
            ),
            "input_tail",
        )
        assert session._audio_uplink is None
    finally:
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_commit_during_final_send_dispatches_without_end():
    sideband = _CommitDuringAppendSideband()
    session = RealtimeVoiceSession(
        session_id="uplink-final-send-commit",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(
            enabled=True,
            provider_call_max_seconds=3600,
            audio_uplink_append_chunk_milliseconds=80,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_uplink_final_send_commit",
    )
    sideband.handler = session.handle_provider_event
    session.state = "ready"
    session._handle_input = AsyncMock()

    try:
        await session.begin_audio_uplink(_begin_command(preroll_ms=80))
        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 1920)

        session._handle_input.assert_awaited_once_with("early", "input_early")
        assert session._audio_uplink is not None
        assert session._audio_uplink.state == "dispatched"
        assert session._audio_uplink.uncommitted_bytes == 0
    finally:
        await session.end_audio_uplink(STREAM_ID)
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_append_failure_quarantines_partial_provider_input():
    sideband = _CommitThenFailAppendSideband()
    session = RealtimeVoiceSession(
        session_id="uplink-append-failure",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(
            enabled=True,
            provider_call_max_seconds=3600,
            audio_uplink_append_chunk_milliseconds=80,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_uplink_append_failure",
    )
    sideband.handler = session.handle_provider_event
    session.state = "ready"
    session._handle_input = AsyncMock()

    with pytest.raises(OpenAIRealtimeError, match="append failed"):
        await session.begin_audio_uplink(_begin_command())
        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 3840)

    session._handle_input.assert_not_awaited()
    assert session._renewal_required is True
    assert session._quarantine_provider_input is True

    await session.handle_provider_event({
        "type": "input_audio_buffer.committed",
        "item_id": "input_late",
    })
    await session.handle_provider_event({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "input_late",
        "transcript": "must be ignored",
    })
    session._handle_input.assert_not_awaited()
    await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_async_append_error_quarantines_late_commit():
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="uplink-async-append-error",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_uplink_async_append_error",
    )
    session.state = "ready"
    session._handle_input = AsyncMock()

    await session.begin_audio_uplink(_begin_command())
    await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 4800)
    append_event = next(
        event
        for event in sideband.sent
        if event["type"] == "input_audio_buffer.append"
    )
    await session.handle_provider_event({
        "type": "error",
        "error": {
            "event_id": append_event["event_id"],
            "code": "invalid_audio",
            "message": "append rejected",
        },
    })

    assert session._renewal_required is True
    assert session._quarantine_provider_input is True
    assert session._audio_uplink is None

    await session.handle_provider_event({
        "type": "input_audio_buffer.committed",
        "item_id": "input_rejected",
    })
    await session.handle_provider_event({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "input_rejected",
        "transcript": "must be ignored",
    })
    session._handle_input.assert_not_awaited()
    await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_late_old_append_error_fails_current_audio_uplink():
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="uplink-old-append-error",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_uplink_old_append_error",
    )
    session.state = "ready"

    first_stream = STREAM_ID
    second_stream = "22222222-2222-4222-8222-222222222222"
    await session.begin_audio_uplink(_begin_command(first_stream))
    await session.append_audio_uplink(first_stream, b"\x01\x00" * 4800)
    old_append = next(
        event
        for event in sideband.sent
        if event["type"] == "input_audio_buffer.append"
    )
    old_uplink = session._audio_uplink
    old_uplink.state = "committed"
    session._audio_uplink = None

    await session.begin_audio_uplink(_begin_command(second_stream))
    current_uplink = session._audio_uplink
    await session.handle_provider_event({
        "type": "error",
        "error": {
            "event_id": old_append["event_id"],
            "code": "invalid_audio",
            "message": "old append rejected",
        },
    })

    assert old_uplink.state == "failed"
    assert current_uplink.state == "failed"
    assert session._audio_uplink is None
    assert session._quarantine_provider_input is True
    with pytest.raises(RealtimeProtocolError) as quarantined:
        await session.append_audio_uplink(
            second_stream,
            b"\x01\x00" * 2400,
        )
    assert quarantined.value.code == "session_renewal_required"
    await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_uses_provider_timeline_baseline_for_vad_boundary():
    session = RealtimeVoiceSession(
        session_id="uplink-provider-timeline",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(
            enabled=True,
            provider_call_max_seconds=3600,
            audio_uplink_append_chunk_milliseconds=80,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_uplink_provider_timeline",
    )
    session.state = "ready"
    session._provider_input_audio_bytes_sent = 10_000 * 48
    session._provider_input_timeline_known = True
    session._handle_input = AsyncMock()

    try:
        await session.begin_audio_uplink(_begin_command())
        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 5760)
        await session.handle_provider_event({
            "type": "input_audio_buffer.speech_stopped",
            "audio_end_ms": 10_160,
        })
        await session.handle_provider_event({
            "type": "input_audio_buffer.committed",
            "item_id": "input_boundary",
        })
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_boundary",
            "transcript": "first segment",
        })
        result = await session.end_audio_uplink(STREAM_ID)

        assert result["status"] == "ending"
        assert session._audio_uplink is not None
        assert session._audio_uplink.uncommitted_bytes == 80 * 48
        session._handle_input.assert_not_awaited()
    finally:
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_recovered_call_requires_renewal_for_timeline():
    session = RealtimeVoiceSession(
        session_id="uplink-recovered-timeline",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_uplink_recovered_timeline",
        call_started_at=1,
    )
    session.state = "ready"

    with pytest.raises(RealtimeProtocolError) as renewal:
        await session.begin_audio_uplink(_begin_command())

    assert renewal.value.code == "session_renewal_required"
    assert session._renewal_required is True
    assert session.state == "degraded"
    await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_rejects_concurrent_socket_attachment():
    session = RealtimeVoiceSession(
        session_id="uplink-reattach",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_uplink_reattach",
    )
    session.state = "ready"

    try:
        await session.begin_audio_uplink(_begin_command(), owner_id="old")
        with pytest.raises(RealtimeProtocolError) as duplicate:
            await session.begin_audio_uplink(
                _begin_command(),
                owner_id="new",
            )
        assert duplicate.value.code == "audio_stream_already_attached"
        assert session._audio_uplink is not None
        assert session._audio_uplink.owner_id == "old"

        with pytest.raises(RealtimeProtocolError) as wrong_owner:
            await session.append_audio_uplink(
                STREAM_ID,
                b"\x01\x00" * 2400,
                owner_id="new",
            )
        assert wrong_owner.value.code == "audio_stream_not_owner"

        await session.append_audio_uplink(
            STREAM_ID,
            b"\x01\x00" * 2400,
            owner_id="old",
        )
        await session.end_audio_uplink(STREAM_ID, owner_id="old")
    finally:
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_disconnect_commits_long_audio_and_discards_short_audio():
    long_sideband = _FakeSideband()
    long_session = RealtimeVoiceSession(
        session_id="uplink-long-disconnect",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=long_sideband,
        call_id="call_uplink_long",
    )
    long_session.state = "ready"
    await long_session.begin_audio_uplink(_begin_command())
    await long_session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 4800)
    await long_session.teardown_audio_uplink(
        STREAM_ID,
        reason="audio_uplink_disconnected",
    )
    assert long_sideband.sent[-1]["type"] == "input_audio_buffer.commit"
    await long_session.close(reason="test_complete", end_session=False)

    short_sideband = _FakeSideband()
    short_session = RealtimeVoiceSession(
        session_id="uplink-short-disconnect",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=short_sideband,
        call_id="call_uplink_short",
    )
    short_session.state = "ready"
    await short_session.begin_audio_uplink(_begin_command())
    await short_session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 1200)
    result = await short_session.teardown_audio_uplink(
        STREAM_ID,
        reason="audio_uplink_disconnected",
    )
    assert short_sideband.sent[-1]["type"] == "input_audio_buffer.clear"
    assert result["status"] == "discarded"
    assert short_session.can_renew is True
    await short_session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_disconnect_dispatches_audio_already_committed_by_vad():
    session = RealtimeVoiceSession(
        session_id="uplink-committed-disconnect",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_uplink_committed_disconnect",
    )
    session.state = "ready"
    session._handle_input = AsyncMock()

    try:
        await session.begin_audio_uplink(_begin_command())
        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 4800)
        await session.handle_provider_event({
            "type": "input_audio_buffer.committed",
            "item_id": "input_committed",
        })
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_committed",
            "transcript": "committed before disconnect",
        })

        result = await session.teardown_audio_uplink(
            STREAM_ID,
            reason="audio_uplink_disconnected",
        )

        assert result["status"] == "committed"
        assert session._audio_uplink is None
        session._handle_input.assert_awaited_once_with(
            "[Voice transcript pending; use the committed audio input.]",
            "input_committed",
        )
    finally:
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_fallback_commit_ack_cancels_commit_timeout():
    sideband = _AckFallbackCommitSideband()
    session = RealtimeVoiceSession(
        session_id="uplink-fallback-ack",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(
            enabled=True,
            provider_call_max_seconds=3600,
            audio_uplink_idle_timeout_seconds=0.05,
            transcription_timeout_seconds=1,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_uplink_fallback_ack",
    )
    sideband.handler = session.handle_provider_event
    session.state = "ready"
    session._handle_input = AsyncMock()

    try:
        await session.begin_audio_uplink(_begin_command())
        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 4800)
        await session.teardown_audio_uplink(
            STREAM_ID,
            reason="audio_uplink_disconnected",
        )
        await asyncio.sleep(0.1)

        assert session._renewal_required is False
        assert session._audio_uplink is None
        session._handle_input.assert_awaited_once_with(
            "[Voice transcript pending; use the committed audio input.]",
            "input_fallback",
        )

        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_fallback",
            "transcript": "fallback committed",
        })
        assert session._audio_uplink is None
        assert session._handle_input.await_count == 1
    finally:
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_vad_commit_precedes_fallback_commit_ack():
    session = RealtimeVoiceSession(
        session_id="uplink-vad-before-fallback",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(
            enabled=True,
            provider_call_max_seconds=3600,
            audio_uplink_idle_timeout_seconds=1,
        ),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_uplink_vad_before_fallback",
    )
    session.state = "ready"
    session._handle_input = AsyncMock()

    try:
        await session.begin_audio_uplink(_begin_command())
        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 24_000)
        await session.handle_provider_event({
            "type": "input_audio_buffer.speech_stopped",
            "audio_end_ms": 800,
        })
        await session.teardown_audio_uplink(
            STREAM_ID,
            reason="audio_uplink_disconnected",
        )

        await session.handle_provider_event({
            "type": "input_audio_buffer.committed",
            "item_id": "input_vad",
        })
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_vad",
            "transcript": "vad segment",
        })
        assert session._audio_uplink is not None
        assert session._audio_uplink.uncommitted_bytes == 200 * 48
        assert session._audio_uplink.fallback_commit_sent is True
        session._handle_input.assert_not_awaited()

        await session.handle_provider_event({
            "type": "input_audio_buffer.committed",
            "item_id": "input_tail",
        })
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_tail",
            "transcript": "tail",
        })
        assert session._audio_uplink is None
        session._handle_input.assert_awaited_once_with(
            (
                "vad segment\n"
                "[Voice transcript pending; use the committed audio input.]"
            ),
            "input_tail",
        )
    finally:
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_barge_in_targets_item_clamps_and_is_idempotent():
    sideband = _FakeSideband()
    agent = SimpleNamespace(
        tools=[],
        _session_db=None,
        _skip_unstarted_tool_calls=False,
    )
    session = RealtimeVoiceSession(
        session_id="uplink-barge-in",
        agent=agent,
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_uplink_barge",
    )
    session.state = "responding"
    session._turn = SimpleNamespace(messages=[])
    session._active_response_id = "resp_target"
    session._active_output_audio_response_id = "resp_target"
    session._generation_active_response_ids.add("resp_target")
    await session.handle_provider_event({
        "type": "response.output_item.added",
        "response_id": "resp_target",
        "item": {
            "id": "item_target",
            "type": "message",
            "role": "assistant",
        },
    })
    await session.handle_provider_event({
        "type": "response.output_audio.delta",
        "response_id": "resp_target",
        "item_id": "item_target",
        "delta": base64.b64encode(b"\x00\x00" * 24_000).decode("ascii"),
    })
    command = {
        "type": "turn.begin",
        "stream_id": STREAM_ID,
        "barge_in": True,
        "preroll_ms": 0,
        "interrupted": {
            "response_id": "resp_target",
            "item_id": "item_target",
            "audio_end_ms": 1200,
        },
    }

    try:
        first = await session.begin_audio_uplink(command)
        repeated = await session.begin_audio_uplink(command)

        assert first == repeated
        assert first["interruption"]["audio_end_ms"] == 1000
        assert [event["type"] for event in sideband.sent] == [
            "response.cancel",
            "output_audio_buffer.clear",
            "conversation.item.truncate",
        ]
        assert sideband.sent[-1]["item_id"] == "item_target"
        assert sideband.sent[-1]["audio_end_ms"] == 1000

        changed = {
            **command,
            "interrupted": {**command["interrupted"], "audio_end_ms": 900},
        }
        with pytest.raises(RealtimeProtocolError) as conflict:
            await session.begin_audio_uplink(changed)
        assert conflict.value.code == "audio_stream_idempotency_conflict"
    finally:
        session._turn = None
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_stale_barge_target_does_not_block_new_turn():
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="uplink-stale-barge",
        agent=SimpleNamespace(
            tools=[],
            _session_db=None,
            _skip_unstarted_tool_calls=False,
        ),
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_uplink_stale",
    )
    session.state = "responding"
    session._turn = SimpleNamespace(messages=[])

    try:
        result = await session.begin_audio_uplink({
            "type": "turn.begin",
            "stream_id": STREAM_ID,
            "barge_in": True,
            "preroll_ms": 0,
            "interrupted": {
                "response_id": "resp_stale",
                "item_id": "item_stale",
                "audio_end_ms": 250,
            },
        })

        assert result["status"] == "accepted"
        assert result["interruption"]["target_status"] == "stale"
        assert sideband.sent == []
    finally:
        session._turn = None
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_tool_barge_retains_transcript_for_tool_boundary():
    sideband = _FakeSideband()
    release = asyncio.Event()
    tool_task = asyncio.create_task(release.wait())
    agent = SimpleNamespace(
        tools=[],
        _session_db=None,
        _skip_unstarted_tool_calls=False,
        steer=Mock(),
    )
    session = RealtimeVoiceSession(
        session_id="uplink-tool-steer",
        agent=agent,
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_uplink_tool_steer",
    )
    session.state = "tool_wait"
    session._turn = SimpleNamespace(messages=[])
    session._tool_task = tool_task

    try:
        await session.begin_audio_uplink({
            "type": "turn.begin",
            "stream_id": STREAM_ID,
            "barge_in": True,
            "preroll_ms": 0,
            "interrupted": {
                "response_id": "resp_stale",
                "item_id": "item_stale",
                "audio_end_ms": 0,
            },
        })
        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 4800)
        await session.handle_provider_event({
            "type": "input_audio_buffer.committed",
            "item_id": "input_steer",
        })

        agent.steer.assert_not_called()
        assert session._pending_next_inputs[-1][0] == (
            "[Voice transcript pending; use the committed audio input.]"
        )

        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_steer",
            "transcript": "Use the shorter answer.",
        })

        agent.steer.assert_not_called()
        assert session._pending_next_inputs[-1] == (
            "Use the shorter answer.",
            "input_steer",
        )
    finally:
        release.set()
        await tool_task
        session._tool_task = None
        session._turn = None
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_settled_tool_barge_resumes_waiting_continuation_immediately():
    session = RealtimeVoiceSession(
        session_id="uplink-tool-continuation",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_uplink_tool_continuation",
    )
    session.state = "tool_wait"
    session._turn = SimpleNamespace(messages=[])
    session._continuation_pending = True
    session._client_interrupt_pending = True
    session._finalize_current_turn = AsyncMock()

    await session.begin_audio_uplink({
        "type": "turn.begin",
        "stream_id": STREAM_ID,
        "barge_in": True,
        "preroll_ms": 0,
        "interrupted": {
            "response_id": "resp_stale",
            "item_id": "item_stale",
            "audio_end_ms": 0,
        },
    })
    await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 4800)
    await session.handle_provider_event({
        "type": "input_audio_buffer.committed",
        "item_id": "input_waiting",
    })
    await session.handle_provider_event({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "input_waiting",
        "transcript": "Continue with this request.",
    })

    assert session._continuation_pending is False
    assert session._pending_next_inputs[-1] == (
        "Continue with this request.",
        "input_waiting",
    )
    session._finalize_current_turn.assert_awaited_once_with(
        "[Assistant response interrupted by new speech.]",
        response_id="",
        interrupted=True,
    )
    session._turn = None
    await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_pending_transcript_does_not_block_next_audio_uplink():
    session = RealtimeVoiceSession(
        session_id="uplink-pending-transcript-admission",
        agent=SimpleNamespace(tools=[], _session_db=None),
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=_FakeSideband(),
        call_id="call_uplink_pending_transcript",
    )
    session.state = "ready"
    pending = asyncio.create_task(asyncio.sleep(10))
    session._transcript_wait_tasks["previous_item"] = pending

    try:
        result = await session.begin_audio_uplink(_begin_command())

        assert result["status"] == "accepted"
        assert session.can_renew is False
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        session._transcript_wait_tasks.clear()
        await session.close(reason="test_complete", end_session=False)


@pytest.mark.asyncio
async def test_audio_uplink_interrupt_failure_requires_renewal_and_clears_flags():
    sideband = _FailingSideband()
    agent = SimpleNamespace(
        tools=[],
        _session_db=None,
        _skip_unstarted_tool_calls=False,
    )
    session = RealtimeVoiceSession(
        session_id="uplink-failed-barge",
        agent=agent,
        config=RealtimeVoiceConfig(enabled=True, provider_call_max_seconds=3600),
        frozen_instructions="prompt",
        conversation_history=[],
        sideband=sideband,
        call_id="call_uplink_failed_barge",
    )
    session.state = "responding"
    session._turn = SimpleNamespace(messages=[])
    session._active_response_id = "resp_target"
    session._generation_active_response_ids.add("resp_target")
    session._response_audio_items["resp_target"].add("item_target")
    session._response_audio_bytes[("resp_target", "item_target")] = 48_000

    with pytest.raises(OpenAIRealtimeError, match="provider send failed"):
        await session.begin_audio_uplink({
            "type": "turn.begin",
            "stream_id": STREAM_ID,
            "barge_in": True,
            "preroll_ms": 0,
            "interrupted": {
                "response_id": "resp_target",
                "item_id": "item_target",
                "audio_end_ms": 500,
            },
        })

    assert session._renewal_required is True
    assert session.state == "degraded"
    assert session._audio_uplink is None
    assert (session.call_id, STREAM_ID) not in session._audio_uplinks
    assert session._client_interrupt_pending is False
    assert session._barge_in_during_response is False
    assert agent._skip_unstarted_tool_calls is False
    session._turn = None
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
            audio_uplink_idle_timeout_seconds=0.01,
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
async def test_audio_uplink_response_create_precedes_transcript_and_updates_db(
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
        session_id="uplink-async-transcript",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="uplink-async-transcript",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
            transcription_timeout_seconds=1,
        ),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=sideband,
        call_id="call_uplink_async_transcript",
    )

    await session.start()
    try:
        await session.begin_audio_uplink(_begin_command(preroll_ms=0))
        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 4800)
        await session.handle_provider_event({
            "type": "input_audio_buffer.speech_stopped",
            "audio_end_ms": 200,
        })
        await session.handle_provider_event({
            "type": "input_audio_buffer.committed",
            "item_id": "input_async",
        })

        assert sideband.sent[-1]["type"] == "response.create"
        user_rows = [
            message
            for message in db.get_messages("uplink-async-transcript")
            if message["role"] == "user"
        ]
        assert user_rows[-1]["content"] == (
            "[Voice transcript pending; use the committed audio input.]"
        )

        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_async",
            "transcript": "Tell me the latest news.",
        })

        user_rows = [
            message
            for message in db.get_messages("uplink-async-transcript")
            if message["role"] == "user"
        ]
        assert user_rows[-1]["content"] == "Tell me the latest news."
        assert user_rows[-1]["display_metadata"]["transcript_available"] is True
        assert (
            session._turn.messages[session._turn.current_turn_user_idx]["content"]
            == "Tell me the latest news."
        )
        assert not any(
            event["type"] == "input_audio_buffer.commit"
            for event in sideband.sent
        )
        await session.end_audio_uplink(STREAM_ID)
    finally:
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_audio_uplink_defers_host_finalization_until_transcript(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["todo"],
        session_id="uplink-deferred-finalization",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="uplink-deferred-finalization",
        agent=agent,
        config=RealtimeVoiceConfig(
            enabled=True,
            intermediate_speech_enabled=False,
            provider_call_max_seconds=3600,
            transcription_timeout_seconds=1,
        ),
        frozen_instructions=prepare_realtime_agent(agent, []),
        conversation_history=[],
        sideband=sideband,
        call_id="call_uplink_deferred_finalization",
    )

    await session.start()
    try:
        await session.begin_audio_uplink(_begin_command())
        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 4800)
        await session.handle_provider_event({
            "type": "input_audio_buffer.committed",
            "item_id": "input_deferred",
        })
        await session._finalize_current_turn(
            "Fast answer.",
            response_id="response_fast",
        )

        assert session._turn is not None
        assert session._deferred_turn_finalization is not None
        assert [
            message["role"]
            for message in db.get_messages("uplink-deferred-finalization")
        ] == ["user"]

        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_deferred",
            "transcript": "Actual spoken request.",
        })

        assert session._turn is None
        assert session._deferred_turn_finalization is None
        messages = db.get_messages("uplink-deferred-finalization")
        assert [message["role"] for message in messages[-2:]] == [
            "user",
            "assistant",
        ]
        assert messages[-2]["content"] == "Actual spoken request."
        assert messages[-1]["content"] == "Fast answer."
    finally:
        await session.close(reason="test_complete", end_session=False)
        db.close()


@pytest.mark.asyncio
async def test_audio_uplink_transcript_timeout_does_not_delay_response(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-realtime",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai-api",
        api_mode="chat_completions",
        enabled_toolsets=["todo"],
        session_id="uplink-transcript-timeout",
        platform="realtime_voice",
        session_db=db,
        quiet_mode=True,
    )
    agent._skill_nudge_interval = 0
    sideband = _FakeSideband()
    session = RealtimeVoiceSession(
        session_id="uplink-transcript-timeout",
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
        call_id="call_uplink_transcript_timeout",
    )

    await session.start()
    try:
        await session.begin_audio_uplink(_begin_command())
        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 4800)
        await session.handle_provider_event({
            "type": "input_audio_buffer.committed",
            "item_id": "input_timeout",
        })
        assert sideband.sent[-1]["type"] == "response.create"

        await _wait_until(
            lambda: any(
                event["type"] == "warning"
                and event["data"]["code"] == "transcription_timeout"
                for event in session.broker.snapshot()
            )
        )
        user_rows = [
            message
            for message in db.get_messages("uplink-transcript-timeout")
            if message["role"] == "user"
        ]
        assert "transcript unavailable" in user_rows[-1]["content"].lower()
        assert user_rows[-1]["display_metadata"]["transcript_available"] is False

        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_timeout",
            "transcript": "Too late.",
        })
        user_rows = [
            message
            for message in db.get_messages("uplink-transcript-timeout")
            if message["role"] == "user"
        ]
        assert "transcript unavailable" in user_rows[-1]["content"].lower()
        await session.end_audio_uplink(STREAM_ID)
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
async def test_atomic_uplink_interrupt_cancels_truncates_and_accepts_next_turn(
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
    sideband = _FakeSideband()
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
        await session.handle_provider_event({
            "type": "response.output_audio.delta",
            "response_id": "response_1",
            "item_id": "assistant_1",
            "delta": base64.b64encode(b"\x00\x00" * 24_000).decode("ascii"),
        })
        command = {
            "type": "turn.begin",
            "stream_id": STREAM_ID,
            "barge_in": True,
            "preroll_ms": 0,
            "interrupted": {
                "response_id": "response_1",
                "item_id": "assistant_1",
                "audio_end_ms": 875,
            },
        }
        first = await session.begin_audio_uplink(command)
        repeated = await session.begin_audio_uplink(command)
        assert first == repeated
        assert first["interruption"]["truncation_requested"] is True
        assert [event["type"] for event in sideband.sent].count("response.cancel") == 1
        truncate = next(
            event
            for event in sideband.sent
            if event["type"] == "conversation.item.truncate"
        )
        assert truncate["item_id"] == "assistant_1"
        assert truncate["audio_end_ms"] == 875

        await session.append_audio_uplink(STREAM_ID, b"\x01\x00" * 7200)
        await session.handle_provider_event({
            "type": "input_audio_buffer.committed",
            "item_id": "input_uplink",
        })
        await session.handle_provider_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input_uplink",
            "transcript": "Actually, answer briefly.",
        })
        uplink = await session.end_audio_uplink(STREAM_ID)
        assert uplink["status"] == "committed"
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
            audio_uplink_idle_timeout_seconds=0.01,
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
