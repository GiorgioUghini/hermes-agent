import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.realtime.protocol import RealtimeVoiceConfig
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
        await session.handle_provider_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "input_1",
                "transcript": "Use my voice research skill.",
            }
        )
        assert agent._cached_system_prompt == frozen_prompt
        assert json.dumps(agent.tools, sort_keys=True) == frozen_tools
        assert sideband.sent[-1]["type"] == "response.create"

        await session.handle_provider_event(
            {
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
            }
        )
        await session._tool_task

        durable = db.get_tool_result_by_call_id(
            "voice-session", "tool_call_1"
        )
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

        await session.handle_provider_event(
            {
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
            }
        )

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
        await session.handle_provider_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "input_1",
                "transcript": "Track one item.",
            }
        )
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
        sideband=_FakeSideband(),
        call_id="call_transcript",
    )
    await session.start()
    try:
        await session.handle_provider_event(
            {
                "type": "input_audio_buffer.committed",
                "item_id": "input_slow",
            }
        )
        await _wait_until(lambda: bool(session.sideband.sent))

        user_rows = [
            message
            for message in db.get_messages("transcript-timeout")
            if message["role"] == "user"
        ]
        assert len(user_rows) == 1
        assert "transcript unavailable" in user_rows[0]["content"].lower()
        assert session.sideband.sent[-1]["type"] == "response.create"

        await session.handle_provider_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "input_slow",
                "transcript": "This arrived too late.",
            }
        )
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
        await session.handle_provider_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "input_1",
                "transcript": "Create both todos.",
            }
        )
        await session.handle_provider_event(
            {"type": "input_audio_buffer.speech_started"}
        )
        await session.handle_provider_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "input_2",
                "transcript": "Actually, do neither.",
            }
        )
        await session.handle_provider_event(
            {
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
            }
        )
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
        assert [message["content"] for message in messages if message["role"] == "user"] == [
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
        await session.handle_provider_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "input_1",
                "transcript": "Read both skills.",
            }
        )
        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            await session.handle_provider_event(
                {
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
                }
            )
            tool_task = session._tool_task
            assert tool_task is not None
            assert await asyncio.to_thread(started.wait, 2)
            await session.handle_provider_event(
                {"type": "input_audio_buffer.speech_started"}
            )
            await session.handle_provider_event(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "input_2",
                    "transcript": "Only use the first result.",
                }
            )
            release.set()
            await tool_task

        assert executed == ["skill_1"]
        messages = db.get_messages("in-tool-barge-in")
        second = next(
            message
            for message in messages
            if message.get("tool_call_id") == "skill_2"
        )
        assert "skipped" in second["content"].lower()
        assert "Only use the first result." in second["content"]
        assert len([message for message in messages if message["role"] == "user"]) == 1
        assert sideband.sent[-1]["type"] == "response.create"
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
            calls.append(
                {
                    "last": snapshot[-1]["content"],
                    "memory": review_memory,
                    "skills": review_skills,
                }
            )
            if len(calls) == 1:
                first_started.set()
                release.wait(timeout=5)
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

            second_boundary = db.append_message(
                "review-coalesce", role="assistant", content="second"
            )
            session._schedule_background_review_sync(
                messages_snapshot=db.get_messages("review-coalesce"),
                review_skills=True,
            )
            release.set()
            await asyncio.to_thread(session._review_thread.join, 5)

        assert calls == [
            {"last": "first", "memory": True, "skills": False},
            {"last": "second", "memory": False, "skills": True},
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
