import base64
import json

import pytest

from gateway.realtime.controls import ControlEventBroker
from gateway.realtime.protocol import (
    RealtimeProtocolError,
    RealtimeVoiceConfig,
    PREROLL_SAMPLE_RATE_HZ,
    conversation_item_truncate_event,
    conversation_function_call_event,
    derive_safety_identifier,
    extract_call_id,
    flatten_realtime_tools,
    function_call_output_event,
    function_calls_from_event,
    input_audio_buffer_append_event,
    input_audio_buffer_clear_event,
    input_audio_buffer_commit_event,
    output_audio_buffer_clear_event,
    response_cancel_event,
    response_transcript,
    session_turn_detection_update_event,
    validate_preroll_idempotency_key,
    validate_control_command,
)


def test_config_builds_hermes_owned_vad_session():
    config = RealtimeVoiceConfig.from_config({
        "realtime_voice": {
            "enabled": True,
            "model": "gpt-realtime",
            "voice": "marin",
            "turn_detection": {"silence_duration_ms": 750},
            "preroll": {"max_seconds": 12, "timeout_seconds": 9},
        }
    })

    payload = config.openai_session(
        instructions="frozen prompt",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "skill_view",
                    "description": "Read one skill.",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
            }
        ],
    )

    assert payload["instructions"] == "frozen prompt"
    assert payload["output_modalities"] == ["audio"]
    assert payload["audio"]["input"]["turn_detection"]["create_response"] is False
    assert payload["audio"]["input"]["turn_detection"]["interrupt_response"] is True
    assert payload["audio"]["input"]["turn_detection"]["silence_duration_ms"] == 750
    assert payload["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": PREROLL_SAMPLE_RATE_HZ,
    }
    assert config.preroll_max_seconds == 12
    assert config.preroll_timeout_seconds == 9
    assert config.preroll_max_bytes == 12 * PREROLL_SAMPLE_RATE_HZ * 2
    assert payload["tools"][0]["name"] == "skill_view"
    assert "function" not in payload["tools"][0]


def test_preroll_events_are_typed_and_server_owned():
    audio = b"\x01\x02" * 2400
    append = input_audio_buffer_append_event(audio, event_id="append_1")
    assert append["type"] == "input_audio_buffer.append"
    assert base64.b64decode(append["audio"]) == audio
    assert input_audio_buffer_clear_event(event_id="clear_1") == {
        "type": "input_audio_buffer.clear",
        "event_id": "clear_1",
    }
    assert input_audio_buffer_commit_event(event_id="commit_1") == {
        "type": "input_audio_buffer.commit",
        "event_id": "commit_1",
    }
    assert output_audio_buffer_clear_event(event_id="output_clear_1") == {
        "type": "output_audio_buffer.clear",
        "event_id": "output_clear_1",
    }
    assert (
        session_turn_detection_update_event(None)["session"]["audio"]["input"][
            "turn_detection"
        ]
        is None
    )
    assert validate_preroll_idempotency_key("wake:123") == "wake:123"
    with pytest.raises(RealtimeProtocolError, match="Idempotency-Key"):
        validate_preroll_idempotency_key("")


def test_config_accepts_secure_openai_compatible_proxy_endpoints():
    config = RealtimeVoiceConfig.from_config({
        "realtime_voice": {
            "transport": {
                "call_url": "https://proxy.example/v1/realtime/calls?route=stock",
                "sideband_url": "wss://proxy.example/v1/realtime?route=stock",
            }
        }
    })

    assert config.call_url == ("https://proxy.example/v1/realtime/calls?route=stock")
    assert config.sideband_url == ("wss://proxy.example/v1/realtime?route=stock")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("call_url", "http://proxy.example/v1/realtime/calls"),
        ("call_url", "https://user:secret@proxy.example/v1/realtime/calls"),
        ("sideband_url", "ws://proxy.example/v1/realtime"),
        ("sideband_url", "wss://proxy.example/v1/realtime#fragment"),
    ],
)
def test_config_rejects_insecure_or_credentialed_proxy_endpoints(field, value):
    with pytest.raises(RealtimeProtocolError, match=field):
        RealtimeVoiceConfig.from_config({
            "realtime_voice": {"transport": {field: value}}
        })


def test_safety_identifier_is_stable_scoped_and_non_reversible():
    first = derive_safety_identifier("gateway-secret", "work")
    assert first == derive_safety_identifier("gateway-secret", "work")
    assert first != derive_safety_identifier("gateway-secret", "personal")
    assert "gateway-secret" not in first
    assert len(first) == 64

    with pytest.raises(RealtimeProtocolError, match="authentication secret"):
        derive_safety_identifier("")


def test_tool_flattening_deduplicates_names_and_rejects_unbounded_surface():
    tools = [
        {"name": "one", "parameters": {"type": "object"}},
        {
            "type": "function",
            "function": {"name": "one", "parameters": {"type": "object"}},
        },
        {"type": "function", "function": {"name": "two"}},
    ]

    assert [tool["name"] for tool in flatten_realtime_tools(tools)] == [
        "one",
        "two",
    ]
    assert flatten_realtime_tools(tools)[1]["parameters"]["type"] == "object"


def test_call_id_is_constrained_to_header_location_or_query():
    assert extract_call_id({"X-OpenAI-Call-Id": "call_123"}) == "call_123"
    assert (
        extract_call_id({
            "Location": "https://api.openai.com/v1/realtime/calls/call_456"
        })
        == "call_456"
    )
    assert extract_call_id({"Location": "/v1/realtime?call_id=call_789"}) == "call_789"
    with pytest.raises(RealtimeProtocolError, match="call identifier"):
        extract_call_id({"Location": "https://example.invalid/"})


def test_function_calls_and_transcript_parse_from_response_done():
    event = {
        "type": "response.done",
        "response": {
            "id": "resp_1",
            "output": [
                {
                    "type": "function_call",
                    "id": "item_1",
                    "call_id": "call_1",
                    "name": "skill_view",
                    "arguments": '{"name":"research"}',
                },
                {
                    "type": "message",
                    "content": [{"type": "output_audio", "transcript": "I found it."}],
                },
            ],
        },
    }

    calls = function_calls_from_event(event)
    assert calls[0].call_id == "call_1"
    assert calls[0].response_id == "resp_1"
    assert json.loads(calls[0].arguments) == {"name": "research"}
    assert response_transcript(event) == "I found it."
    assert function_call_output_event("call_1", {"ok": True})["item"]["output"]
    replay = conversation_function_call_event(
        "call_1",
        "skill_view",
        {"name": "research"},
        event_id="event_1",
    )
    assert replay["item"] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "skill_view",
        "arguments": '{"name":"research"}',
    }


def test_control_commands_require_structured_authorization():
    command = validate_control_command({
        "version": 1,
        "type": "approval.respond",
        "request_id": "req_1",
        "data": {"approval_id": "approval_1", "choice": "once"},
    })
    assert command["data"]["choice"] == "once"

    secret = validate_control_command({
        "type": "secret.respond",
        "data": {"prompt_id": "secret_1", "value": "not-logged"},
    })
    assert secret["type"] == "secret.respond"

    with pytest.raises(RealtimeProtocolError, match="approval choice"):
        validate_control_command({
            "type": "approval.respond",
            "data": {"choice": "spoken-yes"},
        })


def test_interrupt_command_requires_playout_position_and_correlation_id():
    command = validate_control_command({
        "type": "response.interrupt",
        "request_id": "wake-2",
        "data": {"audio_end_ms": 875},
    })

    assert command["data"] == {"audio_end_ms": 875}
    assert response_cancel_event(
        response_id="resp_1",
        event_id="event_1",
    ) == {
        "type": "response.cancel",
        "response_id": "resp_1",
        "event_id": "event_1",
    }
    assert conversation_item_truncate_event(
        "item_1",
        875,
        event_id="event_2",
    ) == {
        "type": "conversation.item.truncate",
        "item_id": "item_1",
        "content_index": 0,
        "audio_end_ms": 875,
        "event_id": "event_2",
    }

    with pytest.raises(RealtimeProtocolError) as missing_request:
        validate_control_command({
            "type": "response.interrupt",
            "data": {"audio_end_ms": 875},
        })
    assert missing_request.value.code == "invalid_request_id"

    with pytest.raises(RealtimeProtocolError) as invalid_position:
        validate_control_command({
            "type": "response.interrupt",
            "request_id": "wake-3",
            "data": {"audio_end_ms": -1},
        })
    assert invalid_position.value.code == "invalid_interrupt_audio_end_ms"


def test_playback_completion_requires_response_and_correlation_ids():
    command = validate_control_command({
        "type": "response.playback_completed",
        "request_id": "played-1",
        "data": {"response_id": "response_1"},
    })

    assert command["data"] == {"response_id": "response_1"}

    with pytest.raises(RealtimeProtocolError) as missing_request:
        validate_control_command({
            "type": "response.playback_completed",
            "data": {"response_id": "response_1"},
        })
    assert missing_request.value.code == "invalid_request_id"

    with pytest.raises(RealtimeProtocolError) as missing_response:
        validate_control_command({
            "type": "response.playback_completed",
            "request_id": "played-2",
            "data": {},
        })
    assert missing_response.value.code == "invalid_response_id"


def test_control_broker_replays_sequence_and_marks_expired_cursor():
    broker = ControlEventBroker("session_1", buffer_events=2, subscriber_queue_events=2)
    broker.publish("session.state", {"state": "ready"})
    broker.publish("tool.started", {"name": "one"})
    broker.publish("tool.completed", {"name": "one"})

    subscription = broker.subscribe(after_sequence=0)

    assert subscription.cursor_expired is False
    assert [event["sequence"] for event in subscription.backlog] == [2, 3]
    assert {event["stream_id"] for event in subscription.backlog} == {broker.stream_id}
    assert ControlEventBroker("session_1").stream_id != broker.stream_id
    expired = broker.subscribe(after_sequence=1)
    assert expired.cursor_expired is False
    very_old = broker.subscribe(after_sequence=-1)
    assert very_old.cursor_expired is True
