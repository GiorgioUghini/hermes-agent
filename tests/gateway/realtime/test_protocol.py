import json

import pytest

from gateway.realtime.controls import ControlEventBroker
from gateway.realtime.protocol import (
    RealtimeProtocolError,
    RealtimeVoiceConfig,
    conversation_function_call_event,
    derive_safety_identifier,
    extract_call_id,
    flatten_realtime_tools,
    function_call_output_event,
    function_calls_from_event,
    response_transcript,
    validate_control_command,
)


def test_config_builds_hermes_owned_vad_session():
    config = RealtimeVoiceConfig.from_config(
        {
            "realtime_voice": {
                "enabled": True,
                "model": "gpt-realtime",
                "voice": "marin",
                "turn_detection": {"silence_duration_ms": 750},
            }
        }
    )

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
    assert payload["tools"][0]["name"] == "skill_view"
    assert "function" not in payload["tools"][0]


def test_config_accepts_secure_openai_compatible_proxy_endpoints():
    config = RealtimeVoiceConfig.from_config(
        {
            "realtime_voice": {
                "transport": {
                    "call_url": "https://proxy.example/v1/realtime/calls?route=stock",
                    "sideband_url": "wss://proxy.example/v1/realtime?route=stock",
                }
            }
        }
    )

    assert config.call_url == (
        "https://proxy.example/v1/realtime/calls?route=stock"
    )
    assert config.sideband_url == (
        "wss://proxy.example/v1/realtime?route=stock"
    )


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
        RealtimeVoiceConfig.from_config(
            {"realtime_voice": {"transport": {field: value}}}
        )


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
        extract_call_id({"Location": "https://api.openai.com/v1/realtime/calls/call_456"})
        == "call_456"
    )
    assert (
        extract_call_id({"Location": "/v1/realtime?call_id=call_789"})
        == "call_789"
    )
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
                    "content": [
                        {"type": "output_audio", "transcript": "I found it."}
                    ],
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
    command = validate_control_command(
        {
            "version": 1,
            "type": "approval.respond",
            "request_id": "req_1",
            "data": {"approval_id": "approval_1", "choice": "once"},
        }
    )
    assert command["data"]["choice"] == "once"

    secret = validate_control_command(
        {
            "type": "secret.respond",
            "data": {"prompt_id": "secret_1", "value": "not-logged"},
        }
    )
    assert secret["type"] == "secret.respond"

    with pytest.raises(RealtimeProtocolError, match="approval choice"):
        validate_control_command(
            {"type": "approval.respond", "data": {"choice": "spoken-yes"}}
        )


def test_control_broker_replays_sequence_and_marks_expired_cursor():
    broker = ControlEventBroker("session_1", buffer_events=2, subscriber_queue_events=2)
    broker.publish("session.state", {"state": "ready"})
    broker.publish("tool.started", {"name": "one"})
    broker.publish("tool.completed", {"name": "one"})

    subscription = broker.subscribe(after_sequence=0)

    assert subscription.cursor_expired is False
    assert [event["sequence"] for event in subscription.backlog] == [2, 3]
    expired = broker.subscribe(after_sequence=1)
    assert expired.cursor_expired is False
    very_old = broker.subscribe(after_sequence=-1)
    assert very_old.cursor_expired is True
