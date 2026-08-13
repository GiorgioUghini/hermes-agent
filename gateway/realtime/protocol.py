"""Validated protocol contracts for Hermes native realtime voice sessions."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import hmac
import json
import re
from time import time
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import parse_qs, urlparse
import uuid


CONTROL_PROTOCOL_VERSION = 1
MAX_CONTROL_COMMAND_BYTES = 64 * 1024
MAX_TOOL_COUNT = 128
MAX_DEVICE_CAPABILITIES = 32
AUDIO_UPLINK_SAMPLE_RATE_HZ = 24_000
AUDIO_UPLINK_CHANNELS = 1
AUDIO_UPLINK_SAMPLE_WIDTH_BYTES = 2
AUDIO_UPLINK_MIN_MILLISECONDS = 100
AUDIO_UPLINK_APPEND_CHUNK_MILLISECONDS = 80
MAX_AUDIO_UPLINK_PREROLL_MILLISECONDS = 30_000
# Backward-compatible names for integrations that imported the original
# pre-roll format constants. The transport itself is now streaming-only.
PREROLL_SAMPLE_RATE_HZ = AUDIO_UPLINK_SAMPLE_RATE_HZ
PREROLL_CHANNELS = AUDIO_UPLINK_CHANNELS
PREROLL_SAMPLE_WIDTH_BYTES = AUDIO_UPLINK_SAMPLE_WIDTH_BYTES
PREROLL_MIN_MILLISECONDS = AUDIO_UPLINK_MIN_MILLISECONDS
PREROLL_APPEND_CHUNK_MILLISECONDS = AUDIO_UPLINK_APPEND_CHUNK_MILLISECONDS
MAX_INTERRUPT_AUDIO_END_MS = 60 * 60 * 1000

SESSION_STATES = frozenset({
    "negotiating",
    "connecting",
    "ready",
    "listening",
    "responding",
    "tool_wait",
    "rotating",
    "suspended",
    "closing",
    "closed",
    "degraded",
})

CONTROL_COMMAND_TYPES = frozenset({
    "approval.respond",
    "clarification.respond",
    "device.action.result",
    "response.interrupt",
    "response.playback_completed",
    "secret.respond",
    "session.close",
    "session.ping",
})

APPROVAL_CHOICES = frozenset({"once", "session", "always", "deny"})

_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")
_CAPABILITY_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DEVICE_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_VOICE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_DEFAULT_CALL_URL = "https://api.openai.com/v1/realtime/calls"
_DEFAULT_SIDEBAND_URL = "wss://api.openai.com/v1/realtime"


class RealtimeProtocolError(ValueError):
    """A client or provider payload violated the realtime contract."""

    def __init__(self, message: str, *, code: str = "invalid_realtime_payload"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class DeviceAdvertisement:
    """Strict client advertisement before server-owned catalog intersection."""

    protocol_version: int
    instance_id: str
    capabilities: tuple[tuple[str, int], ...]


def validate_device_advertisement(raw: Any) -> Optional[DeviceAdvertisement]:
    """Validate optional device metadata without accepting any client schemas."""

    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RealtimeProtocolError(
            "device must be an object",
            code="invalid_device_advertisement",
        )
    unknown = set(raw) - {"protocol_version", "instance_id", "capabilities"}
    if unknown:
        raise RealtimeProtocolError(
            f"device contains unknown fields: {sorted(unknown)}",
            code="invalid_device_advertisement",
        )
    protocol_version = raw.get("protocol_version")
    if isinstance(protocol_version, bool) or not isinstance(protocol_version, int):
        raise RealtimeProtocolError(
            "device.protocol_version must be an integer",
            code="invalid_device_protocol_version",
        )
    from gateway.realtime.device_actions import DEVICE_CAPABILITY_CATALOG

    if protocol_version != DEVICE_CAPABILITY_CATALOG.protocol_version:
        raise RealtimeProtocolError(
            f"Unsupported device protocol version: {protocol_version}",
            code="unsupported_device_protocol_version",
        )
    instance_id = str(raw.get("instance_id") or "").strip()
    try:
        canonical_instance_id = str(uuid.UUID(instance_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise RealtimeProtocolError(
            "device.instance_id must be a canonical installation UUID",
            code="invalid_device_instance_id",
        ) from exc
    if canonical_instance_id != instance_id.lower():
        raise RealtimeProtocolError(
            "device.instance_id must use canonical UUID notation",
            code="invalid_device_instance_id",
        )
    raw_capabilities = raw.get("capabilities")
    if not isinstance(raw_capabilities, list):
        raise RealtimeProtocolError(
            "device.capabilities must be an array",
            code="invalid_device_capabilities",
        )
    if len(raw_capabilities) > MAX_DEVICE_CAPABILITIES:
        raise RealtimeProtocolError(
            f"device.capabilities supports at most {MAX_DEVICE_CAPABILITIES} entries",
            code="too_many_device_capabilities",
        )
    capabilities = []
    seen_names = set()
    for raw_capability in raw_capabilities:
        if (
            not isinstance(raw_capability, Mapping)
            or set(raw_capability) != {"name", "version"}
        ):
            raise RealtimeProtocolError(
                "Each device capability must contain only name and version",
                code="invalid_device_capability",
            )
        name = raw_capability.get("name")
        version = raw_capability.get("version")
        if not isinstance(name, str) or not _CAPABILITY_NAME_RE.fullmatch(name):
            raise RealtimeProtocolError(
                "Device capability name is invalid",
                code="invalid_device_capability_name",
            )
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or not 1 <= version <= 1_000_000
        ):
            raise RealtimeProtocolError(
                "Device capability version is invalid",
                code="invalid_device_capability_version",
            )
        if name in seen_names:
            raise RealtimeProtocolError(
                "A device capability may advertise only one version",
                code="duplicate_device_capability",
            )
        seen_names.add(name)
        capabilities.append((name, version))
    return DeviceAdvertisement(
        protocol_version=protocol_version,
        instance_id=canonical_instance_id,
        capabilities=tuple(capabilities),
    )


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clean_string(
    value: Any,
    *,
    field: str,
    default: str = "",
    max_length: int = 200,
) -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    if len(text) > max_length or any(ch in text for ch in ("\r", "\n", "\x00")):
        raise RealtimeProtocolError(f"{field} is invalid", code=f"invalid_{field}")
    return text


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, parsed))


def _bounded_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, parsed))


def _validated_transport_url(
    value: Any,
    *,
    field: str,
    default: str,
    required_scheme: str,
) -> str:
    text = _clean_string(value, field=field, default=default, max_length=2048)
    parsed = urlparse(text)
    try:
        hostname = parsed.hostname
        parsed.port
    except ValueError as exc:
        raise RealtimeProtocolError(
            f"realtime_voice.transport.{field} is invalid",
            code=f"invalid_{field}",
        ) from exc
    if (
        parsed.scheme.lower() != required_scheme
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.fragment)
    ):
        raise RealtimeProtocolError(
            (
                f"realtime_voice.transport.{field} must use "
                f"{required_scheme} with a host and no embedded "
                "credentials or fragment"
            ),
            code=f"invalid_{field}",
        )
    return text


def derive_safety_identifier(server_secret: str, profile_name: str = "") -> str:
    """Derive a stable, non-reversible OpenAI safety identifier."""

    if not server_secret:
        raise RealtimeProtocolError(
            "A server authentication secret is required",
            code="missing_safety_identifier_secret",
        )
    scope = f"hermes-realtime-voice\0{profile_name or 'default'}".encode("utf-8")
    return hmac.new(server_secret.encode("utf-8"), scope, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class RealtimeVoiceConfig:
    """Normalized non-secret configuration for one realtime runtime."""

    enabled: bool = False
    model: str = "gpt-realtime"
    voice: str = "marin"
    transcription_model: str = "gpt-4o-mini-transcribe"
    vad_type: str = "server_vad"
    vad_threshold: float = 0.5
    vad_prefix_padding_ms: int = 300
    vad_silence_duration_ms: int = 500
    intermediate_speech_enabled: bool = True
    intermediate_speech_delay_seconds: float = 2.5
    max_active_sessions: int = 4
    max_creations_per_minute: int = 10
    max_sdp_bytes: int = 64 * 1024
    max_control_event_bytes: int = MAX_CONTROL_COMMAND_BYTES
    idle_timeout_seconds: int = 15 * 60
    provider_call_max_seconds: int = 55 * 60
    provider_call_max_input_tokens: int = 24_000
    reconnect_grace_seconds: int = 30
    request_timeout_seconds: float = 20.0
    connect_timeout_seconds: float = 10.0
    transcription_timeout_seconds: float = 5.0
    audio_uplink_enabled: bool = True
    audio_uplink_max_seconds: int = 30
    audio_uplink_idle_timeout_seconds: float = 5.0
    audio_uplink_append_chunk_milliseconds: int = (
        AUDIO_UPLINK_APPEND_CHUNK_MILLISECONDS
    )
    call_url: str = _DEFAULT_CALL_URL
    sideband_url: str = _DEFAULT_SIDEBAND_URL
    approval_timeout_seconds: int = 10 * 60
    control_buffer_events: int = 256
    control_subscriber_queue_events: int = 128
    history_message_limit: int = 40
    device_action_timeout_cap_seconds: float = 30.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "RealtimeVoiceConfig":
        root = _as_mapping(config)
        raw = _as_mapping(root.get("realtime_voice", root))
        vad = _as_mapping(raw.get("turn_detection"))
        interim = _as_mapping(raw.get("intermediate_speech"))
        legacy_preroll = _as_mapping(raw.get("preroll"))
        audio_uplink = _as_mapping(raw.get("audio_uplink"))
        device_actions = _as_mapping(raw.get("device_actions"))
        limits = _as_mapping(raw.get("limits"))
        transport = _as_mapping(raw.get("transport"))

        model = _clean_string(
            raw.get("model"),
            field="model",
            default=cls.model,
            max_length=200,
        )
        voice = _clean_string(
            raw.get("voice"),
            field="voice",
            default=cls.voice,
            max_length=64,
        )
        if not _VOICE_RE.fullmatch(voice):
            raise RealtimeProtocolError("voice is invalid", code="invalid_voice")
        transcription_model = _clean_string(
            raw.get("transcription_model"),
            field="transcription_model",
            default=cls.transcription_model,
            max_length=200,
        )
        vad_type = _clean_string(
            vad.get("type"),
            field="turn_detection_type",
            default=cls.vad_type,
            max_length=32,
        )
        if vad_type not in {"server_vad", "semantic_vad"}:
            raise RealtimeProtocolError(
                "turn_detection.type must be server_vad or semantic_vad",
                code="invalid_turn_detection",
            )

        return cls(
            enabled=bool(raw.get("enabled", cls.enabled)),
            model=model,
            voice=voice,
            transcription_model=transcription_model,
            vad_type=vad_type,
            vad_threshold=_bounded_float(
                vad.get("threshold"), cls.vad_threshold, 0.0, 1.0
            ),
            vad_prefix_padding_ms=_bounded_int(
                vad.get("prefix_padding_ms"), cls.vad_prefix_padding_ms, 0, 5000
            ),
            vad_silence_duration_ms=_bounded_int(
                vad.get("silence_duration_ms"),
                cls.vad_silence_duration_ms,
                100,
                10000,
            ),
            intermediate_speech_enabled=bool(
                interim.get("enabled", cls.intermediate_speech_enabled)
            ),
            intermediate_speech_delay_seconds=_bounded_float(
                interim.get("delay_seconds"),
                cls.intermediate_speech_delay_seconds,
                0.5,
                30.0,
            ),
            max_active_sessions=_bounded_int(
                limits.get("max_active_sessions"),
                cls.max_active_sessions,
                1,
                100,
            ),
            max_creations_per_minute=_bounded_int(
                limits.get("max_creations_per_minute"),
                cls.max_creations_per_minute,
                1,
                1000,
            ),
            max_sdp_bytes=_bounded_int(
                limits.get("max_sdp_bytes"),
                cls.max_sdp_bytes,
                4096,
                1024 * 1024,
            ),
            max_control_event_bytes=_bounded_int(
                limits.get("max_control_event_bytes"),
                cls.max_control_event_bytes,
                4096,
                1024 * 1024,
            ),
            idle_timeout_seconds=_bounded_int(
                limits.get("idle_timeout_seconds"),
                cls.idle_timeout_seconds,
                30,
                24 * 60 * 60,
            ),
            provider_call_max_seconds=_bounded_int(
                limits.get("provider_call_max_seconds"),
                cls.provider_call_max_seconds,
                60,
                60 * 60,
            ),
            provider_call_max_input_tokens=_bounded_int(
                limits.get("provider_call_max_input_tokens"),
                cls.provider_call_max_input_tokens,
                1_000,
                1_000_000,
            ),
            reconnect_grace_seconds=_bounded_int(
                transport.get("reconnect_grace_seconds"),
                cls.reconnect_grace_seconds,
                0,
                300,
            ),
            request_timeout_seconds=_bounded_float(
                transport.get("request_timeout_seconds"),
                cls.request_timeout_seconds,
                1.0,
                120.0,
            ),
            connect_timeout_seconds=_bounded_float(
                transport.get("connect_timeout_seconds"),
                cls.connect_timeout_seconds,
                1.0,
                120.0,
            ),
            transcription_timeout_seconds=_bounded_float(
                transport.get("transcription_timeout_seconds"),
                cls.transcription_timeout_seconds,
                0.5,
                30.0,
            ),
            audio_uplink_enabled=bool(
                audio_uplink.get(
                    "enabled",
                    legacy_preroll.get("enabled", cls.audio_uplink_enabled),
                )
            ),
            audio_uplink_max_seconds=_bounded_int(
                audio_uplink.get(
                    "max_seconds",
                    legacy_preroll.get("max_seconds"),
                ),
                cls.audio_uplink_max_seconds,
                1,
                120,
            ),
            audio_uplink_idle_timeout_seconds=_bounded_float(
                audio_uplink.get("idle_timeout_seconds"),
                cls.audio_uplink_idle_timeout_seconds,
                1.0,
                30.0,
            ),
            audio_uplink_append_chunk_milliseconds=_bounded_int(
                audio_uplink.get("append_chunk_milliseconds"),
                cls.audio_uplink_append_chunk_milliseconds,
                20,
                250,
            ),
            call_url=_validated_transport_url(
                transport.get("call_url"),
                field="call_url",
                default=cls.call_url,
                required_scheme="https",
            ),
            sideband_url=_validated_transport_url(
                transport.get("sideband_url"),
                field="sideband_url",
                default=cls.sideband_url,
                required_scheme="wss",
            ),
            approval_timeout_seconds=_bounded_int(
                limits.get("approval_timeout_seconds"),
                cls.approval_timeout_seconds,
                30,
                60 * 60,
            ),
            control_buffer_events=_bounded_int(
                limits.get("control_buffer_events"),
                cls.control_buffer_events,
                16,
                4096,
            ),
            control_subscriber_queue_events=_bounded_int(
                limits.get("control_subscriber_queue_events"),
                cls.control_subscriber_queue_events,
                8,
                1024,
            ),
            history_message_limit=_bounded_int(
                limits.get("history_message_limit"),
                cls.history_message_limit,
                4,
                200,
            ),
            device_action_timeout_cap_seconds=_bounded_float(
                device_actions.get("timeout_cap_seconds"),
                cls.device_action_timeout_cap_seconds,
                1.0,
                120.0,
            ),
        )

    @property
    def audio_uplink_max_bytes(self) -> int:
        return (
            self.audio_uplink_max_seconds
            * AUDIO_UPLINK_SAMPLE_RATE_HZ
            * AUDIO_UPLINK_CHANNELS
            * AUDIO_UPLINK_SAMPLE_WIDTH_BYTES
        )

    def turn_detection_config(self) -> dict[str, Any]:
        turn_detection: dict[str, Any] = {
            "type": self.vad_type,
            "create_response": False,
            "interrupt_response": True,
        }
        if self.vad_type == "server_vad":
            turn_detection.update({
                "threshold": self.vad_threshold,
                "prefix_padding_ms": self.vad_prefix_padding_ms,
                "silence_duration_ms": self.vad_silence_duration_ms,
            })
        return turn_detection

    def openai_session(
        self,
        *,
        instructions: str,
        tools: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build the immutable OpenAI Realtime session configuration."""

        return {
            "type": "realtime",
            "model": self.model,
            "instructions": instructions,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": AUDIO_UPLINK_SAMPLE_RATE_HZ,
                    },
                    "transcription": {"model": self.transcription_model},
                    "turn_detection": self.turn_detection_config(),
                },
                "output": {"voice": self.voice},
            },
            "tools": flatten_realtime_tools(tools),
            "tool_choice": "auto",
        }


@dataclass(frozen=True)
class RealtimeCall:
    """The result of creating an OpenAI WebRTC call."""

    answer_sdp: str
    call_id: str


@dataclass(frozen=True)
class RealtimeFunctionCall:
    """A complete function call emitted by a Realtime response."""

    call_id: str
    name: str
    arguments: str
    item_id: str = ""
    response_id: str = ""


def validate_audio_stream_id(value: Any) -> str:
    stream_id = str(value or "").strip()
    try:
        normalized = str(uuid.UUID(stream_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise RealtimeProtocolError(
            "stream_id must be a UUID",
            code="invalid_audio_stream_id",
        ) from exc
    if normalized != stream_id.lower():
        raise RealtimeProtocolError(
            "stream_id must use canonical UUID notation",
            code="invalid_audio_stream_id",
        )
    return normalized


def validate_audio_uplink_command(
    raw: Any,
    *,
    expected_stream_id: str,
) -> dict[str, Any]:
    """Validate one JSON control frame on an audio uplink socket."""

    if isinstance(raw, (bytes, bytearray)):
        if len(raw) > MAX_CONTROL_COMMAND_BYTES:
            raise RealtimeProtocolError(
                "Audio uplink command is too large",
                code="audio_uplink_command_too_large",
            )
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        if len(raw.encode("utf-8")) > MAX_CONTROL_COMMAND_BYTES:
            raise RealtimeProtocolError(
                "Audio uplink command is too large",
                code="audio_uplink_command_too_large",
            )
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RealtimeProtocolError(
                "Audio uplink command is not valid JSON",
                code="invalid_audio_uplink_json",
            ) from exc
    if not isinstance(raw, Mapping):
        raise RealtimeProtocolError(
            "Audio uplink command must be an object",
            code="invalid_audio_uplink_command",
        )

    command_type = str(raw.get("type") or "").strip()
    if command_type == "turn.end":
        return {"type": "turn.end"}
    if command_type != "turn.begin":
        raise RealtimeProtocolError(
            f"Unsupported audio uplink command: {command_type or '(missing)'}",
            code="unsupported_audio_uplink_command",
        )

    stream_id = validate_audio_stream_id(raw.get("stream_id"))
    if stream_id != expected_stream_id:
        raise RealtimeProtocolError(
            "turn.begin stream_id does not match the uplink URL",
            code="audio_stream_id_mismatch",
        )
    barge_in = raw.get("barge_in")
    if not isinstance(barge_in, bool):
        raise RealtimeProtocolError(
            "turn.begin barge_in must be a boolean",
            code="invalid_audio_barge_in",
        )
    preroll_ms = raw.get("preroll_ms")
    if (
        isinstance(preroll_ms, bool)
        or not isinstance(preroll_ms, int)
        or not 0 <= preroll_ms <= MAX_AUDIO_UPLINK_PREROLL_MILLISECONDS
    ):
        raise RealtimeProtocolError(
            "turn.begin preroll_ms must be an integer between 0 and 30000",
            code="invalid_audio_preroll_ms",
        )

    interrupted = raw.get("interrupted")
    if not barge_in:
        if "interrupted" in raw:
            raise RealtimeProtocolError(
                "interrupted must be absent when barge_in is false",
                code="unexpected_audio_interruption",
            )
        return {
            "type": "turn.begin",
            "stream_id": stream_id,
            "barge_in": False,
            "preroll_ms": preroll_ms,
            "interrupted": None,
        }
    if not isinstance(interrupted, Mapping):
        raise RealtimeProtocolError(
            "interrupted is required when barge_in is true",
            code="missing_audio_interruption",
        )

    response_id = str(interrupted.get("response_id") or "").strip()
    item_id = str(interrupted.get("item_id") or "").strip()
    if not _ID_RE.fullmatch(response_id):
        raise RealtimeProtocolError(
            "interrupted.response_id is invalid",
            code="invalid_response_id",
        )
    if not _ID_RE.fullmatch(item_id):
        raise RealtimeProtocolError(
            "interrupted.item_id is invalid",
            code="invalid_assistant_item_id",
        )
    audio_end_ms = interrupted.get("audio_end_ms")
    if (
        isinstance(audio_end_ms, bool)
        or not isinstance(audio_end_ms, int)
        or not 0 <= audio_end_ms <= MAX_INTERRUPT_AUDIO_END_MS
    ):
        raise RealtimeProtocolError(
            "interrupted.audio_end_ms must be an integer between 0 and 3600000",
            code="invalid_interrupt_audio_end_ms",
        )
    return {
        "type": "turn.begin",
        "stream_id": stream_id,
        "barge_in": True,
        "preroll_ms": preroll_ms,
        "interrupted": {
            "response_id": response_id,
            "item_id": item_id,
            "audio_end_ms": audio_end_ms,
        },
    }


def flatten_realtime_tools(
    tools: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Chat Completions tool wrappers to Realtime function schemas."""

    flattened: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_tool in tools:
        tool = _as_mapping(raw_tool)
        function = _as_mapping(tool.get("function")) or tool
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if name in seen:
            continue
        if len(flattened) >= MAX_TOOL_COUNT:
            raise RealtimeProtocolError(
                f"Realtime sessions support at most {MAX_TOOL_COUNT} tools",
                code="too_many_tools",
            )
        parameters = function.get("parameters")
        if not isinstance(parameters, Mapping):
            parameters = {"type": "object", "properties": {}}
        entry: dict[str, Any] = {
            "type": "function",
            "name": name,
            "parameters": dict(parameters),
        }
        description = function.get("description")
        if isinstance(description, str) and description.strip():
            entry["description"] = description.strip()
        flattened.append(entry)
        seen.add(name)
    return flattened


def extract_call_id(headers: Mapping[str, Any]) -> str:
    """Extract a provider call ID without trusting an arbitrary Location value."""

    for key in ("x-openai-call-id", "X-OpenAI-Call-Id"):
        raw = headers.get(key)
        if isinstance(raw, str) and _ID_RE.fullmatch(raw.strip()):
            return raw.strip()

    location = headers.get("location") or headers.get("Location")
    if isinstance(location, str) and location.strip():
        parsed = urlparse(location.strip())
        query = parse_qs(parsed.query)
        for key in ("call_id", "id"):
            values = query.get(key) or []
            if values and _ID_RE.fullmatch(values[0]):
                return values[0]
        candidate = parsed.path.rstrip("/").rsplit("/", 1)[-1]
        if _ID_RE.fullmatch(candidate):
            return candidate

    raise RealtimeProtocolError(
        "OpenAI did not return a valid call identifier",
        code="missing_call_id",
    )


def function_calls_from_event(event: Mapping[str, Any]) -> list[RealtimeFunctionCall]:
    """Return complete function calls from supported Realtime server events."""

    event_type = event.get("type")
    response_id = str(event.get("response_id") or "")
    items: list[Mapping[str, Any]] = []
    if event_type == "response.output_item.done":
        item = event.get("item")
        if isinstance(item, Mapping):
            items.append(item)
    elif event_type == "response.done":
        response = _as_mapping(event.get("response"))
        response_id = str(response.get("id") or response_id)
        output = response.get("output")
        if isinstance(output, list):
            items.extend(item for item in output if isinstance(item, Mapping))
    else:
        return []

    calls: list[RealtimeFunctionCall] = []
    for item in items:
        if item.get("type") != "function_call":
            continue
        call_id = str(item.get("call_id") or "").strip()
        name = str(item.get("name") or "").strip()
        if not _ID_RE.fullmatch(call_id) or not name:
            continue
        arguments = item.get("arguments", "")
        if isinstance(arguments, Mapping):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        elif not isinstance(arguments, str):
            arguments = str(arguments or "")
        calls.append(
            RealtimeFunctionCall(
                call_id=call_id,
                name=name,
                arguments=arguments,
                item_id=str(item.get("id") or ""),
                response_id=response_id,
            )
        )
    return calls


def response_transcript(event: Mapping[str, Any]) -> str:
    """Extract a finalized assistant transcript from a response.done event."""

    if event.get("type") != "response.done":
        return ""
    response = _as_mapping(event.get("response"))
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            text = part.get("transcript") or part.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "".join(parts).strip()


def function_call_output_event(
    call_id: str,
    output: Any,
    *,
    event_id: Optional[str] = None,
) -> dict[str, Any]:
    if not _ID_RE.fullmatch(call_id):
        raise RealtimeProtocolError("Invalid function call ID", code="invalid_call_id")
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False, default=str)
    event: dict[str, Any] = {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        },
    }
    if event_id:
        event["event_id"] = event_id
    return event


def conversation_function_call_event(
    call_id: str,
    name: str,
    arguments: Any,
    *,
    event_id: Optional[str] = None,
) -> dict[str, Any]:
    """Project one canonical Hermes assistant call into a new provider call."""

    if not _ID_RE.fullmatch(call_id):
        raise RealtimeProtocolError("Invalid function call ID", code="invalid_call_id")
    if not isinstance(name, str) or not name.strip():
        raise RealtimeProtocolError("Invalid function name", code="invalid_tool_name")
    if isinstance(arguments, Mapping):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    elif not isinstance(arguments, str):
        arguments = str(arguments or "")
    event: dict[str, Any] = {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call",
            "call_id": call_id,
            "name": name.strip(),
            "arguments": arguments,
        },
    }
    if event_id:
        event["event_id"] = event_id
    return event


def input_audio_buffer_clear_event(
    *,
    event_id: Optional[str] = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {"type": "input_audio_buffer.clear"}
    if event_id:
        event["event_id"] = event_id
    return event


def input_audio_buffer_append_event(
    audio: bytes,
    *,
    event_id: Optional[str] = None,
) -> dict[str, Any]:
    if not isinstance(audio, bytes) or not audio:
        raise RealtimeProtocolError(
            "Audio uplink chunk is empty",
            code="empty_audio_uplink",
        )
    event: dict[str, Any] = {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(audio).decode("ascii"),
    }
    if event_id:
        event["event_id"] = event_id
    return event


def input_audio_buffer_commit_event(
    *,
    event_id: Optional[str] = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {"type": "input_audio_buffer.commit"}
    if event_id:
        event["event_id"] = event_id
    return event


def response_create_event(
    *,
    instructions: Optional[str] = None,
    status_message: bool = False,
    event_id: Optional[str] = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {}
    if instructions:
        response["instructions"] = instructions
    if status_message:
        response.update({
            "conversation": "none",
            "output_modalities": ["audio"],
            "tools": [],
            "tool_choice": "none",
            "metadata": {"hermes_kind": "tool_wait_status"},
        })
    event: dict[str, Any] = {"type": "response.create", "response": response}
    if event_id:
        event["event_id"] = event_id
    return event


def response_cancel_event(
    *,
    response_id: Optional[str] = None,
    event_id: Optional[str] = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {"type": "response.cancel"}
    if response_id:
        event["response_id"] = response_id
    if event_id:
        event["event_id"] = event_id
    return event


def output_audio_buffer_clear_event(
    *,
    event_id: Optional[str] = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {"type": "output_audio_buffer.clear"}
    if event_id:
        event["event_id"] = event_id
    return event


def conversation_item_truncate_event(
    item_id: str,
    audio_end_ms: int,
    *,
    event_id: Optional[str] = None,
) -> dict[str, Any]:
    if not _ID_RE.fullmatch(str(item_id or "")):
        raise RealtimeProtocolError(
            "Invalid assistant item ID",
            code="invalid_assistant_item_id",
        )
    if (
        isinstance(audio_end_ms, bool)
        or not isinstance(audio_end_ms, int)
        or not 0 <= audio_end_ms <= MAX_INTERRUPT_AUDIO_END_MS
    ):
        raise RealtimeProtocolError(
            "audio_end_ms is outside the supported range",
            code="invalid_interrupt_audio_end_ms",
        )
    event: dict[str, Any] = {
        "type": "conversation.item.truncate",
        "item_id": item_id,
        "content_index": 0,
        "audio_end_ms": audio_end_ms,
    }
    if event_id:
        event["event_id"] = event_id
    return event


def conversation_message_event(
    role: str,
    content: str,
    *,
    event_id: Optional[str] = None,
) -> dict[str, Any]:
    if role not in {"user", "assistant"}:
        raise RealtimeProtocolError("Invalid conversation role", code="invalid_role")
    event: dict[str, Any] = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": role,
            "content": [
                {
                    "type": "input_text" if role == "user" else "output_text",
                    "text": str(content),
                }
            ],
        },
    }
    if event_id:
        event["event_id"] = event_id
    return event


def control_event(
    *,
    sequence: int,
    session_id: str,
    stream_id: str = "",
    event_type: str,
    data: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    if sequence < 1:
        raise RealtimeProtocolError("Control sequence must be positive")
    if not _ID_RE.fullmatch(session_id):
        raise RealtimeProtocolError("Invalid session ID", code="invalid_session_id")
    event = {
        "version": CONTROL_PROTOCOL_VERSION,
        "sequence": sequence,
        "session_id": session_id,
        "type": event_type,
        "timestamp": time(),
        "data": dict(data or {}),
    }
    if stream_id:
        if not _ID_RE.fullmatch(stream_id):
            raise RealtimeProtocolError(
                "Invalid control stream ID",
                code="invalid_control_stream_id",
            )
        event["stream_id"] = stream_id
    return event


def validate_control_command(raw: Any) -> dict[str, Any]:
    """Validate an Android control command and return a normalized envelope."""

    if isinstance(raw, (bytes, bytearray)):
        if len(raw) > MAX_CONTROL_COMMAND_BYTES:
            raise RealtimeProtocolError(
                "Control command is too large", code="control_command_too_large"
            )
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        if len(raw.encode("utf-8")) > MAX_CONTROL_COMMAND_BYTES:
            raise RealtimeProtocolError(
                "Control command is too large", code="control_command_too_large"
            )
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RealtimeProtocolError(
                "Control command is not valid JSON", code="invalid_control_json"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RealtimeProtocolError(
            "Control command must be an object", code="invalid_control_command"
        )
    version = raw.get("version", CONTROL_PROTOCOL_VERSION)
    if version != CONTROL_PROTOCOL_VERSION:
        raise RealtimeProtocolError(
            f"Unsupported control protocol version: {version}",
            code="unsupported_control_version",
        )
    command_type = str(raw.get("type") or "").strip()
    if command_type not in CONTROL_COMMAND_TYPES:
        raise RealtimeProtocolError(
            f"Unsupported control command: {command_type or '(missing)'}",
            code="unsupported_control_command",
        )
    request_id = str(raw.get("request_id") or "").strip()
    if request_id and not _ID_RE.fullmatch(request_id):
        raise RealtimeProtocolError("request_id is invalid", code="invalid_request_id")
    data = raw.get("data")
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise RealtimeProtocolError(
            "data must be an object", code="invalid_control_data"
        )
    normalized = {
        "version": CONTROL_PROTOCOL_VERSION,
        "type": command_type,
        "request_id": request_id,
        "data": dict(data),
    }
    if command_type == "approval.respond":
        choice = str(data.get("choice") or "").strip().lower()
        if choice not in APPROVAL_CHOICES:
            raise RealtimeProtocolError(
                "approval choice must be once, session, always, or deny",
                code="invalid_approval_choice",
            )
        normalized["data"]["choice"] = choice
    elif command_type == "response.interrupt":
        if not request_id:
            raise RealtimeProtocolError(
                "response.interrupt requires request_id",
                code="invalid_request_id",
            )
        audio_end_ms = data.get("audio_end_ms")
        if (
            isinstance(audio_end_ms, bool)
            or not isinstance(audio_end_ms, int)
            or not 0 <= audio_end_ms <= MAX_INTERRUPT_AUDIO_END_MS
        ):
            raise RealtimeProtocolError(
                "audio_end_ms must be an integer between 0 and 3600000",
                code="invalid_interrupt_audio_end_ms",
            )
        normalized["data"] = {"audio_end_ms": audio_end_ms}
    elif command_type == "response.playback_completed":
        if not request_id:
            raise RealtimeProtocolError(
                "response.playback_completed requires request_id",
                code="invalid_request_id",
            )
        response_id = str(data.get("response_id") or "").strip()
        if not _ID_RE.fullmatch(response_id):
            raise RealtimeProtocolError(
                "response_id is invalid",
                code="invalid_response_id",
            )
        normalized["data"] = {"response_id": response_id}
    elif command_type == "device.action.result":
        if not request_id:
            raise RealtimeProtocolError(
                "device.action.result requires request_id",
                code="invalid_request_id",
            )
        invocation_id = str(data.get("invocation_id") or "").strip()
        if not _ID_RE.fullmatch(invocation_id):
            raise RealtimeProtocolError(
                "invocation_id is invalid",
                code="invalid_device_invocation_id",
            )
        status = str(data.get("status") or "").strip()
        if status == "ok":
            if set(data) != {"invocation_id", "status", "result"}:
                raise RealtimeProtocolError(
                    "Successful device results require only invocation_id, status, and result",
                    code="invalid_device_action_result",
                )
            result = data.get("result")
            if not isinstance(result, Mapping):
                raise RealtimeProtocolError(
                    "device action result must be an object",
                    code="invalid_device_action_result",
                )
            normalized["data"] = {
                "invocation_id": invocation_id,
                "status": "ok",
                "result": dict(result),
            }
        elif status == "error":
            if set(data) != {"invocation_id", "status", "error"}:
                raise RealtimeProtocolError(
                    "Failed device results require only invocation_id, status, and error",
                    code="invalid_device_action_error",
                )
            error = data.get("error")
            if not isinstance(error, Mapping) or set(error) != {
                "code",
                "message",
                "retryable",
            }:
                raise RealtimeProtocolError(
                    "device action error is invalid",
                    code="invalid_device_action_error",
                )
            error_code = str(error.get("code") or "").strip()
            message = str(error.get("message") or "").strip()
            retryable = error.get("retryable")
            if not _DEVICE_ERROR_CODE_RE.fullmatch(error_code):
                raise RealtimeProtocolError(
                    "device action error code is invalid",
                    code="invalid_device_action_error",
                )
            if (
                not 1 <= len(message) <= 1000
                or any(ch in message for ch in ("\r", "\n", "\x00"))
                or not isinstance(retryable, bool)
            ):
                raise RealtimeProtocolError(
                    "device action error details are invalid",
                    code="invalid_device_action_error",
                )
            normalized["data"] = {
                "invocation_id": invocation_id,
                "status": "error",
                "error": {
                    "code": error_code,
                    "message": message,
                    "retryable": retryable,
                },
            }
        else:
            raise RealtimeProtocolError(
                "device action status must be ok or error",
                code="invalid_device_action_status",
            )
    return normalized
