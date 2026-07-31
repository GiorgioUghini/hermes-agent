"""OpenAI Realtime voice runtime owned by the Hermes gateway."""

from gateway.realtime.protocol import (
    CONTROL_PROTOCOL_VERSION,
    RealtimeProtocolError,
    RealtimeVoiceConfig,
)

__all__ = [
    "CONTROL_PROTOCOL_VERSION",
    "RealtimeProtocolError",
    "RealtimeVoiceConfig",
]
