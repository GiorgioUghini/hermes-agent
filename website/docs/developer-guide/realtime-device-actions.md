---
title: "Realtime Device Actions"
description: "Android reverse-RPC contract for local device tools"
---

# Realtime device actions

Hermes Realtime sessions can expose a bounded set of tools that execute on the
Android device. The authenticated control WebSocket is the reverse-RPC
transport; the APK does not host an HTTP or MCP server.

Hermes owns all model-facing names, descriptions, parameter schemas, result
schemas, timeouts, and side-effect classifications. Android advertises only
capability names and versions, then validates and executes requests selected
from that trusted catalog.

The canonical machine-readable contract is
`gateway/realtime/device_capabilities.json`. Published capability versions are
immutable. Breaking changes require a new capability version.

## Discovery

`GET /v1/capabilities` reports whether the bridge is available:

```json
{
  "features": {
    "device_actions": {
      "available": true,
      "protocol_version": 1,
      "catalog_version": 1,
      "capabilities": {
        "timer": 1
      },
      "transport": "realtime_control_websocket",
      "delivery": "at_least_once"
    }
  }
}
```

The APK should advertise only versions it implements. Unknown names or versions
are ignored during intersection and never become model-visible tools.

## Session negotiation

Include `device` in the JSON SDP request when creating or renewing a session:

```json
{
  "sdp": "v=0\r\n...",
  "session_id": "optional-logical-session-id",
  "device": {
    "protocol_version": 1,
    "instance_id": "22222222-2222-4222-8222-222222222222",
    "capabilities": [
      {
        "name": "timer",
        "version": 1
      }
    ]
  }
}
```

`instance_id` is an installation-scoped UUID generated and stored by the APK.
It must not be derived from hardware, an Android advertising identifier, or
another cross-install identifier.

The session response returns only accepted capabilities:

```json
{
  "version": 1,
  "session_id": "rt_123",
  "call_id": "rtc_456",
  "sdp": "v=0\r\n...",
  "device": {
    "protocol_version": 1,
    "instance_id": "22222222-2222-4222-8222-222222222222",
    "capabilities": [
      {
        "name": "timer",
        "version": 1
      }
    ]
  }
}
```

Omitting `device` preserves the existing server-tool-only behavior. Every
renewal must advertise capabilities again. Omitting `timer` on renewal removes
its tools from the new provider call.

Timer v1 contributes exactly:

- `device_timer_create`
- `device_timer_list`
- `device_timer_cancel`

The JSON catalog is authoritative for their arguments and result shapes. The
APK must not send tool definitions or schemas during negotiation.

## Action request

When the provider selects a negotiated tool, Hermes publishes a replayable
control event:

```json
{
  "version": 1,
  "sequence": 42,
  "session_id": "rt_123",
  "stream_id": "control-stream-uuid",
  "type": "device.action.request",
  "data": {
    "invocation_id": "provider-tool-call-id",
    "device_instance_id": "22222222-2222-4222-8222-222222222222",
    "tool_name": "device_timer_create",
    "capability": "timer",
    "operation": "create",
    "arguments": {
      "duration_seconds": 300,
      "label": null
    },
    "expires_at_epoch_ms": 1786450200000
  }
}
```

Before execution, the APK must verify:

1. The negotiated device instance and capability match.
2. The tool and operation exist in its implementation of that capability
   version.
3. The arguments satisfy the same catalog contract.
4. `expires_at_epoch_ms` has not passed.
5. The `invocation_id` has not already been executed.

The provider tool-call ID is the invocation ID. Retain it when a control event
is replayed. Never create a replacement ID for a retry.

## Action result

Send a normal control command for success:

```json
{
  "version": 1,
  "request_id": "android-123",
  "type": "device.action.result",
  "data": {
    "invocation_id": "provider-tool-call-id",
    "status": "ok",
    "result": {
      "timer_id": "timer-42",
      "fires_at_epoch_ms": 1786450195000,
      "label": null
    }
  }
}
```

Or return a structured execution error:

```json
{
  "version": 1,
  "request_id": "android-124",
  "type": "device.action.result",
  "data": {
    "invocation_id": "provider-tool-call-id",
    "status": "error",
    "error": {
      "code": "permission_denied",
      "message": "Exact alarms are unavailable",
      "retryable": false
    }
  }
}
```

Hermes acknowledges the first valid result with `control.ack`. A repeated valid
result for the same completed invocation is acknowledged and ignored. Unknown,
wrong-session, or expired invocation IDs produce `control.error`.

## Delivery and outcome semantics

Control delivery is at least once. The APK must durably deduplicate side
effects by `invocation_id` for long enough to cover control replay and action
expiry. A replayed request whose expiry has passed must not create a timer.

Hermes does not automatically retry a dispatched side effect. If the control
connection closes, the result deadline passes, or Hermes restarts after
dispatch, Hermes durably returns a nonfatal `outcome_unknown` tool result to
the model. The model can explain the ambiguity, but must not create a
replacement invocation that could duplicate the timer.

An explicit APK error is not ambiguous: Hermes forwards it as a normal,
nonfatal tool error. Tool results are durably persisted and sent to the
Realtime provider before Hermes requests the model's continuation.

Started actions are not cancellable in protocol v1. Barge-in can suppress a
tool that has not started, but it does not imply that a dispatched device
action failed to execute.
