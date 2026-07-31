---
sidebar_position: 8
title: "Programmatic Integration"
description: "Drive hermes-agent through ACP, JSON-RPC, HTTP, or native Realtime voice"
---

# Programmatic Integration

Hermes ships three host protocols for driving the agent from external programs
— IDE plugins, custom UIs, CI pipelines, and embedded sub-agents. The API
server also has a native OpenAI Realtime route family for voice-only WebRTC
clients. Pick the surface that matches your transport and consumer.

| Protocol | Transport | Best for | Defined by |
|----------|-----------|----------|------------|
| **ACP** | JSON-RPC over stdio | IDE clients (VS Code, Zed, JetBrains) that already speak the [Agent Client Protocol](https://github.com/zed-industries/agent-client-protocol) | `acp_adapter/` |
| **TUI gateway** | JSON-RPC over stdio (or WebSocket) | Custom hosts that want fine-grained control of sessions, slash commands, approvals, and streaming events | `tui_gateway/server.py` |
| **API server** | HTTP + SSE + WebSocket; WebRTC signaling for voice | OpenAI-compatible frontends, language-agnostic clients, and native Realtime voice apps | `gateway/platforms/api_server.py`, `gateway/realtime/` |

All three drive the same `AIAgent` core. They differ only in wire format and which set of features they expose.

---

## ACP (Agent Client Protocol)

`hermes acp` starts a stdio JSON-RPC server speaking ACP. Used in production by VS Code (Zed Industries' ACP extension), Zed, and any JetBrains IDE with an ACP plugin.

Capabilities exposed: session creation, prompt submission, streaming agent message chunks, tool-call events, permission requests, session fork, cancel, and authentication. Tool output is rendered into ACP `Diff`/`ToolCall` content blocks the IDE understands.

Full lifecycle, event bridge, and approval flow: [ACP Internals](./acp-internals).

```bash
hermes acp                  # serve ACP on stdio
hermes acp --check          # verify ACP dependencies and adapter imports
hermes acp --setup          # interactive provider/model setup for ACP terminal auth
```

---

## TUI Gateway JSON-RPC

`tui_gateway/server.py` is the protocol the Ink TUI (`hermes --tui`) and the embedded dashboard PTY bridge talk to. Any external host can speak the same protocol over stdio (or WebSocket via `tui_gateway/ws.py`).

### Method catalog (selected)

```
prompt.submit           prompt.background       session.steer
session.create          session.list            session.active_list
session.activate        session.close           session.interrupt
session.history         session.compress        session.branch
session.title           session.usage           session.status
clarify.respond         sudo.respond            secret.respond
approval.respond        config.set / config.get commands.catalog
command.resolve         command.dispatch        cli.exec
reload.mcp              reload.env              process.stop
delegation.status       subagent.interrupt      spawn_tree.save / list / load
terminal.resize         clipboard.paste         image.attach
```

`session.active_list`, `session.activate`, and `session.close` are the process-local live-session controls used by the TUI session switcher. Use `session.list` / `/resume` for saved transcript discovery; use the active-session methods only for sessions that are currently open in the TUI gateway process.

### Events streamed back

`message.delta`, `message.complete`, `tool.start`, `tool.progress`, `tool.complete`, `approval.request`, `clarify.request`, `sudo.request`, `sudo.expire`, `secret.request`, `secret.expire`, `gateway.ready`, plus session lifecycle and error events. Expiry events carry the original `{ request_id }`; external hosts should clear only the matching pending prompt.

### Pi-style RPC mapping

Every command in the Pi-mono RPC spec ([issue #360](https://github.com/NousResearch/hermes-agent/issues/360)) has a TUI-gateway equivalent:

| Pi command | Hermes equivalent |
|------------|-------------------|
| `prompt` | `prompt.submit` (or ACP `session/prompt`) |
| `steer` | `session.steer` |
| `follow_up` | `prompt.submit` queued after current turn |
| `abort` | `session.interrupt` |
| `set_model` | `command.dispatch` for `/model <provider:model>` (mid-session, persistent) |
| `compact` | `session.compress` |
| `get_state` | `session.status` |
| `get_messages` | `session.history` |
| `switch_session` | `session.resume` |
| `fork` | `session.branch` |
| `ui_request` / `ui_response` | `clarify.respond` / `sudo.respond` / `secret.respond` / `approval.respond` |

---

## OpenAI-Compatible API Server

`gateway/platforms/api_server.py` exposes hermes over HTTP for any client that already speaks the OpenAI format. Useful when you want a web frontend, a curl-driven CI runner, or a non-Python consumer.

Endpoints:

```
POST /v1/chat/completions        OpenAI Chat Completions (streaming via SSE)
POST /v1/responses               OpenAI Responses API (stateful)
POST /v1/runs                    Start a run, returns run_id (202)
GET  /v1/runs/{id}               Run status
GET  /v1/runs/{id}/events        SSE stream of lifecycle events
POST /v1/runs/{id}/approval      Resolve a pending approval
POST /v1/runs/{id}/stop          Interrupt the run
GET  /v1/capabilities            Machine-readable feature flags
GET  /v1/models                  Lists hermes-agent
GET  /api/model/options          Provider-aware picker inventory
POST /v1/realtime/sessions       Exchange a WebRTC SDP offer
GET  /v1/realtime/sessions/{id}/control
                                 Structured control WebSocket
POST /v1/realtime/sessions/{id}/renew
                                 Rotate the provider call with a new SDP offer
POST /v1/realtime/sessions/{id}/approval
                                 HTTP approval fallback
DELETE /v1/realtime/sessions/{id}
                                 Close the logical voice session
GET  /health, /health/detailed
```

Setup, headers (`X-Hermes-Session-Id`, `X-Hermes-Session-Key`), and frontend wiring: [API Server](../user-guide/features/api-server).

### Model catalog surfaces

The OpenAI-compatible API intentionally keeps `GET /v1/models` minimal: it is
the compatibility endpoint frontends expect, not the full Hermes provider/model
picker catalog.

If an external control plane needs Hermes' curated provider rows, per-model
pricing, or capability hints, use one of the authenticated picker surfaces:

- API server REST: `GET /api/model/options` with the API-server bearer key
- Dashboard backend REST: `GET /api/model/options` with `X-Hermes-Session-Token`
- TUI gateway RPC: `model.options`

Those surfaces share the same payload builder and the same custom-provider
probe policy:

- Normal open: probe only the current custom provider so offline saved
  endpoints do not stall the picker.
- Explicit refresh (`refresh=1` or `refresh: true`): bust the provider-model
  cache and probe all saved custom providers so live catalogs repopulate fully.

Use `/v1/models` for OpenAI-client compatibility. Use `/api/model/options` or
`model.options` when you are building a Hermes-aware model picker.

### Native Realtime voice (Android/WebRTC)

The native voice API is not text streaming plus TTS. Android and OpenAI carry
bidirectional audio over WebRTC, while a Hermes session actor attaches to the
same provider call over a sideband WebSocket. Hermes remains the sole
orchestrator for instructions, responses, tools, authorization, persistence,
and rotation.

```mermaid
sequenceDiagram
    participant App as Android app
    participant API as Hermes API server
    participant Actor as Hermes Realtime actor
    participant OAI as OpenAI Realtime
    participant Core as Hermes tools and SessionDB

    App->>App: Create peer connection, audio track, and SDP offer
    App->>API: POST /v1/realtime/sessions (offer + bearer auth)
    API->>Core: Build AIAgent, SOUL, memory, skills, and tool snapshot
    API->>OAI: POST /v1/realtime/calls (offer + server API key)
    OAI-->>API: SDP answer + call_id
    API->>Actor: Attach sideband WebSocket by call_id
    API-->>App: session_id + SDP answer + control_url
    App->>App: Apply remote description
    App<<->>OAI: WebRTC microphone and model audio
    App<<->>API: Sequenced structured controls
    OAI-->>Actor: Committed input, function calls, response events
    Actor->>Core: Persist, authorize, and execute tools
    Actor->>OAI: Function outputs and one continuation response
```

#### Prerequisites

1. Enable the authenticated API server and set `API_SERVER_KEY`.
2. Set `OPENAI_API_KEY` on the Hermes host.
3. Set `realtime_voice.enabled: true` in `config.yaml`.
4. For automatic post-turn memory/skill review, configure an explicit
   text-capable `auxiliary.background_review` provider and model.
5. Use HTTPS/WSS for a device connecting outside trusted loopback/private
   networking.

Probe `GET /v1/capabilities` before constructing media. Continue only when
`features.realtime_voice` is `true`. Inspect
`features.realtime_voice_details` for the immutable model/voice, admission
limit, rotation thresholds, barge-in contract, and background-review status.

#### 1. Negotiate the media call

The Android app creates an `RTCPeerConnection`, adds its microphone audio
track, configures remote audio playback, and generates an SDP offer. Send that
offer to Hermes:

```http
POST /v1/realtime/sessions HTTP/1.1
Authorization: Bearer <API_SERVER_KEY>
Content-Type: application/json

{
  "sdp": "v=0\r\n...",
  "session_id": "optional_logical_session_id"
}
```

Raw `application/sdp` request bodies are also accepted. JSON is always returned:

```json
{
  "version": 1,
  "session_id": "rt_123",
  "call_id": "rtc_456",
  "sdp": "v=0\r\n...",
  "model": "gpt-realtime",
  "voice": "marin",
  "control_url": "/v1/realtime/sessions/rt_123/control",
  "renew_url": "/v1/realtime/sessions/rt_123/renew",
  "provider_call_max_seconds": 3300
}
```

Apply `sdp` as the peer connection's remote description. `call_id` is
diagnostic correlation data; the client does not connect a sideband socket or
use it to control OpenAI.

The Android OpenAI data channel is transport-only. It must not send
`session.update`, `conversation.item.create`, function outputs, or
`response.create`. Hermes is the only writer for those events, which prevents
client state from bypassing its frozen prompt, authorization, or durable
ordering.

#### 2. Attach the control WebSocket

Upgrade the returned `control_url` using the same bearer header. Preserve the
largest processed `sequence`, and include it on reconnect:

```text
wss://hermes.example/v1/realtime/sessions/rt_123/control?after=42
Authorization: Bearer <API_SERVER_KEY>
```

Every server event has this envelope:

```json
{
  "version": 1,
  "sequence": 43,
  "session_id": "rt_123",
  "type": "tool.started",
  "timestamp": 1770000000.0,
  "data": {
    "call_id": "call_abc",
    "name": "web_search",
    "argument_keys": ["query"]
  }
}
```

The control plane is not a transcript channel. It carries bounded,
non-conversational state used to render connection, activity, authorization,
usage, and recovery UI. Important event families include:

| Events | Client behavior |
|--------|-----------------|
| `session.state`, `session.reconnected`, `session.rotated` | Update connection/lifecycle UI; `session.state` carries closing and closed states |
| `vad.speech_started`, `vad.speech_stopped` | Show microphone/turn state |
| `turn.input_committed`, `turn.steered`, `turn.completed` | Reconcile logical voice-turn state without rendering assistant text |
| `tool.started`, `tool.completed` | Render tool activity; arguments are reduced to key names |
| `approval.request`, `approval.resolved` | Show and resolve a privileged-action decision |
| `clarification.request`, `secret.request` | Render structured input; treat secret values as sensitive |
| `response.interrupted` | Stop local response UI after barge-in |
| `usage.updated`, `rate_limits.updated` | Update diagnostics or metering |
| `warning`, `error`, `control.error` | Surface redacted degraded/failure state |
| `session.rotation_required` | Renegotiate using `/renew` |
| `control.resync_required` | Discard stale transient UI and rebuild from subsequent state |
| `control.ack`, `session.pong` | Correlate client commands and liveness checks |

Control buffers and per-subscriber queues are bounded. Reconnect promptly with
the last applied cursor. A `control.resync_required` event means the requested
cursor predates the replay buffer; do not infer that missing tool or approval
events never happened.

#### 3. Send structured commands

Client messages are JSON text with a protocol version, optional correlation
ID, command type, and data object:

```json
{
  "version": 1,
  "request_id": "android-104",
  "type": "session.ping",
  "data": {}
}
```

Supported commands:

| Type | Required data |
|------|---------------|
| `session.ping` | none; produces `session.pong` |
| `session.close` | none; closes the logical session |
| `approval.respond` | `choice`: `once`, `session`, `always`, or `deny`; include the current `approval_id` |
| `clarification.respond` | current `prompt_id` and string `value` |
| `secret.respond` | current `prompt_id` and string `value` |

Commands with `request_id` produce `control.ack` after success, except ping,
which produces `session.pong`. Invalid or stale commands produce
`control.error`. Binary control messages are rejected.

Approval may instead use the HTTP fallback:

```http
POST /v1/realtime/sessions/rt_123/approval HTTP/1.1
Authorization: Bearer <API_SERVER_KEY>
Content-Type: application/json

{
  "approval_id": "approval_...",
  "choice": "once"
}
```

Do not map spoken "yes", voice activity, or model output to
`approval.respond`. A dangerous operation resumes only after the authenticated
app submits the matching structured decision. Clarification is conversational;
secrets and sudo credentials are not.

#### Tool and turn semantics

Hermes configures OpenAI VAD with `create_response=false`, so a committed audio
turn does not bypass local preparation. The actor first persists the user
boundary, runs turn-start hooks and memory prefetch, then explicitly requests a
response.

Function calls follow a durable order:

1. Persist the assistant function call before executing side effects.
2. Execute through the shared Hermes tool pipeline.
3. Persist each result before publishing completion or returning it to OpenAI.
4. Send one function output per provider `call_id`.
5. Send one continuation after the ready batch.

The provider `call_id` is the durable idempotency key. Duplicate completion
events or a sideband reconnect reuse the stored output instead of repeating a
side effect.

The model is prompted to acknowledge long work naturally. If a tool remains
silent past the configured delay, Hermes may request one short
`conversation: "none"` status response with no tools. Status speech is
rate-limited, canceled by barge-in, and excluded from canonical history.

On barge-in:

- OpenAI interrupts/truncates current output audio.
- A function call that has not begun is skipped if its response was
  superseded.
- A started tool continues to a persisted result; voice input never kills a
  side effect mid-operation.
- The steering transcript is attached at the next durable tool boundary and
  the model reevaluates.

#### SOUL, skills, memory, and self-improvement

At logical-session creation Hermes initializes a normal `AIAgent`, builds the
same SOUL/context/memory/skills prompt used by text channels, resolves the
session toolset, and freezes both snapshots. Call renewal and gateway recovery
reuse those exact snapshots; changes to SOUL, the injected skill index, or
tool configuration take effect in a new logical session.

Foreground `skills_list`, `skill_view`, and `skill_manage` calls use the real
profile-safe implementations. A newly written skill can be opened explicitly
by name in the current session, while automatic discovery of its index entry
waits for the next session.

Post-turn review is isolated from the media loop. Hermes durably marks review
work due and runs a separate text-capable background `AIAgent`. The voice actor
returns to listening without waiting, the reviewer never emits Realtime
function output or audio, and successful maintenance is silent by default.
Capability discovery reports review as unavailable rather than inheriting
`gpt-realtime` when no text runtime is configured.

#### Recovery, renewal, and shutdown

Provider calls rotate before `provider_call_max_seconds` or
`provider_call_max_input_tokens`. On `session.rotation_required`:

1. Create a replacement peer connection and SDP offer.
2. `POST` the offer to `/v1/realtime/sessions/{id}/renew`.
3. Apply the returned SDP answer.
4. Continue using the same logical `session_id` and control cursor.

Hermes opens a new sideband call, reapplies the frozen instructions/tools, and
replays bounded canonical history including messages, function calls, and
function outputs. This rotates provider transport without changing the Hermes
conversation.

After a gateway restart, a persisted call can be reattached only when it was
idle (`ready` or `listening`) and is still within the recovery age. Otherwise
the control or approval endpoint returns a conflict requiring `/renew`. Active
tool/response state is never guessed.

Close normally with `DELETE /v1/realtime/sessions/{id}` or `session.close`.
The HTTP route returns `204` after finalizing local resources.

#### Security and compatibility boundaries

- The standard OpenAI API key never appears in the Android-facing SDP response
  or control stream. Hermes derives the OpenAI safety identifier from its
  authenticated server/profile scope.
- Raw microphone audio is not persisted by Hermes by default. Canonical
  history stores finalized transcripts, function calls/results, interruption
  metadata, and usage.
- If transcription fails or times out, Hermes stores an explicit unavailable
  marker and emits a warning; it does not invent text or block indefinitely.
- Tool argument values and provider errors are redacted before control-plane
  publication.
- `transform_llm_output` cannot rewrite native audio already played to the
  listener. Post-response observation, memory sync, and review still run.
- v1 supports the OpenAI Realtime protocol only. A provider/media failure is
  surfaced explicitly; there is no silent fallback to text. OpenAI-compatible
  call and sideband proxy endpoints may be configured under
  `realtime_voice.transport`; they must preserve the SDP, call ID, and
  Realtime event contracts.
- The repository defines the server and wire contract, not an Android UI or
  WebRTC client implementation.

---

## Which one should I use?

- **You're writing an IDE plugin and the IDE already speaks ACP** → ACP. Zero protocol work on the IDE side.
- **You're writing a custom desktop / web / TUI host and want every Hermes feature** (slash commands, approvals, clarify, multi-agent, session branching) → TUI gateway JSON-RPC.
- **You want any OpenAI-compatible frontend, a language-agnostic HTTP client, or curl-driven automation** → API server.
- **You're writing a voice-only Android client and need native streamed speech plus Hermes tools** → API server Realtime routes with WebRTC and the control WebSocket.
- **You want a Python in-process embed without a subprocess** → import `run_agent.AIAgent` directly. See [Agent Loop](./agent-loop).

---

## Model hot-swapping

Mid-session model switching works on every surface — it's the `/model` slash command under the hood.

- **CLI / TUI:** `/model claude-sonnet-4` or `/model openrouter:anthropic/claude-sonnet-4.6`
- **TUI gateway RPC:** `command.dispatch` with `{"command": "/model claude-sonnet-4"}`
- **ACP:** the IDE sends the slash command as a prompt; the agent dispatches it
- **API server:** include a `model` field in the request body

Provider-aware resolution (the same model name picks the right format for whatever provider you're on) is built in. See `hermes_cli/model_switch.py`.

---

## A note on `--mode rpc`

Hermes does not have a `--mode rpc` flag. The host protocols above already
cover the use cases — ACP for IDE-protocol clients, the TUI gateway for stdio
JSON-RPC hosts, and the API server for HTTP, SSE, control WebSockets, and
Realtime WebRTC signaling. If you find a real gap that none of them fill, open
an issue with the concrete consumer you're building.
