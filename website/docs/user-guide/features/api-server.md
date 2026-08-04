---
sidebar_position: 14
title: "API Server"
description: "Expose hermes-agent as an OpenAI-compatible API for any frontend"
---

# API Server

The API server exposes hermes-agent as an OpenAI-compatible HTTP endpoint. Any frontend that speaks the OpenAI format — Open WebUI, LobeChat, LibreChat, NextChat, ChatBox, and hundreds more — can connect to hermes-agent and use it as a backend.

Your agent handles requests with its full toolset (terminal, file operations, web search, memory, skills) and returns the final response. When streaming, tool progress indicators appear inline so frontends can show what the agent is doing.

:::tip One backend covers models + tools
Hermes itself needs a configured provider and tool backends for the API server to be useful. A [Nous Portal](/user-guide/features/tool-gateway) subscription handles both — 300+ models plus web/image/TTS/browser via the Tool Gateway. Run `hermes setup --portal` once before starting the API server and frontends like Open WebUI or LobeChat get a fully tool-equipped backend.
:::

## Quick Start

### 1. Enable the API server

Add to `~/.hermes/.env`:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=change-me-local-dev
# Optional: only if a browser must call Hermes directly
# API_SERVER_CORS_ORIGINS=http://localhost:3000
```

### 2. Start the gateway

```bash
hermes gateway
```

You'll see:

```
[API Server] API server listening on http://127.0.0.1:8642
```

### 3. Connect a frontend

Point any OpenAI-compatible client at `http://localhost:8642/v1`:

```bash
# Test with curl
curl http://localhost:8642/v1/chat/completions \
  -H "Authorization: Bearer change-me-local-dev" \
  -H "Content-Type: application/json" \
  -d '{"model": "hermes-agent", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Or connect Open WebUI, LobeChat, or any other frontend — see the [Open WebUI integration guide](/user-guide/messaging/open-webui) for step-by-step instructions.

## Endpoints

### POST /v1/chat/completions

Standard OpenAI Chat Completions format. Stateless — the full conversation is included in each request via the `messages` array.

**Request:**
```json
{
  "model": "hermes-agent",
  "messages": [
    {"role": "system", "content": "You are a Python expert."},
    {"role": "user", "content": "Write a fibonacci function"}
  ],
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "hermes-agent",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Here's a fibonacci function..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 50, "completion_tokens": 200, "total_tokens": 250}
}
```

**Inline image input:** user messages may send `content` as an array of `text` and `image_url` parts. Both remote `http(s)` URLs and `data:image/...` URLs are supported:

```json
{
  "model": "hermes-agent",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}}
      ]
    }
  ]
}
```

Uploaded files (`file` / `input_file` / `file_id`) and non-image `data:` URLs return `400 unsupported_content_type`.

**Streaming** (`"stream": true`): Returns Server-Sent Events (SSE) with token-by-token response chunks. For **Chat Completions**, the stream uses standard `chat.completion.chunk` events plus Hermes' custom `hermes.tool.progress` event for tool-start UX. For **Responses**, the stream uses OpenAI Responses event types such as `response.created`, `response.output_text.delta`, `response.output_item.added`, `response.output_item.done`, and `response.completed`.

**Tool progress in streams**:
- **Chat Completions**: Hermes emits `event: hermes.tool.progress` for tool-start visibility without polluting persisted assistant text.
- **Responses**: Hermes emits spec-native `function_call` and `function_call_output` output items during the SSE stream, so clients can render structured tool UI in real time.

### POST /v1/responses

OpenAI Responses API format. Supports server-side conversation state via `previous_response_id` — the server stores full conversation history (including tool calls and results) so multi-turn context is preserved without the client managing it.

**Request:**
```json
{
  "model": "hermes-agent",
  "input": "What files are in my project?",
  "instructions": "You are a helpful coding assistant.",
  "store": true
}
```

**Response:**
```json
{
  "id": "resp_abc123",
  "object": "response",
  "status": "completed",
  "model": "hermes-agent",
  "output": [
    {"type": "function_call", "name": "terminal", "arguments": "{\"command\": \"ls\"}", "call_id": "call_1"},
    {"type": "function_call_output", "call_id": "call_1", "output": "README.md src/ tests/"},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Your project has..."}]}
  ],
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250}
}
```

**Inline image input:** `input[].content` can contain `input_text` and `input_image` parts. Both remote URLs and `data:image/...` URLs are supported:

```json
{
  "model": "hermes-agent",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Describe this screenshot."},
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0K..."}
      ]
    }
  ]
}
```

Uploaded files (`input_file` / `file_id`) and non-image `data:` URLs return `400 unsupported_content_type`.

#### Multi-turn with previous_response_id

Chain responses to maintain full context (including tool calls) across turns:

```json
{
  "input": "Now show me the README",
  "previous_response_id": "resp_abc123"
}
```

The server reconstructs the full conversation from the stored response chain — all previous tool calls and results are preserved. Chained requests also share the same session, so multi-turn conversations appear as a single entry in the dashboard and session history.

#### Named conversations

Use the `conversation` parameter instead of tracking response IDs:

```json
{"input": "Hello", "conversation": "my-project"}
{"input": "What's in src/?", "conversation": "my-project"}
{"input": "Run the tests", "conversation": "my-project"}
```

The server automatically chains to the latest response in that conversation. Like the `/title` command for gateway sessions.

### GET /v1/responses/\{id\}

Retrieve a previously stored response by ID.

### DELETE /v1/responses/\{id\}

Delete a stored response.

### GET /v1/models

Lists the agent as an available model. The advertised model name defaults to the [profile](/user-guide/profiles) name (or `hermes-agent` for the default profile). Required by most frontends for model discovery.

`/v1/models` is intentionally the cheap OpenAI-compat surface. It does **not**
enumerate every authenticated provider/model combination Hermes can route to,
and it does not do pricing or capability enrichment.

### GET /api/model/options

Hermes-aware clients can request the same curated provider/model inventory used
by the dashboard and TUI. This route uses the API server's normal bearer
authentication and returns provider rows, model capability hints, and pricing
metadata that do not belong in the OpenAI-compatible `/v1/models` response:

```bash
curl \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  "http://127.0.0.1:8642/api/model/options"
```

That payload is the same substrate the dashboard Models page and the TUI
`model.options` RPC use. It returns authenticated providers, curated model
lists, per-model pricing, and model capability hints.

Normal opens are intentionally conservative for custom providers: Hermes probes
only the **currently selected** custom endpoint so a stale or offline saved
endpoint does not block the picker. An explicit refresh flips to full probing
and busts the provider model cache:

```bash
curl \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  "http://127.0.0.1:8642/api/model/options?refresh=1"
```

Use `/v1/models` when an OpenAI-compatible client only needs a model name to
send back in chat/responses requests. Use `/api/model/options` when an
authenticated UI needs the richer Hermes-specific picker metadata.

### GET /v1/capabilities

Returns a machine-readable description of the API server's stable surface for external UIs, orchestrators, and plugin bridges.

```json
{
  "object": "hermes.api_server.capabilities",
  "platform": "hermes-agent",
  "model": "hermes-agent",
  "auth": {"type": "bearer", "required": true},
  "features": {
    "chat_completions": true,
    "responses_api": true,
    "run_submission": true,
    "run_status": true,
    "run_events_sse": true,
    "run_stop": true,
    "realtime_voice": false,
    "realtime_voice_details": {
      "available": false,
      "enabled": false
    }
  }
}
```

Use this endpoint when integrating dashboards, browser UIs, or control planes so they can discover whether the running Hermes version supports runs, streaming, cancellation, and session continuity without depending on private Python internals.

## Native Realtime Voice (WebRTC)

Hermes can expose a voice-only, speech-to-speech agent backed by OpenAI
Realtime. The Android client sends microphone audio to OpenAI and receives
model audio over WebRTC, while Hermes joins the same call over a server-side
sideband connection.

This split keeps the latency-sensitive audio path direct without moving
authority out of Hermes:

- The standard `OPENAI_API_KEY` stays on the Hermes host.
- Hermes supplies the frozen SOUL, context, skill index, and tool schemas.
- Every tool call still runs through Hermes approvals, guardrails, plugins,
  checkpoints, memory, and persistence.
- Only authenticated structured controls can approve privileged work. Spoken
  confirmation is never authorization.
- Hermes, not the Android data channel, sends tool results and creates model
  responses.

The repository provides this backend protocol but does not include an Android
application. See [Programmatic Integration](/developer-guide/programmatic-integration#native-realtime-voice-androidwebrtc)
for the client sequence and control envelopes.

### Enable realtime voice

Add the non-secret settings to `~/.hermes/config.yaml`:

```yaml
realtime_voice:
  enabled: true
  model: gpt-realtime
  voice: marin
```

Add the provider credential to `~/.hermes/.env`:

```bash
OPENAI_API_KEY=your-openai-api-key
```

The API server itself must also be enabled and protected by
`API_SERVER_KEY`. Restart `hermes gateway`, then check availability:

```bash
curl http://127.0.0.1:8642/v1/capabilities \
  -H "Authorization: Bearer $API_SERVER_KEY"
```

`features.realtime_voice` becomes `true` only when both
`realtime_voice.enabled` and `OPENAI_API_KEY` are present. The
`realtime_voice_details` object reports the configured model and voice,
session limits, barge-in behavior, and whether background self-improvement
has a text-capable auxiliary runtime.

### Session endpoints

All routes use the normal API-server bearer authentication.

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/realtime/sessions` | Exchange an Android SDP offer for the OpenAI SDP answer and create or resume a logical Hermes session |
| `GET` | `/v1/realtime/sessions/{id}/control?stream_id={stream_id}&after={sequence}` | Upgrade to the sequenced structured-control WebSocket |
| `POST` | `/v1/realtime/sessions/{id}/audio/preroll` | Commit one complete first utterance buffered locally during wake-word/session startup |
| `POST` | `/v1/realtime/sessions/{id}/approval` | HTTP fallback for an approval decision |
| `POST` | `/v1/realtime/sessions/{id}/renew` | Exchange a fresh SDP offer while preserving the logical session |
| `POST` | `/v1/realtime/sessions/{id}/suspend` | Close provider resources while preserving the logical Hermes session |
| `DELETE` | `/v1/realtime/sessions/{id}` | Close media and sideband resources and finalize the session |

Create and renew accept either `application/sdp` or JSON:

```json
{
  "sdp": "v=0\r\n...",
  "session_id": "optional_existing_logical_session"
}
```

The response is JSON:

```json
{
  "version": 1,
  "session_id": "rt_...",
  "call_id": "rtc_...",
  "sdp": "v=0\r\n...",
  "model": "gpt-realtime",
  "voice": "marin",
  "control_url": "/v1/realtime/sessions/rt_.../control",
  "preroll_url": "/v1/realtime/sessions/rt_.../audio/preroll",
  "renew_url": "/v1/realtime/sessions/rt_.../renew",
  "suspend_url": "/v1/realtime/sessions/rt_.../suspend",
  "provider_call_max_seconds": 3300
}
```

The client sets `sdp` as its WebRTC remote description. Neither this response
nor any control event contains the standard OpenAI API key.

### Wake-word startup without dropped speech

An always-open provider call is not required. Keep wake detection and a rolling
PCM buffer entirely on the Android device, then use `preroll_url` for each
locally captured utterance:

1. One local `AudioRecord` feeds both the wake detector and a 4-6 second ring
   buffer.
2. After detection, start SDP negotiation and continue the same local capture
   through **local end-of-speech**.
3. Keep the WebRTC microphone sender inactive. Freeze the completed utterance
   while retaining the shared local capture source for future wake detection.
4. Upload the complete utterance as `audio/pcm;rate=24000;channels=1;format=s16le`
   with a unique `Idempotency-Key`.
5. Wait until the endpoint returns `status: committed`. A local-capture client
   may keep WebRTC receive-only and repeat the upload with a new idempotency key.

Hermes temporarily disables provider VAD, clears the provider input buffer,
appends and commits the PCM through its server-owned sideband, then restores
the configured VAD before replying. The resulting item follows the same
transcription, persistence, tool, memory, and response path as live WebRTC
speech.

The upload must be a complete first utterance. Do not send an unfinished prefix
and continue it over RTP: the two transports cannot guarantee one coherent VAD
turn. Avoid two simultaneous Android `AudioRecord` owners; use one shared
capture source with receive-only WebRTC, or release wake-word capture before
enabling WebRTC recording.
If an upload times out, keep the microphone inactive and renew/recreate the
provider call rather than resubmitting under a new key.

### Quick PC microphone test

The repository includes a dependency-free localhost client for checking the
voice path before building a mobile app. It uses the browser for microphone
capture and WebRTC audio, while a small Python proxy keeps `API_SERVER_KEY`
out of browser JavaScript.

On Linux or macOS:

```bash
export HERMES_REALTIME_URL="https://your-hermes-api.example.com"
export HERMES_API_KEY="your-api-server-key"
python3 scripts/realtime_voice_pc_client.py
```

On Windows PowerShell:

```powershell
$env:HERMES_REALTIME_URL = "https://your-hermes-api.example.com"
$env:HERMES_API_KEY = "your-api-server-key"
python scripts\realtime_voice_pc_client.py
```

The script binds only to `127.0.0.1:8787` and opens the page in the default
browser. Allow microphone access, select **Start talking**, and speak after the
status becomes live. Use `--port 8788` to choose another local port or
`--no-open` to suppress automatic browser launch. The local proxy tracks active
sessions and reclaims them on Ctrl+C; the page also closes microphone capture
if that proxy disappears.

This is deliberately a media smoke test, not a full application. It does not
open the structured control WebSocket, display approval prompts, or renew a
long-running provider call. Stop the session before closing the page; abnormal
disconnects are eventually reclaimed by the Hermes session manager.

### What remains available

Realtime voice uses the same `AIAgent` capability substrate as text channels:

- SOUL, project context, memory, skills, plugins, MCP tools, delegation,
  checkpoints, and approval policy remain active.
- Skill reads and user-requested skill writes are ordinary foreground tool
  calls. A skill written during a call is immediately readable by name.
- The injected skill index and tool schemas stay frozen for the logical
  session. Newly discovered capabilities appear automatically in the next
  session.
- Automatic memory/skill review runs after a settled voice turn in a separate
  text-model agent. It never delays the next voice turn or speaks its internal
  work.

Because the main model is `gpt-realtime`, automatic background review needs an
explicit text-capable runtime. For example:

```yaml
auxiliary:
  background_review:
    provider: openai-api
    model: gpt-5.4-mini
```

If no compatible runtime can be resolved, voice remains available but
`background_self_improvement.available` is `false` in `/v1/capabilities`.
Successful review notifications are silent by default on the voice surface.

### Interruption and renewal

OpenAI VAD detects turns, but Hermes controls response creation. If the user
speaks while the agent is responding, current audio is interrupted. A tool
that already started is allowed to finish; superseded calls that have not
started are skipped, and the model reevaluates after the durable result
boundary.

A client that owns microphone capture locally and does not send RTP uplink must
request barge-in over the authenticated control WebSocket:

```json
{
  "version": 1,
  "type": "response.interrupt",
  "request_id": "wake-018f3c0d",
  "data": {"audio_end_ms": 875}
}
```

Stop local playback before sending the command. `audio_end_ms` is the duration
of the current response actually rendered from decoded frames, reset for every
response; it is not wall-clock or RTP-receive time. Hermes cancels the exact
provider response if generation is active. It clears an active WebRTC output
buffer so queued RTP stops and OpenAI truncates at its synchronized playback
cursor; after provider drain it uses the client's `audio_end_ms` to truncate
the remaining device-playout tail. The command remains valid after provider
generation finishes while local audio is still playing. Continue local capture
through end-of-speech and upload the replacement utterance to `preroll_url`.
That upload is accepted while the interrupted response or a started tool is
settling.

For a normal completion, `response.generated` means OpenAI finished generation
and `response.output_drained` means its WebRTC output buffer drained. After the
device decoder/render queue also drains, send:

```json
{
  "version": 1,
  "type": "response.playback_completed",
  "request_id": "played-018f3c0d",
  "data": {"response_id": "response_abc"}
}
```

Hermes then emits `response.playback_confirmed` and `turn.completed`. A bounded
fallback protects legacy clients, but sending the command promptly avoids
holding the provider call open unnecessarily.

Keep the wake detector on an echo-cancelled local stream during assistant
playback. Validate hardware AEC at real speaker volume and add wake
confidence/debounce to avoid the assistant triggering its own wake phrase.

For zero idle provider cost, drain local playout, send
`response.playback_completed`, wait for `turn.completed`, then
`POST /v1/realtime/sessions/{id}/suspend` and close the peer connection. The
endpoint returns `204` and preserves SessionDB history, memory, and the logical
`session_id`. At the next wake, create a new peer connection, call `/renew`,
and upload the new complete utterance. During barge-in, interrupt and reuse the
current call first; using `/renew` as cancellation adds latency and does not
replace precise interruption.

The provider call rotates before its age or input-context limit. When the
control socket emits `session.rotation_required`, create a fresh peer
connection, send its offer to `/renew`, and apply the returned answer. The
logical Hermes session, frozen prompt and tools, transcript, memory, and
durable tool results continue across that rotation.

Reconnect the control WebSocket with the last `stream_id` and `sequence` from
its event envelopes. Hermes emits `control.resync_required` when the cursor is
outside the bounded buffer or belongs to a previous stream epoch. Suspension
closes the socket with code `1001`; after `/renew`, reconnect with the old pair,
adopt the new stream ID from the resync event, and rebuild transient UI from
its authoritative `data.snapshot`. Resync events are private to the reconnecting
socket and do not reset healthy subscribers.

## Per-request model selection

Authenticated clients can override Hermes' default model selection per request
by sending:

- `model` — the target model id for this turn
- `provider` — the Hermes provider slug to resolve credentials/runtime for this turn
- `model_options` — request-scoped reasoning / service-tier controls

The same request fields are accepted on:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/runs`
- `POST /api/sessions/{session_id}/chat`
- `POST /api/sessions/{session_id}/chat/stream`

Precedence is deterministic:

1. Session `/model` override, if that session already has one
2. A static `gateway.platforms.api_server.model_routes` mapping selected when
   the request's `model` is a configured route alias
3. Direct request `model` / `provider` when no route alias matches
4. Global gateway config / environment defaults

`model_options` stays request-scoped regardless of which model/provider wins.
If a request sends a `provider` that conflicts with a configured `model_routes`
alias, Hermes rejects the request with `400` instead of silently remixing route
credentials with another provider.

**Bare `model` values on the OpenAI-compatible endpoints are opt-in.** Generic
OpenAI clients routinely hardcode model names (`gpt-4o`, ...), and existing
deployments rely on those falling back to the gateway default. On
`POST /v1/chat/completions` and `POST /v1/responses`, a `model` value sent
WITHOUT a `provider` is therefore ignored unless you enable:

```yaml
gateway:
  platforms:
    api_server:
      direct_model_requests: true
```

Requests that include an explicit `provider` — and the Hermes-native
`/v1/runs` and session-chat endpoints — always honor the requested model
regardless of this flag.

Example:

```json
{
  "model": "MiniMax-M3",
  "provider": "minimax",
  "model_options": {
    "reasoning_effort": "high",
    "service_tier": "priority"
  },
  "messages": [
    {"role": "user", "content": "Summarize the repo status."}
  ]
}
```

### GET /health

Health check. Returns `{"status": "ok"}`. Also available at **GET /v1/health** for OpenAI-compatible clients that expect the `/v1/` prefix.

### GET /health/detailed

Authenticated readiness check for monitoring and control planes. It reports
bounded status for the active profile's config, state database, configured
model, disk space, gateway/platform state, active API runs, pending process
completions, and active delegations. The response exposes status and counts,
not config values, credentials, paths, commands, queue payloads, or raw errors.

The public `/health` route remains a cheap liveness probe and does not run
readiness checks. A degraded readiness result still uses HTTP 200; inspect the
top-level `status` and `readiness.checks` fields.

## Runs API (streaming-friendly alternative)

In addition to `/v1/chat/completions` and `/v1/responses`, the server exposes a **runs** API for long-form sessions where the client wants to subscribe to progress events instead of managing streaming themselves.

### POST /v1/runs

Create a new agent run. Returns a `run_id` that can be used to subscribe to progress events.

```json
{
  "run_id": "run_abc123",
  "status": "started"
}
```

Runs accept a simple `input` string and optional `session_id`, `instructions`, `conversation_history`, or `previous_response_id`. When `session_id` is provided, Hermes surfaces it in the run status so external UIs can correlate runs with their own conversation IDs.

### GET /v1/runs/\{run_id\}

Poll the current run state. This is useful for dashboards that need status without holding an SSE connection open, or for UIs that reconnect after navigation.

```json
{
  "object": "hermes.run",
  "run_id": "run_abc123",
  "status": "completed",
  "session_id": "space-session",
  "model": "hermes-agent",
  "output": "Done.",
  "usage": {"input_tokens": 50, "output_tokens": 200, "total_tokens": 250}
}
```

Statuses are retained briefly after terminal states (`completed`, `failed`, or `cancelled`) for polling and UI reconciliation.

### GET /v1/runs/\{run_id\}/events

Server-Sent Events stream of the run's tool-call progress, token deltas, and lifecycle events. Designed for dashboards and thick clients that want to attach/detach without losing state.

When the agent delegates work to background subagents, the stream also carries
`subagent.start` and `subagent.complete` lifecycle events, so clients can
observe delegation outcomes — including timeouts and failures — instead of the
run going silent while a child works. The `subagent.complete` payload carries
the child's status, summary, duration, token/cost figures, and a
`child_session_id` for correlation; free-text fields pass forced secret
redaction before leaving the process. Per-tool child events
(`subagent.tool`, progress ticks) are intentionally **not** forwarded — they
are high-volume UI noise; use the per-child live transcript files for
play-by-play.

Unconsumed event buffers expire after five minutes so a detached client cannot
grow memory indefinitely. This expires transport state only: a run that is
still executing remains visible to status polling, approval, stop control, and
concurrency accounting until its executor work actually exits. A connected SSE
subscriber continues draining normally.

### POST /v1/runs/\{run_id\}/stop

Interrupt a running agent turn. The endpoint returns immediately with `{"status": "stopping"}` while Hermes asks the active agent to stop at the next safe interruption point.
The run stays tracked as `stopping` until the executor-backed work exits, then
settles as `cancelled`; requesting stop never hides a worker that is still
running.

### POST /v1/runs/\{run_id\}/approval

Resolve a pending approval for a run that is waiting on a human decision (for example, a tool call gated behind an approval policy). The body carries the approval decision; the run resumes once the decision is recorded. This endpoint is advertised in `/v1/capabilities` as the `run_approval` feature so external UIs can detect support before surfacing an approval prompt.

## Jobs API (background scheduled work)

The server exposes a lightweight jobs CRUD surface for managing scheduled / background agent runs from a remote client. All endpoints are gated behind the same bearer auth.

### GET /api/jobs

List all scheduled jobs.

### POST /api/jobs

Create a new scheduled job. Body accepts the same shape as `hermes cron` — prompt, schedule, skills, provider override, delivery target.

### GET /api/jobs/\{job_id\}

Fetch a single job's definition and last-run state.

### PATCH /api/jobs/\{job_id\}

Update fields on an existing job (prompt, schedule, etc.). Partial updates are merged.

### DELETE /api/jobs/\{job_id\}

Remove a job. Also cancels any in-flight run.

### POST /api/jobs/\{job_id\}/pause

Pause a job without deleting it. Next-scheduled-run timestamps are suspended until resumed.

### POST /api/jobs/\{job_id\}/resume

Resume a previously paused job.

### POST /api/jobs/\{job_id\}/run

Trigger the job to run immediately, out of schedule.

## Sessions API (session control over REST)

External UIs can manage Hermes sessions over REST without standing up the dashboard. All endpoints are gated by `API_SERVER_KEY` and live under `/api/sessions/*`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/sessions` | List sessions (paginated — `limit`, `offset`, `source`, `include_children`) |
| `POST` | `/api/sessions` | Create an empty session |
| `GET` | `/api/sessions/{id}` | Read session metadata |
| `PATCH` | `/api/sessions/{id}` | Update title or `end_reason` |
| `DELETE` | `/api/sessions/{id}` | Delete a session |
| `GET` | `/api/sessions/{id}/messages` | Message history for a session |
| `POST` | `/api/sessions/{id}/fork` | Branch the session via `SessionDB` lineage (matches CLI `/branch` semantics) |
| `POST` | `/api/sessions/{id}/chat` | Run one synchronous agent turn |
| `POST` | `/api/sessions/{id}/chat/stream` | SSE wrapper over a single turn — emits `assistant.delta`, `tool.started`, `tool.completed`, `run.completed` events |

`/v1/capabilities` advertises the full surface via `session_*` feature flags and `endpoints.session_*` entries so external UIs can detect support and fall back safely. Inline images are supported in `chat` and `chat/stream` payloads (multimodal-aware path).

```bash
# fork a session and run one turn
curl -X POST http://localhost:8642/api/sessions/$ID/fork \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  -d '{"title": "explore alt path"}'

# stream a turn over SSE
curl -N -X POST http://localhost:8642/api/sessions/$ID/chat/stream \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  -d '{"input": "what files changed in the last hour?"}'
```

## Skills and toolsets discovery

`GET /v1/skills` and `GET /v1/toolsets` let external clients enumerate the agent's capabilities deterministically over REST instead of asking the model. Both are read-only and gated by `API_SERVER_KEY`.

```bash
curl http://localhost:8642/v1/skills \
  -H "Authorization: Bearer $API_SERVER_KEY"
# → [{"name": "github-pr-workflow", "description": "...", "category": "..."}, ...]

curl http://localhost:8642/v1/toolsets \
  -H "Authorization: Bearer $API_SERVER_KEY"
# → [{"name": "core", "label": "...", "description": "...", "enabled": true,
#     "configured": true, "tools": ["read_file", "write_file", ...]}, ...]
```

`/v1/skills` returns the same metadata the skills hub uses internally. `/v1/toolsets` returns toolsets resolved for the `api_server` platform with the concrete `tools` list each one expands to. Both are advertised under `endpoints.*` in `/v1/capabilities`.

## Long-term memory scoping (`X-Hermes-Session-Key`)

Multi-user frontends like Open WebUI need a stable per-channel identifier for long-term memory (Honcho, etc.) that is **independent** of the transcript-scoped `X-Hermes-Session-Id` (which rotates on `/new`). Pass `X-Hermes-Session-Key` on `/v1/chat/completions`, `/v1/responses`, or `/v1/runs` and Hermes threads it through to `AIAgent(gateway_session_key=...)`, where the Honcho memory provider uses it to derive a stable scope.

```http
POST /v1/chat/completions HTTP/1.1
Authorization: Bearer ***
X-Hermes-Session-Id: transcript-alpha
X-Hermes-Session-Key: agent:main:webui:dm:user-42
```

Rules: max 256 chars, control characters (`\r`, `\n`, `\x00`) are rejected, and the value is echoed back on responses (JSON + SSE). `/v1/capabilities` advertises support via `"session_key_header": "X-Hermes-Session-Key"`. Without the key, Honcho's `per-session` strategy produces a different scope per `session_id` — exactly the behavior Hermes had before.

## System Prompt Handling

When a frontend sends a `system` message (Chat Completions) or `instructions` field (Responses API), hermes-agent **layers it on top** of its core system prompt. Your agent keeps all its tools, memory, and skills — the frontend's system prompt adds extra instructions.

This means you can customize behavior per-frontend without losing capabilities:
- Open WebUI system prompt: "You are a Python expert. Always include type hints."
- The agent still has terminal, file tools, web search, memory, etc.

## Authentication

Bearer token auth via the `Authorization` header:

```
Authorization: Bearer ***
```

Configure the key via `API_SERVER_KEY` env var. If you need a browser to call Hermes directly, also set `API_SERVER_CORS_ORIGINS` to an explicit allowlist.

### Multi-profile routing (`/p/<profile>/…`)

When [multi-profile gateway routing](/user-guide/multi-profile-gateways) is
enabled (`gateway.multiplex_profiles`), the shared listener serves every
profile through a `/p/<profile>/` URL prefix — and **authentication is bound
to the routed profile**:

- Requests to `/p/<profile>/v1/...` must present that profile's own
  `API_SERVER_KEY` (from `~/.hermes/profiles/<profile>/.env`). The default
  listener's key is rejected on named-profile prefixes.
- Unprefixed routes and `/p/default/...` keep using the default profile's key.
- A named profile with no `API_SERVER_KEY` of its own fails closed — its
  prefix is unreachable until you set one.

:::warning Breaking change (July 2026)
Before this fix, a valid default-profile key was accepted on any
`/p/<profile>/` prefix. If you relied on one shared key across profile
prefixes, set a distinct `API_SERVER_KEY` in each profile's `.env` — reused
default keys on named prefixes now return `401`.
:::

:::warning Security
The API server gives full access to hermes-agent's toolset, **including terminal commands**. `API_SERVER_KEY` is **required for every deployment**, including the default loopback bind on `127.0.0.1`. Keep `API_SERVER_CORS_ORIGINS` narrow to control browser access when you explicitly allow browser callers.

For Realtime voice, use HTTPS/WSS outside a trusted loopback or private
network. The Android client receives only the Hermes bearer credential, never
the OpenAI key. Raw audio flows over the negotiated WebRTC connection and is
not persisted by Hermes by default; finalized transcripts, function calls, and
tool results are durable session history.
:::

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_SERVER_ENABLED` | `false` | Enable the API server |
| `API_SERVER_PORT` | `8642` | HTTP server port |
| `API_SERVER_HOST` | `127.0.0.1` | Bind address (localhost only by default) |
| `API_SERVER_KEY` | _(required)_ | Bearer token for auth |
| `API_SERVER_CORS_ORIGINS` | _(none)_ | Comma-separated allowed browser origins |
| `API_SERVER_MODEL_NAME` | _(profile name)_ | Model name on `/v1/models`. Defaults to profile name, or `hermes-agent` for default profile. |

### config.yaml

The same settings can live in `~/.hermes/config.yaml` under a nested `gateway.api_server:` section:

```yaml
gateway:
  api_server:
    enabled: true
    port: 8642
    host: 127.0.0.1
    key: your-secret-key
    cors_origins: http://localhost:3000
    model_name: my-hermes
    max_concurrent_runs: 10   # concurrent-run cap; 0 disables the limit
```

`port`, `key`, `host`, `cors_origins`, and `model_name` are automatically bridged into the platform's `extra` settings, so they behave exactly like their `API_SERVER_*` environment-variable counterparts. Environment variables take precedence over `config.yaml` values. The block is also accepted under `gateway.platforms.api_server:` or a top-level `platforms.api_server:` section.

### Realtime voice settings

Realtime behavior belongs in the top-level `realtime_voice` section of
`config.yaml`; do not put these non-secret settings in `.env`.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable native Realtime session routes |
| `model` | `gpt-realtime` | OpenAI Realtime model, fixed for a logical session |
| `voice` | `marin` | Voice, fixed for a logical session |
| `transcription_model` | `gpt-4o-mini-transcribe` | Input transcription used for durable history and lifecycle hooks |
| `turn_detection.type` | `server_vad` | OpenAI turn detection (`server_vad` or `semantic_vad`) |
| `turn_detection.threshold` | `0.5` | Server-VAD activation threshold |
| `turn_detection.silence_duration_ms` | `500` | Silence required to commit a turn |
| `intermediate_speech.enabled` | `true` | Allow brief out-of-band spoken status during a long tool wait |
| `intermediate_speech.delay_seconds` | `2.5` | Delay before eligible status speech |
| `preroll.enabled` | `true` | Accept a complete wake-word startup utterance through the authenticated sideband bridge |
| `preroll.max_seconds` | `30` | Maximum PCM duration accepted by one pre-roll upload |
| `preroll.timeout_seconds` | `15` | Maximum provider-acknowledgment wait for the handoff |
| `limits.max_active_sessions` | `4` | Concurrent logical Realtime sessions per profile |
| `limits.provider_call_max_seconds` | `3300` | Proactive provider-call rotation age |
| `limits.provider_call_max_input_tokens` | `24000` | Proactive input-context rotation threshold |
| `transport.reconnect_grace_seconds` | `30` | Sideband reconnect window before renewal is required |
| `transport.call_url` | `https://api.openai.com/v1/realtime/calls` | Exact HTTPS endpoint for WebRTC call creation |
| `transport.sideband_url` | `wss://api.openai.com/v1/realtime` | Exact WSS endpoint for call-ID sideband attachment |
| `limits.approval_timeout_seconds` | `600` | Structured approval/clarification wait limit |

The full defaults also bound SDP and control-message sizes, replay history,
status-speech frequency, and creation rate. Model and voice overrides in a
session request are rejected. Change the server configuration, restart the
gateway, and create a new logical session instead.

#### OpenAI-compatible Realtime proxies

Both provider connections can be routed through an OpenAI-compatible proxy:

```yaml
realtime_voice:
  enabled: true
  model: gpt-realtime-2.1
  voice: marin
  transport:
    call_url: https://proxy.example/v1/realtime/calls
    sideband_url: wss://proxy.example/v1/realtime
```

`call_url` must accept OpenAI's multipart `sdp` and `session` fields and return
the raw SDP answer with a `Location` call identifier. `sideband_url` must attach
to that same call through a `call_id` query parameter and relay Realtime events
without rewriting them, including `session.update`,
`input_audio_buffer.clear`, `input_audio_buffer.append`, and
`input_audio_buffer.commit`. Existing query parameters are preserved when
Hermes adds `call_id`.

Hermes sends `OPENAI_API_KEY` as a bearer credential to both configured
endpoints. Only configure trusted HTTPS/WSS services, and keep credentials out
of the URLs themselves. These URLs affect only the Realtime media runtime;
configure `auxiliary.background_review` separately if retrospective
self-improvement should use the same proxy.

### Concurrent-run cap

The API server limits how many agent runs may execute at once across the OpenAI-compatible and Runs endpoints. The cap is read from `gateway.api_server.max_concurrent_runs` (default **10**; `0` disables the limit, negative values clamp to 0). When the cap is reached, new run-starting requests are rejected with **HTTP 429** `Too many concurrent runs (max N)` — clients should back off and retry.

## Security Headers

All responses include security headers:
- `X-Content-Type-Options: nosniff` — prevents MIME type sniffing
- `Referrer-Policy: no-referrer` — prevents referrer leakage

## CORS

The API server does **not** enable browser CORS by default.

For direct browser access, set an explicit allowlist:

```bash
API_SERVER_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

When CORS is enabled:
- **Preflight responses** include `Access-Control-Max-Age: 600` (10 minute cache)
- **SSE streaming responses** include CORS headers so browser EventSource clients work correctly
- **`Idempotency-Key`** is an allowed request header — clients can send it for deduplication (responses are cached by key for 5 minutes)

Most documented frontends such as Open WebUI connect server-to-server and do not need CORS at all.

## Compatible Frontends

Any frontend that supports the OpenAI API format works. Tested/documented integrations:

| Frontend | Stars | Connection |
|----------|-------|------------|
| [Open WebUI](/user-guide/messaging/open-webui) | 126k | Full guide available |
| LobeChat | 73k | Custom provider endpoint |
| LibreChat | 34k | Custom endpoint in librechat.yaml |
| AnythingLLM | 56k | Generic OpenAI provider |
| NextChat | 87k | BASE_URL env var |
| ChatBox | 39k | API Host setting |
| Jan | 26k | Remote model config |
| HF Chat-UI | 8k | OPENAI_BASE_URL |
| big-AGI | 7k | Custom endpoint |
| OpenAI Python SDK | — | `OpenAI(base_url="http://localhost:8642/v1")` |
| curl | — | Direct HTTP requests |

## Multi-User Setup with Profiles

To give multiple users their own isolated Hermes instance (separate config, memory, skills), use [profiles](/user-guide/profiles):

```bash
# Create a profile per user
hermes profile create alice
hermes profile create bob

# Configure each profile's API server on a different port. API_SERVER_* are env
# vars (not config.yaml keys), so write them to each profile's .env:
cat >> ~/.hermes/profiles/alice/.env <<EOF
API_SERVER_ENABLED=true
API_SERVER_PORT=8643
API_SERVER_KEY=alice-secret
EOF

cat >> ~/.hermes/profiles/bob/.env <<EOF
API_SERVER_ENABLED=true
API_SERVER_PORT=8644
API_SERVER_KEY=bob-secret
EOF

# Start each profile's gateway
hermes -p alice gateway &
hermes -p bob gateway &
```

Each profile's API server automatically advertises the profile name as the model ID:

- `http://localhost:8643/v1/models` → model `alice`
- `http://localhost:8644/v1/models` → model `bob`

In Open WebUI, add each as a separate connection. The model dropdown shows `alice` and `bob` as distinct models, each backed by a fully isolated Hermes instance. See the [Open WebUI guide](/user-guide/messaging/open-webui#multi-user-setup-with-profiles) for details.

## Limitations

- **Response storage** — stored responses (for `previous_response_id`) are persisted in SQLite and survive gateway restarts. Max 100 stored responses (LRU eviction).
- **No file upload** — inline images are supported on both `/v1/chat/completions` and `/v1/responses`, but uploaded files (`file`, `input_file`, `file_id`) and non-image document inputs are not supported through the API.
- **Simple OpenAI clients still see an alias** — `/v1/models` advertises the
  stable Hermes alias (`hermes-agent` or the active profile name). Richer
  clients can send explicit `provider` / `model_options` overrides on requests.
- **Realtime is voice-only** — audio is streamed over WebRTC; the control
  socket carries status and authorization UI, not an assistant text stream.
- **Streamed audio cannot be rewritten afterward** — post-response observation
  hooks still run, but `transform_llm_output` cannot alter speech the listener
  has already heard.
- **OpenAI Realtime only in v1** — there is no provider fallback to a text
  model when a native voice call fails.

## Proxy Mode

The API server also serves as the backend for **gateway proxy mode**. When another Hermes gateway instance is configured with `GATEWAY_PROXY_URL` pointing at this API server, it forwards all messages here instead of running its own agent. This enables split deployments — for example, a Docker container handling Matrix E2EE that relays to a host-side agent.

See [Matrix Proxy Mode](/user-guide/messaging/matrix#proxy-mode-e2ee-on-macos) for the full setup guide.
