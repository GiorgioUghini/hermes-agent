"""Hermes-owned state machine for one logical native voice session."""

from __future__ import annotations

import asyncio
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import logging
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Optional
import uuid

from agent.turn_context import TurnContext, build_host_turn_context
from agent.turn_finalizer import finalize_host_turn
from gateway.realtime.controls import ControlEventBroker
from gateway.realtime.openai_sideband import (
    OpenAIRealtimeError,
    OpenAIRealtimeSideband,
)
from gateway.realtime.protocol import (
    RealtimeFunctionCall,
    RealtimeProtocolError,
    RealtimeVoiceConfig,
    PREROLL_APPEND_CHUNK_MILLISECONDS,
    PREROLL_CHANNELS,
    PREROLL_MIN_MILLISECONDS,
    PREROLL_SAMPLE_RATE_HZ,
    PREROLL_SAMPLE_WIDTH_BYTES,
    SESSION_STATES,
    conversation_function_call_event,
    conversation_message_event,
    function_call_output_event,
    function_calls_from_event,
    input_audio_buffer_append_event,
    input_audio_buffer_clear_event,
    input_audio_buffer_commit_event,
    response_create_event,
    response_transcript,
    session_turn_detection_update_event,
    validate_preroll_idempotency_key,
)


logger = logging.getLogger(__name__)


REALTIME_SYSTEM_GUIDANCE = """\
You are speaking with the user through a live voice connection.
- Keep spoken progress natural and brief. Before potentially slow tool work, say one short acknowledgement.
- Use relevant skills by calling skills_list or skill_view before relying on them. Do not preload every skill.
- Use skill_manage in the foreground only when the user explicitly requests a skill change or the current task requires it. Retrospective maintenance happens separately after the turn.
- Never treat spoken confirmation as authorization for a privileged operation. Privileged approvals arrive only through Hermes structured controls.
- Do not narrate hidden arguments, secrets, internal policies, or automatic memory/skill review.
"""


@dataclass
class _PendingPrompt:
    prompt_id: str
    kind: str
    event: threading.Event
    response: Optional[str] = None


@dataclass
class _PrerollRequest:
    digest: str
    call_id: str
    task: Optional[asyncio.Task]
    result: Optional[dict[str, Any]] = None
    failure: Optional["_PrerollFailure"] = None


@dataclass(frozen=True)
class _PrerollFailure:
    kind: str
    message: str
    code: str
    status: Optional[int] = None
    retryable: bool = False


@dataclass
class _PrerollWaiter:
    event_type: str
    future: asyncio.Future
    matches: Optional[Callable[[Mapping[str, Any]], bool]] = None


def prepare_realtime_agent(
    agent,
    conversation_history: list[dict[str, Any]],
    *,
    frozen_instructions: Optional[str] = None,
) -> str:
    """Freeze the canonical Hermes prompt before creating an OpenAI call."""

    from agent.conversation_loop import _restore_or_build_system_prompt

    if frozen_instructions:
        agent._cached_system_prompt = frozen_instructions
    else:
        _restore_or_build_system_prompt(agent, None, conversation_history)
        if agent.ephemeral_system_prompt:
            agent._cached_system_prompt = (
                f"{agent._cached_system_prompt or ''}\n\n"
                f"{agent.ephemeral_system_prompt}"
            ).strip()
    # Text transports add this field only while building an API request.
    # Realtime sends one immutable session prompt instead, so fold it into the
    # frozen snapshot above and clear the side channel to prevent duplication.
    agent.ephemeral_system_prompt = None
    agent._ensure_db_session()
    agent._host_system_prompt_frozen = True
    agent._skip_mcp_refresh = True
    agent.memory_notifications = "off"
    agent.background_review_callback = lambda _message: None
    return agent._cached_system_prompt


def _tool_call_object(call: RealtimeFunctionCall) -> SimpleNamespace:
    return SimpleNamespace(
        id=call.call_id,
        type="function",
        function=SimpleNamespace(name=call.name, arguments=call.arguments),
    )


def _tool_call_dict(call: RealtimeFunctionCall) -> dict[str, Any]:
    return {
        "id": call.call_id,
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments},
    }


def _tool_output_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        from agent.tool_executor import _multimodal_text_summary

        summary = _multimodal_text_summary(value)
        if isinstance(summary, str):
            return summary
    except Exception:
        pass
    return json.dumps(value, ensure_ascii=False, default=str)


class RealtimeVoiceSession:
    """Serialize provider events, Hermes tools, persistence, and controls."""

    def __init__(
        self,
        *,
        session_id: str,
        agent: Any,
        config: RealtimeVoiceConfig,
        frozen_instructions: str,
        conversation_history: list[dict[str, Any]],
        sideband: OpenAIRealtimeSideband,
        call_id: str,
        profile_name: str = "",
        frozen_tools: Optional[list[dict[str, Any]]] = None,
        call_started_at: Optional[float] = None,
    ):
        self.session_id = session_id
        self.agent = agent
        self.config = config
        self.frozen_instructions = frozen_instructions
        self.frozen_tools = json.loads(
            json.dumps(
                frozen_tools if isinstance(frozen_tools, list) else (agent.tools or []),
                ensure_ascii=False,
                default=str,
            )
        )
        self.messages = list(conversation_history)
        self.sideband = sideband
        self.call_id = call_id
        self.profile_name = profile_name
        self.broker = ControlEventBroker(
            session_id,
            buffer_events=config.control_buffer_events,
            subscriber_queue_events=config.control_subscriber_queue_events,
        )

        self.state = "negotiating"
        self.created_at = time.time()
        self.call_started_at = call_started_at or self.created_at
        self.last_activity_at = self.created_at
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._rotation_task: Optional[asyncio.Task] = None
        self._event_lock = asyncio.Lock()
        self._preroll_request_lock = asyncio.Lock()
        self._preroll_requests: OrderedDict[str, _PrerollRequest] = OrderedDict()
        self._preroll_tasks: set[asyncio.Task] = set()
        self._preroll_active = False
        self._preroll_waiter: Optional[_PrerollWaiter] = None
        self._preroll_error_future: Optional[asyncio.Future] = None
        self._preroll_event_ids: set[str] = set()
        self._renewal_required = False
        self._turn: Optional[TurnContext] = None
        self._turn_response_count = 0
        self._pending_next_inputs: deque[tuple[str, str]] = deque()
        self._response_calls: dict[str, dict[str, RealtimeFunctionCall]] = defaultdict(dict)
        self._response_transcripts: dict[str, str] = {}
        self._processed_responses: set[str] = set()
        self._processed_calls: set[str] = set()
        self._tool_task: Optional[asyncio.Task] = None
        self._intermediate_task: Optional[asyncio.Task] = None
        self._status_response_active = False
        self._status_response_requested = False
        self._status_response_ids: set[str] = set()
        self._status_watchdog_task: Optional[asyncio.Task] = None
        self._continuation_pending = False
        self._transcript_wait_tasks: dict[str, asyncio.Task] = {}
        self._handled_input_items: set[str] = set()
        self._provider_input_tokens = 0
        self._rotation_notified = False
        self._barge_in_during_response = False
        self._skip_current_tool_batch = False
        self._closing = False
        self._closed = False
        self._logical_end = False
        self._close_lock = asyncio.Lock()
        self._pending_approvals: deque[str] = deque()
        self._pending_prompts: dict[str, _PendingPrompt] = {}
        self._pending_prompts_lock = threading.Lock()
        self._review_lock = threading.Lock()
        self._review_pending: Optional[
            tuple[int, list[dict[str, Any]], bool, bool]
        ] = None
        self._review_running = False
        self._review_thread: Optional[threading.Thread] = None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def can_renew(self) -> bool:
        return (
            not self._closed
            and not self._closing
            and not self._preroll_active
            and self._turn is None
            and (self._tool_task is None or self._tool_task.done())
            and not self._pending_approvals
        )

    @property
    def idle_timeout_eligible(self) -> bool:
        """Return whether inactivity cleanup may safely end this session."""

        with self._pending_prompts_lock:
            has_pending_prompts = bool(self._pending_prompts)
        return (
            not self._closed
            and not self._closing
            and not self._preroll_active
            and self._turn is None
            and (self._tool_task is None or self._tool_task.done())
            and not self._pending_approvals
            and not has_pending_prompts
        )

    def session_config(self) -> dict[str, Any]:
        return self.config.openai_session(
            instructions=self.frozen_instructions,
            tools=self.frozen_tools,
        )

    async def ingest_preroll_audio(
        self,
        audio: bytes,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Commit one complete, locally endpointed first utterance."""

        if not self.config.preroll_enabled:
            raise RealtimeProtocolError(
                "Wake-word pre-roll upload is disabled",
                code="preroll_disabled",
            )
        key = validate_preroll_idempotency_key(idempotency_key)
        if not isinstance(audio, bytes):
            raise RealtimeProtocolError(
                "Pre-roll audio must be raw PCM bytes",
                code="invalid_preroll_audio",
            )
        frame_bytes = PREROLL_CHANNELS * PREROLL_SAMPLE_WIDTH_BYTES
        minimum_bytes = (
            PREROLL_SAMPLE_RATE_HZ
            * frame_bytes
            * PREROLL_MIN_MILLISECONDS
            // 1000
        )
        if len(audio) < minimum_bytes:
            raise RealtimeProtocolError(
                f"Pre-roll audio must contain at least {PREROLL_MIN_MILLISECONDS} ms",
                code="preroll_audio_too_short",
            )
        if len(audio) > self.config.preroll_max_bytes:
            raise RealtimeProtocolError(
                "Pre-roll audio exceeds the configured duration limit",
                code="preroll_audio_too_large",
            )
        if len(audio) % frame_bytes:
            raise RealtimeProtocolError(
                "Pre-roll PCM contains an incomplete sample frame",
                code="invalid_preroll_audio",
            )

        digest = hashlib.sha256(audio).hexdigest()
        async with self._preroll_request_lock:
            existing = self._preroll_requests.get(key)
            if existing is not None and existing.call_id != self.call_id:
                self._preroll_requests.pop(key, None)
                existing = None
            if existing is not None:
                if existing.digest != digest:
                    raise RealtimeProtocolError(
                        "Idempotency-Key was already used with different audio",
                        code="preroll_idempotency_conflict",
                    )
                self._preroll_requests.move_to_end(key)
                if existing.failure is not None:
                    self._raise_preroll_failure(existing.failure)
                if existing.result is not None:
                    return dict(existing.result)
                task = existing.task
                if task is None:
                    raise RuntimeError("Pre-roll request has no cached outcome")
            else:
                if any(
                    request.task is not None
                    and not request.task.done()
                    and request.call_id == self.call_id
                    for request in self._preroll_requests.values()
                ):
                    raise RealtimeProtocolError(
                        "Another pre-roll upload is already in progress",
                        code="preroll_in_progress",
                    )
                self._prune_preroll_requests()
                task = asyncio.create_task(
                    self._run_preroll_upload(audio, key),
                    name=f"realtime-preroll-{self.session_id}",
                )
                self._preroll_tasks.add(task)
                request = _PrerollRequest(
                    digest=digest,
                    call_id=self.call_id,
                    task=task,
                )
                self._preroll_requests[key] = request
                task.add_done_callback(
                    lambda completed, request_key=key, record=request: (
                        self._preroll_task_done(request_key, record, completed)
                    )
                )
        return await asyncio.shield(task)

    def _preroll_task_done(
        self,
        key: str,
        request: _PrerollRequest,
        task: asyncio.Task,
    ) -> None:
        self._preroll_tasks.discard(task)
        if self._preroll_requests.get(key) is not request:
            return
        if task.cancelled():
            self._preroll_requests.pop(key, None)
            return
        try:
            request.result = dict(task.result())
        except Exception as exc:
            request.failure = self._sanitize_preroll_failure(exc)
        finally:
            request.task = None

    @staticmethod
    def _sanitize_preroll_failure(exc: Exception) -> _PrerollFailure:
        message = str(exc).replace("\r", " ").replace("\n", " ").strip()[:1000]
        if isinstance(exc, RealtimeProtocolError):
            return _PrerollFailure(
                kind="protocol",
                message=message,
                code=exc.code,
            )
        if isinstance(exc, OpenAIRealtimeError):
            return _PrerollFailure(
                kind="provider",
                message=message,
                code=exc.code,
                status=exc.status,
                retryable=exc.retryable,
            )
        return _PrerollFailure(
            kind="internal",
            message=message or "Pre-roll upload failed",
            code="realtime_internal_error",
        )

    @staticmethod
    def _raise_preroll_failure(failure: _PrerollFailure) -> None:
        if failure.kind == "protocol":
            raise RealtimeProtocolError(failure.message, code=failure.code)
        if failure.kind == "provider":
            raise OpenAIRealtimeError(
                failure.message,
                code=failure.code,
                status=failure.status,
                retryable=failure.retryable,
            )
        raise RuntimeError(failure.message)

    def _prune_preroll_requests(self) -> None:
        while len(self._preroll_requests) >= 16:
            for key, request in list(self._preroll_requests.items()):
                if request.task is None or request.task.done():
                    self._preroll_requests.pop(key)
                    break
            else:
                raise RealtimeProtocolError(
                    "Pre-roll request capacity is temporarily exhausted",
                    code="preroll_in_progress",
                )

    async def _run_preroll_upload(
        self,
        audio: bytes,
        idempotency_key: str,
    ) -> dict[str, Any]:
        with self._pending_prompts_lock:
            has_pending_prompts = bool(self._pending_prompts)
        async with self._event_lock:
            tool_running = self._tool_task is not None and not self._tool_task.done()
            if self._renewal_required:
                raise RealtimeProtocolError(
                    "The provider call must be renewed before another pre-roll upload",
                    code="session_renewal_required",
                )
            if (
                self._closing
                or self._closed
                or self.state != "ready"
                or self._turn is not None
                or tool_running
                or self._pending_approvals
                or has_pending_prompts
                or self._preroll_active
            ):
                raise RealtimeProtocolError(
                    "Realtime session is busy; pre-roll requires an idle session",
                    code="session_busy",
                )
            self._preroll_active = True
            self.last_activity_at = time.time()

        duration_ms = round(
            len(audio)
            * 1000
            / (
                PREROLL_SAMPLE_RATE_HZ
                * PREROLL_CHANNELS
                * PREROLL_SAMPLE_WIDTH_BYTES
            )
        )
        self.broker.publish(
            "audio.preroll_started",
            {"duration_ms": duration_ms, "audio_bytes": len(audio)},
        )
        self._reset_preroll_provider_signals()
        self._preroll_error_future = asyncio.get_running_loop().create_future()
        deadline = (
            asyncio.get_running_loop().time() + self.config.preroll_timeout_seconds
        )
        vad_may_be_disabled = False
        buffer_may_be_dirty = True
        try:
            vad_may_be_disabled = True
            await self._send_preroll_and_wait(
                session_turn_detection_update_event(
                    None,
                    event_id=self._event_id("preroll_vad_off"),
                ),
                "session.updated",
                deadline,
                matches=lambda event: self._session_update_matches(event, None),
            )
            await self._send_preroll_and_wait(
                input_audio_buffer_clear_event(
                    event_id=self._event_id("preroll_clear")
                ),
                "input_audio_buffer.cleared",
                deadline,
            )
            buffer_may_be_dirty = False

            chunk_bytes = (
                PREROLL_SAMPLE_RATE_HZ
                * PREROLL_CHANNELS
                * PREROLL_SAMPLE_WIDTH_BYTES
                * PREROLL_APPEND_CHUNK_MILLISECONDS
                // 1000
            )
            for offset in range(0, len(audio), chunk_bytes):
                buffer_may_be_dirty = True
                event = input_audio_buffer_append_event(
                    audio[offset : offset + chunk_bytes],
                    event_id=self._event_id("preroll_append"),
                )
                self._preroll_event_ids.add(event["event_id"])
                await self._send_preroll_event(event, deadline)
                self._raise_preroll_provider_error()

            committed = await self._send_preroll_and_wait(
                input_audio_buffer_commit_event(
                    event_id=self._event_id("preroll_commit")
                ),
                "input_audio_buffer.committed",
                deadline,
            )
            buffer_may_be_dirty = False
            await self._restore_preroll_vad(deadline)
            vad_may_be_disabled = False
            item_id = str(committed.get("item_id") or "").strip()
            result = {
                "version": 1,
                "session_id": self.session_id,
                "call_id": self.call_id,
                "idempotency_key": idempotency_key,
                "status": "committed",
                "item_id": item_id,
                "audio_bytes": len(audio),
                "duration_ms": duration_ms,
            }
            self.broker.publish(
                "audio.preroll_committed",
                {
                    "provider_item_id": item_id,
                    "duration_ms": duration_ms,
                    "audio_bytes": len(audio),
                },
            )
            return result
        except (Exception, asyncio.CancelledError):
            if vad_may_be_disabled and not self._closed:
                try:
                    await self._cleanup_failed_preroll(
                        buffer_may_be_dirty=buffer_may_be_dirty,
                    )
                except Exception:
                    logger.exception(
                        "Could not clean provider audio after pre-roll failure for %s",
                        self.session_id,
                    )
                await self._mark_preroll_renewal_required("preroll_upload_failed")
            raise
        finally:
            self._reset_preroll_provider_signals()
            async with self._event_lock:
                self._preroll_active = False

    async def _cleanup_failed_preroll(
        self,
        *,
        buffer_may_be_dirty: bool,
    ) -> None:
        self._reset_preroll_provider_signals()
        self._preroll_error_future = asyncio.get_running_loop().create_future()
        deadline = (
            asyncio.get_running_loop().time()
            + min(5.0, self.config.preroll_timeout_seconds)
        )
        if buffer_may_be_dirty:
            await self._send_preroll_and_wait(
                input_audio_buffer_clear_event(
                    event_id=self._event_id("preroll_cleanup_clear")
                ),
                "input_audio_buffer.cleared",
                deadline,
            )
        await self._restore_preroll_vad(deadline)

    async def _mark_preroll_renewal_required(self, reason: str) -> None:
        first_notification = not self._renewal_required
        self._renewal_required = True
        if self._closed:
            return
        self._set_state("degraded", reason=reason)
        try:
            await self._persist_runtime_state("renewal_required")
        except Exception:
            logger.exception(
                "Could not persist required pre-roll renewal for %s",
                self.session_id,
            )
        if first_notification:
            self.broker.publish("session.rotation_required", {"reason": reason})

    async def _restore_preroll_vad(
        self,
        deadline: float,
        *,
        reset_signals: bool = False,
    ) -> None:
        turn_detection = self.config.turn_detection_config()
        if reset_signals:
            self._reset_preroll_provider_signals()
            self._preroll_error_future = asyncio.get_running_loop().create_future()
        try:
            await self._send_preroll_and_wait(
                session_turn_detection_update_event(
                    turn_detection,
                    event_id=self._event_id("preroll_vad_on"),
                ),
                "session.updated",
                deadline,
                matches=lambda event: self._session_update_matches(
                    event, turn_detection
                ),
            )
        except Exception:
            if self._closed:
                raise
            self._reset_preroll_provider_signals()
            self._preroll_error_future = asyncio.get_running_loop().create_future()
            await self._send_preroll_and_wait(
                session_turn_detection_update_event(
                    turn_detection,
                    event_id=self._event_id("preroll_vad_retry"),
                ),
                "session.updated",
                asyncio.get_running_loop().time()
                + min(5.0, self.config.preroll_timeout_seconds),
                matches=lambda event: self._session_update_matches(
                    event, turn_detection
                ),
            )

    @staticmethod
    def _session_update_matches(
        event: Mapping[str, Any],
        expected: Optional[Mapping[str, Any]],
    ) -> bool:
        session = event.get("session")
        session = session if isinstance(session, Mapping) else {}
        audio = session.get("audio")
        audio = audio if isinstance(audio, Mapping) else {}
        input_config = audio.get("input")
        input_config = input_config if isinstance(input_config, Mapping) else {}
        if "turn_detection" not in input_config:
            return False
        actual = input_config.get("turn_detection")
        if expected is None:
            return actual is None
        if not isinstance(actual, Mapping):
            return False
        return all(actual.get(key) == value for key, value in expected.items())

    async def _send_preroll_event(
        self,
        event: Mapping[str, Any],
        deadline: float,
    ) -> None:
        timeout = deadline - asyncio.get_running_loop().time()
        event_type = str(event.get("type") or "provider event")
        if timeout <= 0:
            raise RealtimeProtocolError(
                f"Timed out sending {event_type}",
                code="preroll_provider_timeout",
            )
        try:
            await asyncio.wait_for(self.sideband.send(event), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RealtimeProtocolError(
                f"Timed out sending {event_type}",
                code="preroll_provider_timeout",
            ) from exc

    async def _send_preroll_and_wait(
        self,
        event: Mapping[str, Any],
        expected_event_type: str,
        deadline: float,
        *,
        matches: Optional[Callable[[Mapping[str, Any]], bool]] = None,
    ) -> Mapping[str, Any]:
        if self._preroll_waiter is not None:
            raise RuntimeError("A pre-roll provider acknowledgment is already pending")
        future = asyncio.get_running_loop().create_future()
        waiter = _PrerollWaiter(expected_event_type, future, matches)
        self._preroll_waiter = waiter
        event_id = str(event.get("event_id") or "")
        if event_id:
            self._preroll_event_ids.add(event_id)
        try:
            await self._send_preroll_event(event, deadline)
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                raise asyncio.TimeoutError
            error_future = self._preroll_error_future
            futures = {future}
            if error_future is not None:
                futures.add(error_future)
            done, _pending = await asyncio.wait(
                futures,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise asyncio.TimeoutError
            if error_future is not None and error_future in done:
                raise error_future.result()
            return future.result()
        except asyncio.TimeoutError as exc:
            raise RealtimeProtocolError(
                f"Timed out waiting for {expected_event_type}",
                code="preroll_provider_timeout",
            ) from exc
        finally:
            if self._preroll_waiter is waiter:
                self._preroll_waiter = None
            if not future.done():
                future.cancel()

    def _raise_preroll_provider_error(self) -> None:
        future = self._preroll_error_future
        if future is not None and future.done():
            raise future.result()

    def _reset_preroll_provider_signals(self) -> None:
        waiter, self._preroll_waiter = self._preroll_waiter, None
        if waiter is not None and not waiter.future.done():
            waiter.future.cancel()
        error_future, self._preroll_error_future = self._preroll_error_future, None
        if error_future is not None and not error_future.done():
            error_future.cancel()
        self._preroll_event_ids.clear()

    async def start(
        self,
        *,
        seed_history: bool = False,
        recover_review: bool = True,
        preserve_call_started_at: bool = False,
    ) -> None:
        self._loop = asyncio.get_running_loop()
        self._install_agent_callbacks()
        self._register_approval_notifications()
        self._set_state("connecting")
        await self.sideband.connect()
        if seed_history:
            await self._seed_history()
        if not preserve_call_started_at:
            self.call_started_at = time.time()
        self._receive_task = asyncio.create_task(
            self._run_sideband(), name=f"realtime-sideband-{self.session_id}"
        )
        self._set_state("ready")
        await self._persist_runtime_state("ready")
        self._rotation_task = asyncio.create_task(
            self._rotation_watch(), name=f"realtime-rotation-{self.session_id}"
        )
        if recover_review:
            await self._recover_due_review()

    def _set_state(self, state: str, **details: Any) -> None:
        if state not in SESSION_STATES:
            raise RealtimeProtocolError(
                f"Invalid realtime state: {state}", code="invalid_session_state"
            )
        self.state = state
        self.broker.publish("session.state", {"state": state, **details})

    def _event_id(self, prefix: str) -> str:
        return f"hermes_{prefix}_{uuid.uuid4().hex}"

    def _install_agent_callbacks(self) -> None:
        self.agent.tool_start_callback = self._tool_start_sync
        self.agent.tool_complete_callback = self._tool_complete_sync
        self.agent.clarify_callback = self._clarify_sync
        self.agent._background_review_dispatch = (
            self._schedule_background_review_sync
        )

    async def _persist_runtime_state(self, state: Optional[str] = None) -> None:
        db = getattr(self.agent, "_session_db", None)
        if db is None:
            return
        persisted_state = (
            "renewal_required" if self._renewal_required else (state or self.state)
        )
        await asyncio.to_thread(
            db.save_realtime_session_state,
            self.session_id,
            provider_call_id=self.call_id,
            provider_call_started_at=self.call_started_at,
            state=persisted_state,
            model=self.config.model,
            voice=self.config.voice,
            frozen_instructions=self.frozen_instructions,
            frozen_tools=self.frozen_tools,
        )

    def _schedule_background_review_sync(
        self,
        *,
        messages_snapshot: list[dict[str, Any]],
        review_memory: bool = False,
        review_skills: bool = False,
    ) -> None:
        """Persist and coalesce review work without blocking the voice turn."""

        if self._logical_end:
            return
        db = getattr(self.agent, "_session_db", None)
        if db is None:
            return
        boundary = db.get_latest_message_id(self.session_id)
        if boundary is None:
            return

        with self._review_lock:
            pending = self._review_pending
            if pending is not None:
                _, _, old_memory, old_skills = pending
                review_memory = review_memory or old_memory
                review_skills = review_skills or old_skills
            self._review_pending = (
                boundary,
                list(messages_snapshot),
                bool(review_memory),
                bool(review_skills),
            )
            db.mark_realtime_review_due(
                self.session_id,
                boundary_message_id=boundary,
                review_memory=review_memory,
                review_skills=review_skills,
            )
            if self._review_running:
                return
            self._start_review_thread_locked()

    def _start_review_thread_locked(self) -> None:
        """Start the review drain while ``_review_lock`` is held."""

        from tools.thread_context import propagate_context_to_thread

        self._review_running = True
        self._review_thread = threading.Thread(
            target=propagate_context_to_thread(self._drain_background_reviews),
            daemon=True,
            name=f"realtime-review-{self.session_id[:24]}",
        )
        self._review_thread.start()

    def _drain_background_reviews(self) -> None:
        from agent.background_review import spawn_background_review_thread

        db = getattr(self.agent, "_session_db", None)
        if db is None:
            with self._review_lock:
                self._review_running = False
            return

        active_pending = None
        try:
            while True:
                with self._review_lock:
                    pending = self._review_pending
                    self._review_pending = None
                    if pending is None:
                        self._review_running = False
                        return
                active_pending = pending
                boundary, snapshot, review_memory, review_skills = pending
                if not db.mark_realtime_review_running(
                    self.session_id, boundary_message_id=boundary
                ):
                    active_pending = None
                    continue
                target, _prompt = spawn_background_review_thread(
                    self.agent,
                    snapshot,
                    review_memory=review_memory,
                    review_skills=review_skills,
                )
                success = bool(target())
                with self._review_lock:
                    newer_pending = self._review_pending is not None
                if not newer_pending:
                    db.finish_realtime_review(
                        self.session_id,
                        boundary_message_id=boundary,
                        success=success,
                        error=None
                        if success
                        else "background review did not complete",
                    )
                active_pending = None
        except Exception:
            logger.warning(
                "Realtime background-review drain failed for %s",
                self.session_id,
                exc_info=True,
            )
            with self._review_lock:
                newer_pending = self._review_pending
                if active_pending is not None:
                    if newer_pending is None:
                        self._review_pending = active_pending
                    else:
                        (
                            newer_boundary,
                            newer_snapshot,
                            newer_memory,
                            newer_skills,
                        ) = newer_pending
                        self._review_pending = (
                            newer_boundary,
                            newer_snapshot,
                            newer_memory or active_pending[2],
                            newer_skills or active_pending[3],
                        )
                self._review_running = False
                if newer_pending is not None and not self._logical_end:
                    self._start_review_thread_locked()

    async def _recover_due_review(self) -> None:
        db = getattr(self.agent, "_session_db", None)
        if db is None:
            return
        record = await asyncio.to_thread(
            db.get_realtime_session_state, self.session_id
        )
        if not record or record.get("review_state") not in {
            "due",
            "running",
            "failed",
        }:
            return
        boundary = record.get("review_boundary_message_id")
        if not isinstance(boundary, int):
            return
        history = await asyncio.to_thread(db.get_messages, self.session_id)
        snapshot = [
            message
            for message in history
            if isinstance(message.get("id"), int) and message["id"] <= boundary
        ]
        if not snapshot:
            return
        await asyncio.to_thread(
            self._run_in_session_context,
            self._schedule_background_review_sync,
            messages_snapshot=snapshot,
            review_memory=bool(record.get("review_memory")),
            review_skills=bool(record.get("review_skills")),
        )

    def _publish_threadsafe(self, event_type: str, data: Mapping[str, Any]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self.broker.publish, event_type, dict(data))

    def _tool_start_sync(
        self,
        call_id: str,
        name: str,
        display_args: Mapping[str, Any],
    ) -> None:
        self._publish_threadsafe(
            "tool.started",
            {
                "call_id": call_id,
                "name": name,
                "argument_keys": sorted(str(key) for key in display_args),
            },
        )

    def _tool_complete_sync(
        self,
        call_id: str,
        name: str,
        _display_args: Mapping[str, Any],
        _result: Any,
    ) -> None:
        self._publish_threadsafe(
            "tool.completed", {"call_id": call_id, "name": name}
        )

    def _register_approval_notifications(self) -> None:
        from tools.approval import register_gateway_notify

        register_gateway_notify(self.session_id, self._approval_notify_sync)

    def _approval_notify_sync(self, approval_data: Mapping[str, Any]) -> None:
        approval_id = f"approval_{uuid.uuid4().hex}"
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def publish() -> None:
            from gateway.run import _redact_approval_command

            self._pending_approvals.append(approval_id)
            self.broker.publish(
                "approval.request",
                {
                    "approval_id": approval_id,
                    "command": _redact_approval_command(approval_data.get("command")),
                    "description": str(approval_data.get("description") or "")[:1000],
                    "choices": ["once", "session", "always", "deny"],
                },
            )

        loop.call_soon_threadsafe(publish)

    def _blocking_prompt(
        self,
        kind: str,
        *,
        event_data: Mapping[str, Any],
    ) -> str:
        prompt = _PendingPrompt(
            prompt_id=f"{kind}_{uuid.uuid4().hex}",
            kind=kind,
            event=threading.Event(),
        )
        with self._pending_prompts_lock:
            self._pending_prompts[prompt.prompt_id] = prompt

        loop = self._loop
        if loop is None or loop.is_closed():
            with self._pending_prompts_lock:
                self._pending_prompts.pop(prompt.prompt_id, None)
            return ""
        payload = {"prompt_id": prompt.prompt_id, **dict(event_data)}
        loop.call_soon_threadsafe(
            self.broker.publish, f"{kind}.request", payload
        )
        prompt.event.wait(timeout=self.config.approval_timeout_seconds)
        with self._pending_prompts_lock:
            self._pending_prompts.pop(prompt.prompt_id, None)
        return prompt.response or ""

    def _clarify_sync(
        self,
        question: str,
        choices: Iterable[str] | None,
        multi_select: bool = False,
    ) -> str:
        return self._blocking_prompt(
            "clarification",
            event_data={
                "question": str(question)[:4000],
                "choices": [str(choice)[:500] for choice in (choices or [])],
                "multi_select": bool(multi_select),
            },
        )

    def _secret_sync(self) -> str:
        return self._blocking_prompt(
            "secret",
            event_data={"label": "Sudo password required"},
        )

    async def resolve_approval(
        self,
        *,
        choice: str,
        approval_id: str = "",
        resolve_all: bool = False,
        reason: Optional[str] = None,
    ) -> int:
        if choice not in {"once", "session", "always", "deny"}:
            raise RealtimeProtocolError(
                "Invalid approval choice", code="invalid_approval_choice"
            )
        if approval_id and (
            not self._pending_approvals
            or self._pending_approvals[0] != approval_id
        ):
            raise RealtimeProtocolError(
                "Approval is no longer pending", code="stale_approval"
            )
        from tools.approval import resolve_gateway_approval

        resolved = resolve_gateway_approval(
            self.session_id,
            choice,
            resolve_all=resolve_all,
            reason=reason,
        )
        for _ in range(min(resolved, len(self._pending_approvals))):
            resolved_id = self._pending_approvals.popleft()
            self.broker.publish(
                "approval.resolved",
                {"approval_id": resolved_id, "choice": choice},
            )
        return resolved

    def resolve_prompt(self, prompt_id: str, value: str, *, kind: str) -> bool:
        with self._pending_prompts_lock:
            prompt = self._pending_prompts.get(prompt_id)
            if prompt is None or prompt.kind != kind:
                return False
            prompt.response = value
            prompt.event.set()
            return True

    async def handle_control_command(self, command: Mapping[str, Any]) -> None:
        command_type = command["type"]
        data = command.get("data") or {}
        request_id = command.get("request_id") or ""
        if command_type == "approval.respond":
            await self.resolve_approval(
                choice=data["choice"],
                approval_id=str(data.get("approval_id") or ""),
                resolve_all=bool(data.get("all", False)),
                reason=str(data.get("reason") or "") or None,
            )
        elif command_type == "clarification.respond":
            if not self.resolve_prompt(
                str(data.get("prompt_id") or ""),
                str(data.get("value") or ""),
                kind="clarification",
            ):
                raise RealtimeProtocolError(
                    "Clarification is no longer pending",
                    code="stale_clarification",
                )
        elif command_type == "secret.respond":
            if not self.resolve_prompt(
                str(data.get("prompt_id") or ""),
                str(data.get("value") or ""),
                kind="secret",
            ):
                raise RealtimeProtocolError(
                    "Secret prompt is no longer pending", code="stale_secret_prompt"
                )
        elif command_type == "session.close":
            await self.close(reason="client_closed", end_session=True)
        elif command_type == "session.ping":
            self.broker.publish("session.pong", {"request_id": request_id})
        if request_id and command_type != "session.ping":
            self.broker.publish(
                "control.ack",
                {"request_id": request_id, "command": command_type},
            )

    async def _run_sideband(self) -> None:
        deadline: Optional[float] = None
        while not self._closed:
            try:
                await self.sideband.receive_loop()
                if self._closed:
                    return
                raise OpenAIRealtimeError(
                    "OpenAI sideband closed unexpectedly",
                    code="sideband_closed",
                    retryable=True,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._closed:
                    return
                if deadline is None:
                    deadline = time.monotonic() + self.config.reconnect_grace_seconds
                    self._set_state("degraded", reason="sideband_disconnected")
                    self.broker.publish(
                        "warning",
                        {
                            "code": "sideband_disconnected",
                            "message": "Realtime control connection was interrupted.",
                        },
                    )
                if time.monotonic() >= deadline:
                    await self._persist_runtime_state("renewal_required")
                    self.broker.publish(
                        "session.rotation_required",
                        {"reason": "sideband_reconnect_exhausted"},
                    )
                    return
                await asyncio.sleep(0.5)
                try:
                    await self.sideband.reconnect()
                except Exception:
                    logger.debug(
                        "Realtime sideband reconnect failed for %s",
                        self.session_id,
                        exc_info=True,
                    )
                    continue
                deadline = None
                self._set_state("ready", reconnected=True)
                self.broker.publish("session.reconnected", {})

    async def handle_provider_event(self, event: Mapping[str, Any]) -> None:
        """Consume one OpenAI server event without blocking the receive loop."""

        self.last_activity_at = time.time()
        async with self._event_lock:
            event_type = str(event.get("type") or "")
            waiter = self._preroll_waiter
            if (
                waiter is not None
                and event_type == waiter.event_type
                and not waiter.future.done()
                and (waiter.matches is None or waiter.matches(event))
            ):
                waiter.future.set_result(dict(event))
            if event_type in {"session.updated", "input_audio_buffer.cleared"}:
                return
            if event_type == "input_audio_buffer.speech_started":
                self.broker.publish("vad.speech_started", {})
                if self.state in {"responding", "tool_wait"}:
                    self.broker.publish("response.interrupted", {"reason": "barge_in"})
                if self.state == "responding":
                    self._barge_in_during_response = True
                return
            if event_type == "input_audio_buffer.speech_stopped":
                self.broker.publish("vad.speech_stopped", {})
                return
            if event_type == "input_audio_buffer.committed":
                item_id = str(event.get("item_id") or "").strip()
                if item_id:
                    self._start_transcript_wait(item_id)
                return
            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = str(event.get("transcript") or "").strip()
                await self._handle_input_once(
                    transcript or "[Voice transcript was empty]",
                    str(event.get("item_id") or ""),
                )
                return
            if event_type == "conversation.item.input_audio_transcription.failed":
                self.broker.publish(
                    "warning",
                    {
                        "code": "transcription_failed",
                        "message": "Input transcription was unavailable; audio may still be understood.",
                    },
                )
                await self._handle_input_once(
                    "[Voice transcript unavailable; use the committed audio input.]",
                    str(event.get("item_id") or ""),
                )
                return
            if event_type in {
                "response.output_audio_transcript.delta",
                "response.audio_transcript.delta",
            }:
                response_id = str(event.get("response_id") or "")
                self._response_transcripts[response_id] = (
                    self._response_transcripts.get(response_id, "")
                    + str(event.get("delta") or "")
                )
                return
            if event_type in {
                "response.output_audio_transcript.done",
                "response.audio_transcript.done",
            }:
                response_id = str(event.get("response_id") or "")
                transcript = str(event.get("transcript") or "")
                if transcript:
                    self._response_transcripts[response_id] = transcript
                return
            if event_type == "response.created":
                response = event.get("response")
                response = response if isinstance(response, Mapping) else {}
                metadata = response.get("metadata")
                metadata = metadata if isinstance(metadata, Mapping) else {}
                response_id = str(response.get("id") or "")
                if (
                    metadata.get("hermes_kind") == "tool_wait_status"
                    or self._status_response_requested
                ):
                    self._status_response_requested = False
                    self._status_response_active = True
                    if response_id:
                        self._status_response_ids.add(response_id)
                else:
                    self._set_state("responding")
                return
            if event_type == "response.output_item.done":
                for call in function_calls_from_event(event):
                    key = call.response_id or str(event.get("response_id") or "")
                    self._response_calls[key][call.call_id] = call
                return
            if event_type == "response.done":
                await self._handle_response_done(event)
                return
            if event_type == "rate_limits.updated":
                self.broker.publish(
                    "rate_limits.updated",
                    {"rate_limits": event.get("rate_limits") or []},
                )
                return
            if event_type == "error":
                error = event.get("error")
                error = error if isinstance(error, Mapping) else {}
                event_id = str(error.get("event_id") or event.get("event_id") or "")
                provider_error = OpenAIRealtimeError(
                    str(error.get("message") or "Realtime provider error")[:1000],
                    code=str(error.get("code") or "openai_realtime_error"),
                    retryable=False,
                )
                if (
                    self._preroll_active
                    and event_id in self._preroll_event_ids
                    and self._preroll_error_future is not None
                    and not self._preroll_error_future.done()
                ):
                    self._preroll_error_future.set_result(provider_error)
                self.broker.publish(
                    "error",
                    {
                        "fatal": False,
                        "code": provider_error.code,
                        "message": str(provider_error),
                        "event_id": event_id,
                    },
                )

    def _start_transcript_wait(self, item_id: str) -> None:
        if (
            not item_id
            or item_id in self._handled_input_items
            or item_id in self._transcript_wait_tasks
        ):
            return
        self._transcript_wait_tasks[item_id] = asyncio.create_task(
            self._transcript_timeout(item_id),
            name=f"realtime-transcript-{self.session_id}",
        )

    async def _transcript_timeout(self, item_id: str) -> None:
        try:
            await asyncio.sleep(self.config.transcription_timeout_seconds)
            self.broker.publish(
                "warning",
                {
                    "code": "transcription_timeout",
                    "message": (
                        "Input transcription did not finish before the response "
                        "deadline; Hermes continued with the committed audio."
                    ),
                },
            )
            await self._handle_input_once(
                "[Voice transcript unavailable; use the committed audio input.]",
                item_id,
            )
        finally:
            self._transcript_wait_tasks.pop(item_id, None)

    async def _handle_input_once(self, transcript: str, item_id: str) -> None:
        if item_id and item_id in self._handled_input_items:
            return
        if item_id:
            self._handled_input_items.add(item_id)
            task = self._transcript_wait_tasks.pop(item_id, None)
            if task is not None and task is not asyncio.current_task():
                task.cancel()
        await self._handle_input(transcript, item_id)

    async def _handle_input(self, transcript: str, item_id: str) -> None:
        tool_running = self._tool_task is not None and not self._tool_task.done()
        if self._turn is None:
            await self._begin_turn(transcript, item_id)
            return
        if tool_running:
            if self._skip_current_tool_batch:
                self._pending_next_inputs.append((transcript, item_id))
                self.broker.publish(
                    "turn.steered",
                    {
                        "during": "response",
                        "transcript_available": not transcript.startswith("["),
                    },
                )
                return
            self.agent._skip_unstarted_tool_calls = True
            self.agent.steer(f"[User voice steer]\n{transcript}")
            self.broker.publish(
                "turn.steered",
                {"during": "tool", "transcript_available": not transcript.startswith("[")},
            )
            return
        self._pending_next_inputs.append((transcript, item_id))
        self.broker.publish(
            "turn.steered",
            {"during": "response", "transcript_available": not transcript.startswith("[")},
        )

    async def _begin_turn(self, transcript: str, item_id: str) -> None:
        if self._closed:
            return
        self._set_state("listening")
        history = list(self.messages)
        self._turn = await asyncio.to_thread(
            self._run_in_session_context,
            build_host_turn_context,
            self.agent,
            transcript,
            conversation_history=history,
            persist_user_display_kind="realtime_voice",
            persist_user_display_metadata={
                "provider_item_id": item_id,
                "transcript_available": not transcript.startswith("[Voice transcript"),
            },
        )
        self.messages = self._turn.messages
        self._turn_response_count = 0
        self.broker.publish(
            "turn.input_committed",
            {
                "provider_item_id": item_id,
                "transcript_available": not transcript.startswith("[Voice transcript"),
            },
        )
        await self._persist_runtime_state("turn_active")
        dynamic_instructions = self._response_instructions(self._turn)
        await self._send_response_create(instructions=dynamic_instructions)

    def _response_instructions(self, turn: TurnContext) -> Optional[str]:
        active = turn.active_system_prompt or self.frozen_instructions
        user_message = (
            turn.messages[turn.current_turn_user_idx]
            if 0 <= turn.current_turn_user_idx < len(turn.messages)
            else {}
        )
        clean = user_message.get("content") if isinstance(user_message, Mapping) else ""
        api_content = (
            user_message.get("api_content") if isinstance(user_message, Mapping) else None
        )
        suffix = ""
        if (
            isinstance(clean, str)
            and isinstance(api_content, str)
            and api_content.startswith(clean)
        ):
            suffix = api_content[len(clean) :].strip()
        if active != self.frozen_instructions or suffix:
            return active + (f"\n\n{suffix}" if suffix else "")
        return None

    async def _send_response_create(
        self,
        *,
        instructions: Optional[str] = None,
        status_message: bool = False,
    ) -> None:
        if not status_message:
            if self._turn is None:
                return
            if not self.agent.iteration_budget.consume():
                self.broker.publish(
                    "error",
                    {
                        "fatal": True,
                        "code": "iteration_budget_exhausted",
                        "message": "The realtime turn exceeded its iteration budget.",
                    },
                )
                await self._finalize_current_turn(
                    "[Realtime turn stopped after reaching its tool iteration limit.]",
                    response_id="",
                    failed=True,
                )
                return
            self._turn_response_count += 1
            if (
                self.agent._skill_nudge_interval > 0
                and "skill_manage" in self.agent.valid_tool_names
            ):
                self.agent._iters_since_skill += 1
        await self.sideband.send(
            response_create_event(
                instructions=instructions,
                status_message=status_message,
                event_id=self._event_id("response"),
            )
        )
        if status_message:
            self._status_response_requested = True
        else:
            self._set_state("responding")

    async def _handle_response_done(self, event: Mapping[str, Any]) -> None:
        response = event.get("response")
        response = response if isinstance(response, Mapping) else {}
        response_id = str(response.get("id") or "")
        metadata = response.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        if (
            metadata.get("hermes_kind") == "tool_wait_status"
            or response_id in self._status_response_ids
        ):
            self._status_response_ids.discard(response_id)
            self._status_response_active = False
            self._status_response_requested = False
            if self._status_watchdog_task is not None:
                self._status_watchdog_task.cancel()
                self._status_watchdog_task = None
            if self._continuation_pending:
                self._continuation_pending = False
                await self._send_response_create()
            return
        if response_id and response_id in self._processed_responses:
            return
        if response_id:
            self._processed_responses.add(response_id)

        await self._record_usage(response.get("usage"))
        calls = {
            call.call_id: call for call in function_calls_from_event(event)
        }
        calls.update(self._response_calls.pop(response_id, {}))
        transcript = (
            self._response_transcripts.pop(response_id, "")
            or response_transcript(event)
        )
        if calls:
            if self._turn is None:
                self.broker.publish(
                    "error",
                    {
                        "fatal": False,
                        "code": "orphan_function_call",
                        "message": "The provider emitted a tool call without an active turn.",
                    },
                )
                return
            self._set_state("tool_wait")
            skip_unstarted = self._barge_in_during_response or bool(
                self._pending_next_inputs
            )
            self._skip_current_tool_batch = skip_unstarted
            self._tool_task = asyncio.create_task(
                self._execute_tool_batch(
                    list(calls.values()),
                    response_id=response_id,
                    spoken_preamble=transcript,
                    skip_unstarted=skip_unstarted,
                ),
                name=f"realtime-tools-{self.session_id}",
            )
            return

        status = str(response.get("status") or "completed")
        interrupted = status in {"cancelled", "canceled"} and (
            self._barge_in_during_response or bool(self._pending_next_inputs)
        )
        await self._finalize_current_turn(
            transcript or "[Assistant audio transcript unavailable]",
            response_id=response_id,
            interrupted=interrupted,
            failed=status == "failed",
        )

    async def _execute_tool_batch(
        self,
        calls: list[RealtimeFunctionCall],
        *,
        response_id: str,
        spoken_preamble: str,
        skip_unstarted: bool = False,
    ) -> None:
        try:
            outputs: dict[str, str] = {}
            new_calls: list[RealtimeFunctionCall] = []
            db = getattr(self.agent, "_session_db", None)
            for call in calls:
                if call.call_id in self._processed_calls:
                    durable = (
                        db.get_tool_result_by_call_id(self.session_id, call.call_id)
                        if db is not None
                        else None
                    )
                    if durable is not None:
                        outputs[call.call_id] = _tool_output_text(
                            durable.get("content", "")
                        )
                    continue
                durable = (
                    db.get_tool_result_by_call_id(self.session_id, call.call_id)
                    if db is not None
                    else None
                )
                if durable is not None:
                    self._processed_calls.add(call.call_id)
                    outputs[call.call_id] = _tool_output_text(
                        durable.get("content", "")
                    )
                else:
                    new_calls.append(call)

            if new_calls:
                assistant_row = {
                    "role": "assistant",
                    "content": spoken_preamble or "",
                    "tool_calls": [_tool_call_dict(call) for call in new_calls],
                    "platform_message_id": response_id or None,
                    "display_kind": "realtime_voice",
                    "display_metadata": {"provider_response_id": response_id},
                }
                assert self._turn is not None
                self._turn.messages.append(assistant_row)
                persisted = await asyncio.to_thread(
                    self._run_in_session_context,
                    self.agent._flush_messages_to_session_db,
                    self._turn.messages,
                    self._turn.conversation_history,
                )
                if persisted is False:
                    raise RuntimeError(
                        "Assistant function-call row could not be persisted"
                    )

                assistant_message = SimpleNamespace(
                    content=spoken_preamble or "",
                    tool_calls=[_tool_call_object(call) for call in new_calls],
                )
                self._intermediate_task = asyncio.create_task(
                    self._send_intermediate_after_delay(),
                    name=f"realtime-intermediate-{self.session_id}",
                )
                await asyncio.to_thread(
                    self._execute_tools_sync,
                    assistant_message,
                    self._turn.messages,
                    self._turn.effective_task_id,
                    self._turn_response_count,
                    skip_unstarted,
                )
                # The canonical executor may append a queued /steer marker to
                # the last result after its per-result flush. Persist that final
                # form before projecting any function output to Realtime.
                persisted = await asyncio.to_thread(
                    self._run_in_session_context,
                    self.agent._flush_messages_to_session_db,
                    self._turn.messages,
                    self._turn.conversation_history,
                )
                if persisted is False:
                    raise RuntimeError("Final tool result state could not be persisted")
                if getattr(self.agent, "_incremental_persistence_failed", False):
                    raise RuntimeError("Tool result could not be persisted")
                for call in new_calls:
                    result_row = next(
                        (
                            message
                            for message in reversed(self._turn.messages)
                            if isinstance(message, Mapping)
                            and message.get("role") == "tool"
                            and message.get("tool_call_id") == call.call_id
                        ),
                        None,
                    )
                    if result_row is None:
                        raise RuntimeError(
                            f"Tool {call.name} produced no durable result"
                        )
                    outputs[call.call_id] = _tool_output_text(
                        result_row.get("content", "")
                    )
                    self._processed_calls.add(call.call_id)

            for call in calls:
                output = outputs.get(call.call_id)
                if output is None:
                    durable = (
                        db.get_tool_result_by_call_id(self.session_id, call.call_id)
                        if db is not None
                        else None
                    )
                    output = _tool_output_text(
                        durable.get("content", "") if durable else
                        "[Duplicate tool call was suppressed without a recoverable result.]"
                    )
                if self._closed:
                    return
                await self.sideband.send(
                    function_call_output_event(
                        call.call_id,
                        output,
                        event_id=self._event_id("tool"),
                    )
                )

            if skip_unstarted:
                await self._finalize_current_turn(
                    spoken_preamble or "[Assistant response interrupted by new speech.]",
                    response_id=response_id,
                    interrupted=True,
                )
                return
            if self._status_response_active or self._status_response_requested:
                self._continuation_pending = True
            else:
                await self._send_response_create()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Realtime tool batch failed for %s", self.session_id)
            self.broker.publish(
                "error",
                {
                    "fatal": True,
                    "code": "tool_batch_failed",
                    "message": str(exc)[:1000],
                },
            )
            await self._finalize_current_turn(
                "[Realtime tool execution failed.]",
                response_id=response_id,
                failed=True,
            )
        finally:
            if self._intermediate_task is not None:
                self._intermediate_task.cancel()
                self._intermediate_task = None

    def _execute_tools_sync(
        self,
        assistant_message: SimpleNamespace,
        messages: list[dict[str, Any]],
        task_id: str,
        api_call_count: int,
        skip_unstarted: bool = False,
    ) -> None:
        from tools.terminal_tool import (
            set_approval_callback,
            set_sudo_password_callback,
        )

        tokens = self._bind_session_context()
        set_approval_callback(None)
        set_sudo_password_callback(self._secret_sync)
        self.agent._skip_unstarted_tool_calls = skip_unstarted
        try:
            # Realtime batches stay sequential so a barge-in can suppress calls
            # that have not started without cancelling the operation already in
            # progress. Every call still uses the canonical executor pipeline.
            self.agent._execute_tool_calls_sequential(
                assistant_message, messages, task_id, api_call_count
            )
        finally:
            self.agent._skip_unstarted_tool_calls = False
            set_sudo_password_callback(None)
            set_approval_callback(None)
            self._clear_session_context(tokens)

    async def _send_intermediate_after_delay(self) -> None:
        if not self.config.intermediate_speech_enabled:
            return
        await asyncio.sleep(self.config.intermediate_speech_delay_seconds)
        if (
            self._closed
            or self._tool_task is None
            or self._tool_task.done()
            or self._pending_approvals
            or self._status_response_active
            or self._status_response_requested
        ):
            return
        try:
            await self._send_response_create(
                instructions=(
                    "Say one short, natural sentence that you are still working and "
                    "will continue as soon as the current operation finishes. Do not "
                    "mention hidden details or invent progress."
                ),
                status_message=True,
            )
            self._status_watchdog_task = asyncio.create_task(
                self._status_response_watchdog(),
                name=f"realtime-status-watchdog-{self.session_id}",
            )
        except Exception as exc:
            logger.warning(
                "Could not send realtime intermediate speech for %s: %s",
                self.session_id,
                exc,
            )

    async def _status_response_watchdog(self) -> None:
        await asyncio.sleep(max(5.0, self.config.intermediate_speech_delay_seconds * 4))
        if (
            self._closed
            or not (self._status_response_active or self._status_response_requested)
        ):
            return
        self.broker.publish(
            "warning",
            {
                "code": "intermediate_response_timeout",
                "message": "A progress utterance timed out and was cancelled.",
            },
        )
        try:
            await self.sideband.send(
                {"type": "response.cancel", "event_id": self._event_id("cancel")}
            )
        except Exception:
            logger.debug("Could not cancel stalled status response", exc_info=True)
        self._status_response_active = False
        self._status_response_requested = False
        self._status_response_ids.clear()
        if self._continuation_pending:
            self._continuation_pending = False
            await self._send_response_create()

    async def _finalize_current_turn(
        self,
        transcript: str,
        *,
        response_id: str,
        interrupted: bool = False,
        failed: bool = False,
    ) -> None:
        turn = self._turn
        if turn is None:
            return
        turn.messages.append(
            {
                "role": "assistant",
                "content": transcript,
                "finish_reason": "interrupted" if interrupted else "stop",
                "platform_message_id": response_id or None,
                "display_kind": "realtime_voice",
                "display_metadata": {
                    "provider_response_id": response_id,
                    "interrupted": interrupted,
                },
            }
        )
        self.agent._turn_received_provider_response = True
        result = await asyncio.to_thread(
            self._run_in_session_context,
            finalize_host_turn,
            self.agent,
            turn,
            final_response=transcript,
            api_call_count=self._turn_response_count,
            interrupted=interrupted,
            failed=failed,
            turn_exit_reason=(
                "realtime_interrupted"
                if interrupted
                else "realtime_failed"
                if failed
                else "text_response(realtime_voice)"
            ),
        )
        self.messages = list(result.get("messages") or turn.messages)
        self._turn = None
        self._turn_response_count = 0
        self._tool_task = None
        self._barge_in_during_response = False
        self._skip_current_tool_batch = False
        self.broker.publish(
            "turn.completed",
            {
                "provider_response_id": response_id,
                "interrupted": interrupted,
                "failed": failed,
            },
        )
        self._set_state("ready")
        await self._persist_runtime_state("ready")
        if self._pending_next_inputs:
            transcript_parts: list[str] = []
            item_id = ""
            while self._pending_next_inputs:
                next_text, next_item_id = self._pending_next_inputs.popleft()
                transcript_parts.append(next_text)
                item_id = next_item_id or item_id
            await self._begin_turn("\n".join(transcript_parts), item_id)

    async def _record_usage(self, usage: Any) -> None:
        if not isinstance(usage, Mapping):
            return
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        details = usage.get("input_token_details")
        details = details if isinstance(details, Mapping) else {}
        cached = int(details.get("cached_tokens") or 0)
        self.agent.session_input_tokens += input_tokens
        self.agent.session_output_tokens += output_tokens
        self.agent.session_cache_read_tokens += cached
        self.agent.session_total_tokens += input_tokens + output_tokens
        self.agent._last_turn_usage = dict(usage)
        self._provider_input_tokens = max(self._provider_input_tokens, input_tokens)
        self.broker.publish(
            "usage.updated",
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_input_tokens": cached,
            },
        )
        if (
            not self._rotation_notified
            and self._provider_input_tokens
            >= self.config.provider_call_max_input_tokens
        ):
            self._rotation_notified = True
            await self._persist_runtime_state("renewal_required")
            self.broker.publish(
                "session.rotation_required",
                {
                    "reason": "provider_context_threshold",
                    "input_tokens": self._provider_input_tokens,
                    "threshold": self.config.provider_call_max_input_tokens,
                },
            )

    def _bind_session_context(self) -> list:
        from gateway.session_context import set_session_vars

        return set_session_vars(
            platform="realtime_voice",
            source="realtime_voice",
            chat_id=self.session_id,
            chat_type="private",
            session_key=self.session_id,
            session_id=self.session_id,
            profile=self.profile_name,
            async_delivery=True,
        )

    @staticmethod
    def _clear_session_context(tokens: list) -> None:
        from gateway.session_context import clear_session_vars

        clear_session_vars(tokens)

    def _run_in_session_context(self, function, *args, **kwargs):
        tokens = self._bind_session_context()
        try:
            return function(*args, **kwargs)
        finally:
            self._clear_session_context(tokens)

    async def _rotation_watch(self) -> None:
        expires_at = self.call_started_at + self.config.provider_call_max_seconds
        warning_delay = max(0.0, expires_at - 60 - time.time())
        await asyncio.sleep(warning_delay)
        if not self._closed:
            self._rotation_notified = True
            await self._persist_runtime_state("renewal_required")
            self.broker.publish(
                "session.rotation_required",
                {
                    "reason": "provider_call_age",
                    "renew_within_seconds": max(0, int(expires_at - time.time())),
                },
            )

    async def replace_sideband(
        self,
        *,
        sideband: OpenAIRealtimeSideband,
        call_id: str,
    ) -> None:
        if not self.can_renew:
            raise RealtimeProtocolError(
                "Session is busy and cannot rotate yet", code="session_busy"
            )
        self._set_state("rotating")
        old_task, self._receive_task = self._receive_task, None
        if old_task is not None:
            old_task.cancel()
        await self.sideband.close()
        self.sideband = sideband
        self.call_id = call_id
        self._preroll_requests.clear()
        await self.sideband.connect()
        await self._seed_history()
        self.call_started_at = time.time()
        self._provider_input_tokens = 0
        self._rotation_notified = False
        self._renewal_required = False
        if self._rotation_task is not None:
            self._rotation_task.cancel()
        self._rotation_task = asyncio.create_task(
            self._rotation_watch(), name=f"realtime-rotation-{self.session_id}"
        )
        self._receive_task = asyncio.create_task(
            self._run_sideband(), name=f"realtime-sideband-{self.session_id}"
        )
        self._set_state("ready", rotated=True)
        await self._persist_runtime_state("ready")
        self.broker.publish("session.rotated", {"call_id": call_id})

    async def _seed_history(self) -> None:
        from agent.message_content import flatten_message_text

        start = max(0, len(self.messages) - self.config.history_message_limit)
        while (
            start > 0
            and isinstance(self.messages[start], Mapping)
            and self.messages[start].get("role") == "tool"
        ):
            start -= 1
        candidates = self.messages[start:]
        seeded_call_ids: set[str] = set()
        for message in candidates:
            if not isinstance(message, Mapping):
                continue
            role = message.get("role")
            if role == "tool":
                call_id = str(message.get("tool_call_id") or "")
                if call_id not in seeded_call_ids:
                    continue
                await self.sideband.send(
                    function_call_output_event(
                        call_id,
                        _tool_output_text(message.get("content", "")),
                        event_id=self._event_id("history_tool"),
                    )
                )
                continue
            api_content = message.get("api_content")
            text = flatten_message_text(
                api_content if api_content is not None else message.get("content")
            ).strip()
            if role not in {"user", "assistant"}:
                continue
            if text:
                await self.sideband.send(
                    conversation_message_event(
                        role,
                        text,
                        event_id=self._event_id("history"),
                    )
                )
            if role != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for raw_call in tool_calls:
                if not isinstance(raw_call, Mapping):
                    continue
                function = raw_call.get("function")
                function = function if isinstance(function, Mapping) else {}
                call_id = str(raw_call.get("id") or "")
                name = str(function.get("name") or "")
                if not call_id or not name:
                    continue
                await self.sideband.send(
                    conversation_function_call_event(
                        call_id,
                        name,
                        function.get("arguments", ""),
                        event_id=self._event_id("history_call"),
                    )
                )
                seeded_call_ids.add(call_id)

    async def close(
        self,
        *,
        reason: str = "closed",
        end_session: bool = False,
    ) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            self._logical_end = end_session
            if self._renewal_required or self._preroll_active:
                # The provider may currently have VAD disabled. A fresh call is
                # required because shutdown cancels the acknowledgment sequence.
                recoverable_state = "renewal_required"
            else:
                recoverable_state = (
                    "turn_active" if self._turn is not None else "ready"
                )
            if not end_session:
                try:
                    await self._persist_runtime_state(recoverable_state)
                except Exception:
                    logger.warning(
                        "Could not persist recoverable realtime state for %s",
                        self.session_id,
                        exc_info=True,
                    )
            self._set_state("closing", reason=reason)
            self._closed = True
            from tools.approval import unregister_gateway_notify

            unregister_gateway_notify(self.session_id)
            with self._pending_prompts_lock:
                prompts = list(self._pending_prompts.values())
                self._pending_prompts.clear()
            for prompt in prompts:
                prompt.event.set()
            for task in (
                self._receive_task,
                self._rotation_task,
                self._intermediate_task,
                self._status_watchdog_task,
                *self._preroll_tasks,
                *self._transcript_wait_tasks.values(),
            ):
                if task is not None:
                    task.cancel()
            current_task = asyncio.current_task()
            cancelled = [
                task
                for task in (
                    self._receive_task,
                    self._rotation_task,
                    self._intermediate_task,
                    self._status_watchdog_task,
                    *self._preroll_tasks,
                    *self._transcript_wait_tasks.values(),
                )
                if task is not None and task is not current_task
            ]
            if cancelled:
                await asyncio.gather(*cancelled, return_exceptions=True)
            self._reset_preroll_provider_signals()
            if self._tool_task is not None and not self._tool_task.done():
                # Started tools intentionally survive barge-in, but explicit
                # session destruction waits for their thread-bound work rather
                # than orphaning an in-process side effect.
                try:
                    await asyncio.wait_for(
                        asyncio.shield(self._tool_task),
                        timeout=self.config.reconnect_grace_seconds,
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            await self.sideband.close()
            if end_session:
                db = getattr(self.agent, "_session_db", None)
                if db is not None:
                    try:
                        await asyncio.to_thread(
                            db.delete_realtime_session_state, self.session_id
                        )
                        await asyncio.to_thread(
                            db.end_session, self.session_id, reason
                        )
                    except Exception:
                        logger.warning(
                            "Could not end realtime session %s",
                            self.session_id,
                            exc_info=True,
                        )
            self._set_state("closed", reason=reason)
