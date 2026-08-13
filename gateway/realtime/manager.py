"""Admission, lookup, renewal, and cleanup for realtime voice sessions."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, replace
import json
import logging
import re
import time
from typing import Any, Callable, Mapping, Optional
import uuid

from gateway.realtime.openai_sideband import (
    OpenAIRealtimeCallClient,
    OpenAIRealtimeSideband,
)
from gateway.realtime.device_actions import (
    DEVICE_CAPABILITY_CATALOG,
    DeviceCatalogError,
    DeviceSessionBinding,
)
from gateway.realtime.protocol import (
    DeviceAdvertisement,
    RealtimeCall,
    RealtimeProtocolError,
    RealtimeVoiceConfig,
)
from gateway.realtime.session import RealtimeVoiceSession, prepare_realtime_agent


logger = logging.getLogger(__name__)

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")


def _normalize_frozen_tools(value: Any, fallback: Any = None) -> list[dict[str, Any]]:
    candidate = value if isinstance(value, list) else fallback
    if not isinstance(candidate, list):
        return []
    return [dict(tool) for tool in candidate if isinstance(tool, Mapping)]


def _tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    function = function if isinstance(function, Mapping) else tool
    return str(function.get("name") or "")


class RealtimeSessionError(RuntimeError):
    """A logical-session operation failed with a client-safe code."""

    def __init__(self, message: str, *, code: str, status: int):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class CreatedRealtimeSession:
    session: RealtimeVoiceSession
    answer_sdp: str
    call_id: str


class RealtimeSessionManager:
    """Own every live logical voice session for one API-server process."""

    def __init__(
        self,
        *,
        config: RealtimeVoiceConfig,
        api_key: str,
        agent_factory: Callable[[str], Any],
        profile_name: str = "",
        safety_identifier: str = "",
        call_client: Optional[OpenAIRealtimeCallClient] = None,
        sideband_factory: Optional[
            Callable[[str, Callable], OpenAIRealtimeSideband]
        ] = None,
    ):
        if not config.enabled:
            raise ValueError("Realtime voice is disabled")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.config = config
        self._api_key = api_key
        self._agent_factory = agent_factory
        self._profile_name = profile_name
        self._call_client = call_client or OpenAIRealtimeCallClient(
            api_key,
            call_url=config.call_url,
            request_timeout_seconds=config.request_timeout_seconds,
            safety_identifier=safety_identifier,
        )
        self._sideband_factory = sideband_factory or (
            lambda call_id, handler: OpenAIRealtimeSideband(
                api_key,
                call_id,
                handler,
                websocket_url=config.sideband_url,
                connect_timeout_seconds=config.connect_timeout_seconds,
            )
        )
        self._sessions: dict[str, RealtimeVoiceSession] = {}
        self._lock = asyncio.Lock()
        self._creation_times: deque[float] = deque()
        self._maintenance_task: Optional[asyncio.Task] = None

    @property
    def active_count(self) -> int:
        return sum(1 for session in self._sessions.values() if not session.closed)

    def get(self, session_id: str) -> Optional[RealtimeVoiceSession]:
        session = self._sessions.get(session_id)
        if session is None or session.closed:
            return None
        return session

    def require(self, session_id: str) -> RealtimeVoiceSession:
        session = self.get(session_id)
        if session is None:
            raise RealtimeSessionError(
                "Realtime session was not found",
                code="session_not_found",
                status=404,
            )
        return session

    def _logical_config(self, record: Optional[dict[str, Any]]) -> RealtimeVoiceConfig:
        """Keep model and voice stable across restart and provider-call rotation."""

        if not record:
            return self.config
        return replace(
            self.config,
            model=str(record.get("model") or self.config.model),
            voice=str(record.get("voice") or self.config.voice),
        )

    @staticmethod
    def _device_binding(
        advertisement: Optional[DeviceAdvertisement],
    ) -> Optional[DeviceSessionBinding]:
        if advertisement is None:
            return None
        try:
            return DEVICE_CAPABILITY_CATALOG.negotiate(
                protocol_version=advertisement.protocol_version,
                instance_id=advertisement.instance_id,
                capabilities=advertisement.capabilities,
            )
        except DeviceCatalogError as exc:
            raise RealtimeSessionError(
                str(exc),
                code="invalid_device_capabilities",
                status=400,
            ) from exc

    @staticmethod
    def _persisted_device_binding(
        record: Optional[dict[str, Any]],
    ) -> Optional[DeviceSessionBinding]:
        if not record or not record.get("device_context"):
            return None
        try:
            return DEVICE_CAPABILITY_CATALOG.binding_from_persisted(
                record.get("device_context")
            )
        except DeviceCatalogError as exc:
            raise RealtimeSessionError(
                "The persisted device capability version is unavailable; renew the session",
                code="device_capability_renewal_required",
                status=409,
            ) from exc

    @staticmethod
    def _base_tools(
        agent: Any,
        record: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not record:
            return _normalize_frozen_tools(agent.tools)
        persisted_base = record.get("frozen_base_tools")
        if isinstance(persisted_base, list):
            return _normalize_frozen_tools(persisted_base)
        return DEVICE_CAPABILITY_CATALOG.strip_device_tools(
            _normalize_frozen_tools(record.get("frozen_tools"), agent.tools)
        )

    @staticmethod
    def _combine_tools(
        base_tools: list[dict[str, Any]],
        binding: Optional[DeviceSessionBinding],
    ) -> list[dict[str, Any]]:
        device_tools = binding.model_tools() if binding is not None else []
        base_names = {_tool_name(tool) for tool in base_tools}
        conflicts = base_names & {
            _tool_name(tool) for tool in device_tools
        }
        if conflicts:
            raise RealtimeSessionError(
                f"Device tool names conflict with base tools: {sorted(conflicts)}",
                code="device_tool_name_conflict",
                status=500,
            )
        return [dict(tool) for tool in base_tools] + device_tools

    @staticmethod
    def _require_device_persistence(
        agent: Any,
        device_binding: Optional[DeviceSessionBinding],
    ) -> None:
        if (
            device_binding is not None
            and device_binding.tools
            and getattr(agent, "_session_db", None) is None
        ):
            raise RealtimeSessionError(
                "Device actions require durable session persistence",
                code="device_persistence_unavailable",
                status=503,
            )

    async def require_active(self, session_id: str) -> RealtimeVoiceSession:
        """Return a live actor, reattaching to a recoverable provider call."""

        session = self.get(session_id)
        if session is not None:
            return session
        async with self._lock:
            session = self.get(session_id)
            if session is not None:
                return session
            return await self._recover_active_session(session_id)

    async def _prepare_agent(
        self, session_id: str
    ) -> tuple[
        Any,
        list[dict[str, Any]],
        str,
        Optional[dict[str, Any]],
        list[str],
    ]:
        agent = await asyncio.to_thread(self._agent_factory, session_id)
        db = getattr(agent, "_session_db", None)
        recovered_device_calls: list[str] = []
        record = (
            await asyncio.to_thread(db.get_realtime_session_state, session_id)
            if db is not None
            else None
        )
        if db is not None:
            recovered_device_calls = await asyncio.to_thread(
                self._recover_device_invocations,
                db,
                session_id,
                bool(record and record.get("state") == "turn_active"),
            )
        history = await asyncio.to_thread(self._load_history, agent, session_id)
        frozen_instructions = await asyncio.to_thread(
            prepare_realtime_agent,
            agent,
            history,
            frozen_instructions=(
                str(record.get("frozen_instructions") or "") if record else None
            ),
        )
        return (
            agent,
            history,
            frozen_instructions,
            record,
            recovered_device_calls,
        )

    def _recover_device_invocations(
        self,
        db: Any,
        session_id: str,
        recover_active_turn: bool = False,
    ) -> list[str]:
        """Finish crash-interrupted device calls without redispatching them."""

        recovered: list[str] = []
        messages = db.get_messages(session_id)
        rows = db.list_unfinished_realtime_device_invocations(session_id)
        for row in rows:
            invocation_id = str(row.get("invocation_id") or "")
            if row.get("status") == "pending":
                if row.get("dispatch_state") == "claimed":
                    output = json.dumps(
                        {
                            "success": False,
                            "error": {
                                "code": "device_action_not_dispatched",
                                "message": (
                                    "Hermes restarted before this device action "
                                    "entered dispatch. The device did not receive it."
                                ),
                                "retryable": False,
                            },
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    effect_disposition = "none"
                elif row.get("side_effecting"):
                    output = json.dumps(
                        {
                            "success": False,
                            "error": {
                                "code": "outcome_unknown",
                                "message": (
                                    "Hermes restarted before the device returned a "
                                    "result. The action may have reached the device; "
                                    "Hermes will not retry it with a new invocation ID."
                                ),
                                "retryable": False,
                            },
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    effect_disposition = "unknown"
                else:
                    output = json.dumps(
                        {
                            "success": False,
                            "error": {
                                "code": "device_action_interrupted",
                                "message": (
                                    "Hermes restarted before the device returned a "
                                    "result."
                                ),
                                "retryable": False,
                            },
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    effect_disposition = "none"
                if not db.record_realtime_device_invocation_result(
                    session_id,
                    invocation_id,
                    output=output,
                    effect_disposition=effect_disposition,
                    result_source="recovery",
                ):
                    raise RuntimeError(
                        f"Could not recover device invocation {invocation_id}"
                    )
                row = db.get_realtime_device_invocation(
                    session_id,
                    invocation_id,
                )
            if not row or not isinstance(row.get("output"), str):
                raise RuntimeError(
                    f"Device invocation {invocation_id} has no durable result"
                )
            if not db.complete_realtime_device_invocation(
                session_id,
                invocation_id,
            ):
                current = db.get_realtime_device_invocation(
                    session_id,
                    invocation_id,
                )
                if not current or current.get("status") != "completed":
                    raise RuntimeError(
                        f"Could not complete device invocation {invocation_id}"
                    )
            recovered.append(invocation_id)

        for message in messages:
            if message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for raw_call in tool_calls:
                if not isinstance(raw_call, Mapping):
                    continue
                function = raw_call.get("function")
                function = function if isinstance(function, Mapping) else {}
                invocation_id = str(
                    raw_call.get("id") or raw_call.get("call_id") or ""
                )
                tool_name = str(function.get("name") or "")
                tool = DEVICE_CAPABILITY_CATALOG.tool(tool_name)
                if (
                    not invocation_id
                    or tool is None
                    or db.get_tool_result_by_call_id(session_id, invocation_id)
                    is not None
                    or db.get_realtime_device_invocation(
                        session_id,
                        invocation_id,
                    )
                    is not None
                ):
                    continue
                output = json.dumps(
                    {
                        "success": False,
                        "error": {
                            "code": "device_action_not_dispatched",
                            "message": (
                                "Hermes restarted before this device action was "
                                "dispatched. The device did not receive it."
                            ),
                            "retryable": False,
                        },
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                created = db.create_realtime_device_invocation(
                    session_id,
                    invocation_id=invocation_id,
                    tool_name=tool_name,
                    side_effecting=tool.side_effecting,
                    request_data={"recovered_without_dispatch": True},
                    expires_at_epoch_ms=int(time.time() * 1000),
                )
                captured = created and db.record_realtime_device_invocation_result(
                    session_id,
                    invocation_id,
                    output=output,
                    effect_disposition="none",
                    result_source="recovery",
                )
                if not captured:
                    raise RuntimeError(
                        f"Could not reconcile device invocation {invocation_id}"
                    )
                if not db.complete_realtime_device_invocation(
                    session_id,
                    invocation_id,
                ):
                    raise RuntimeError(
                        f"Could not complete device invocation {invocation_id}"
                    )
                recovered.append(invocation_id)

        if recover_active_turn and not recovered:
            for message in reversed(messages):
                if message.get("role") != "assistant":
                    continue
                tool_calls = message.get("tool_calls")
                if not isinstance(tool_calls, list) or not tool_calls:
                    break
                for raw_call in tool_calls:
                    if not isinstance(raw_call, Mapping):
                        continue
                    invocation_id = str(
                        raw_call.get("id") or raw_call.get("call_id") or ""
                    )
                    invocation = db.get_realtime_device_invocation(
                        session_id,
                        invocation_id,
                    )
                    if (
                        invocation_id
                        and invocation is not None
                        and isinstance(invocation.get("output"), str)
                    ):
                        recovered.append(invocation_id)
                break
            if recovered:
                rearmed = db.require_realtime_device_continuations(
                    session_id,
                    recovered,
                )
                if rearmed != len(recovered):
                    raise RuntimeError(
                        "Could not re-arm the interrupted device continuation"
                    )

        recovered_set = set(recovered)
        continuation_calls: list[str] = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            call_ids = [
                str(call.get("id") or call.get("call_id") or "")
                for call in tool_calls
                if isinstance(call, Mapping)
            ]
            if not recovered_set.intersection(call_ids):
                continue
            first_recovered_index = next(
                index
                for index, call_id in enumerate(call_ids)
                if call_id in recovered_set
            )
            for call_index, raw_call in enumerate(tool_calls):
                if not isinstance(raw_call, Mapping):
                    continue
                invocation_id = str(
                    raw_call.get("id") or raw_call.get("call_id") or ""
                )
                if not invocation_id:
                    continue
                function = raw_call.get("function")
                function = function if isinstance(function, Mapping) else {}
                tool_name = str(function.get("name") or "")
                if (
                    db.get_tool_result_by_call_id(session_id, invocation_id)
                    is None
                ):
                    invocation = db.get_realtime_device_invocation(
                        session_id,
                        invocation_id,
                    )
                    if invocation is not None:
                        output = invocation.get("output")
                        effect_disposition = invocation.get("effect_disposition")
                        if not isinstance(output, str):
                            raise RuntimeError(
                                f"Recovered invocation {invocation_id} has no output"
                            )
                    else:
                        if call_index < first_recovered_index:
                            output = json.dumps(
                                {
                                    "success": False,
                                    "error": {
                                        "code": "outcome_unknown",
                                        "message": (
                                            "Hermes restarted while this server "
                                            "action may have been executing. Its "
                                            "outcome is unknown; do not retry a "
                                            "side effect without user confirmation."
                                        ),
                                        "retryable": False,
                                    },
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            effect_disposition = "unknown"
                        else:
                            output = (
                                "[Tool execution skipped - Hermes restarted before "
                                "this server action started, so the model must "
                                "reevaluate it.]"
                            )
                            effect_disposition = "none"
                    db.append_message(
                        session_id,
                        role="tool",
                        content=output,
                        tool_name=tool_name,
                        tool_call_id=invocation_id,
                        effect_disposition=effect_disposition,
                    )
                if invocation_id not in continuation_calls:
                    continuation_calls.append(invocation_id)
        return continuation_calls

    async def _recover_active_session(self, session_id: str) -> RealtimeVoiceSession:
        (
            agent,
            history,
            frozen_instructions,
            record,
            recovered_device_calls,
        ) = await self._prepare_agent(session_id)
        if not record or not record.get("provider_call_id"):
            raise RealtimeSessionError(
                "Realtime session was not found",
                code="session_not_found",
                status=404,
            )
        if record.get("state") not in {"ready", "listening"}:
            raise RealtimeSessionError(
                "The prior provider call cannot be safely reattached; renew it",
                code="session_renewal_required",
                status=409,
            )
        call_started_at = float(record.get("provider_call_started_at") or 0)
        if (
            call_started_at <= 0
            or time.time() - call_started_at >= self.config.provider_call_max_seconds
        ):
            raise RealtimeSessionError(
                "The prior provider call has expired; renew it",
                code="session_renewal_required",
                status=409,
            )

        session_ref: dict[str, RealtimeVoiceSession] = {}

        async def handle_event(event):
            session = session_ref.get("session")
            if session is not None:
                await session.handle_provider_event(event)

        call_id = str(record["provider_call_id"])
        logical_config = self._logical_config(record)
        agent.model = logical_config.model
        sideband = self._sideband_factory(call_id, handle_event)
        base_tools = self._base_tools(agent, record)
        device_binding = self._persisted_device_binding(record)
        session = RealtimeVoiceSession(
            session_id=session_id,
            agent=agent,
            config=logical_config,
            frozen_instructions=frozen_instructions,
            frozen_tools=_normalize_frozen_tools(
                record.get("frozen_tools"),
                self._combine_tools(base_tools, device_binding),
            ),
            frozen_base_tools=base_tools,
            device_binding=device_binding,
            conversation_history=history,
            sideband=sideband,
            call_id=call_id,
            call_started_at=call_started_at,
            profile_name=self._profile_name,
        )
        session_ref["session"] = session
        try:
            await session.start(preserve_call_started_at=True)
            if recovered_device_calls:
                await session.resume_recovered_device_turn(
                    recovered_device_calls,
                    outputs_seeded=False,
                )
        except Exception:
            await session.close(reason="recovery_failed", end_session=False)
            raise
        self._sessions[session_id] = session
        self._ensure_maintenance_task()
        return session

    async def create_session(
        self,
        offer_sdp: str,
        *,
        requested_session_id: Optional[str] = None,
        device: Optional[DeviceAdvertisement] = None,
    ) -> CreatedRealtimeSession:
        async with self._lock:
            await self._prune_locked()
            self._check_creation_rate()
            if self.active_count >= self.config.max_active_sessions:
                raise RealtimeSessionError(
                    "Realtime session capacity has been reached",
                    code="realtime_overloaded",
                    status=429,
                )
            session_id = self._normalize_session_id(requested_session_id)
            if session_id in self._sessions and not self._sessions[session_id].closed:
                raise RealtimeSessionError(
                    "Realtime session is already active; use renew",
                    code="session_already_active",
                    status=409,
                )

            (
                agent,
                history,
                frozen_instructions,
                prior_state,
                recovered_device_calls,
            ) = await self._prepare_agent(session_id)
            base_tools = self._base_tools(agent, prior_state)
            device_binding = self._device_binding(device)
            self._require_device_persistence(agent, device_binding)
            frozen_tools = self._combine_tools(base_tools, device_binding)
            logical_config = self._logical_config(prior_state)
            agent.model = logical_config.model
            session_config = logical_config.openai_session(
                instructions=frozen_instructions,
                tools=frozen_tools,
            )
            call = await self._call_client.create_call(offer_sdp, session_config)
            session_ref: dict[str, RealtimeVoiceSession] = {}

            async def handle_event(event):
                session = session_ref.get("session")
                if session is not None:
                    await session.handle_provider_event(event)

            sideband = self._sideband_factory(call.call_id, handle_event)
            session = RealtimeVoiceSession(
                session_id=session_id,
                agent=agent,
                config=logical_config,
                frozen_instructions=frozen_instructions,
                frozen_tools=frozen_tools,
                frozen_base_tools=base_tools,
                device_binding=device_binding,
                conversation_history=history,
                sideband=sideband,
                call_id=call.call_id,
                profile_name=self._profile_name,
            )
            session_ref["session"] = session
            try:
                await session.start(seed_history=bool(history))
                if recovered_device_calls:
                    await session.resume_recovered_device_turn(
                        recovered_device_calls,
                        outputs_seeded=bool(history),
                    )
            except Exception:
                await session.close(reason="negotiation_failed", end_session=False)
                raise
            self._sessions[session_id] = session
            self._creation_times.append(time.monotonic())
            self._ensure_maintenance_task()
            return CreatedRealtimeSession(
                session=session,
                answer_sdp=call.answer_sdp,
                call_id=call.call_id,
            )

    async def renew_session(
        self,
        session_id: str,
        offer_sdp: str,
        *,
        device: Optional[DeviceAdvertisement] = None,
    ) -> CreatedRealtimeSession:
        async with self._lock:
            session = self.get(session_id)
            if session is not None:
                if not session.can_renew:
                    raise RealtimeSessionError(
                        "Realtime session is busy; wait for the current turn to settle",
                        code="session_busy",
                        status=409,
                    )
                device_binding = self._device_binding(device)
                self._require_device_persistence(session.agent, device_binding)
                frozen_tools = self._combine_tools(
                    session.frozen_base_tools,
                    device_binding,
                )
                call = await self._call_client.create_call(
                    offer_sdp,
                    session.session_config(tools=frozen_tools),
                )
                sideband = self._sideband_factory(
                    call.call_id,
                    session.handle_provider_event,
                )
                try:
                    await session.replace_sideband(
                        sideband=sideband,
                        call_id=call.call_id,
                        frozen_tools=frozen_tools,
                        device_binding=device_binding,
                    )
                except RealtimeProtocolError as exc:
                    await sideband.close()
                    raise RealtimeSessionError(
                        str(exc),
                        code=exc.code,
                        status=409,
                    ) from exc
                return CreatedRealtimeSession(
                    session=session,
                    answer_sdp=call.answer_sdp,
                    call_id=call.call_id,
                )
        return await self._renew_persisted_session(
            session_id,
            offer_sdp,
            device=device,
        )

    async def _renew_persisted_session(
        self,
        session_id: str,
        offer_sdp: str,
        *,
        device: Optional[DeviceAdvertisement] = None,
    ) -> CreatedRealtimeSession:
        """Create a fresh provider call for a logical session after restart."""

        async with self._lock:
            existing = self.get(session_id)
            if existing is not None:
                if not existing.can_renew:
                    raise RealtimeSessionError(
                        "Realtime session is busy; wait for the current turn to settle",
                        code="session_busy",
                        status=409,
                    )
                device_binding = self._device_binding(device)
                self._require_device_persistence(existing.agent, device_binding)
                frozen_tools = self._combine_tools(
                    existing.frozen_base_tools,
                    device_binding,
                )
                call = await self._call_client.create_call(
                    offer_sdp, existing.session_config(tools=frozen_tools)
                )
                sideband = self._sideband_factory(
                    call.call_id, existing.handle_provider_event
                )
                await existing.replace_sideband(
                    sideband=sideband,
                    call_id=call.call_id,
                    frozen_tools=frozen_tools,
                    device_binding=device_binding,
                )
                return CreatedRealtimeSession(
                    session=existing,
                    answer_sdp=call.answer_sdp,
                    call_id=call.call_id,
                )

            await self._prune_locked()
            if self.active_count >= self.config.max_active_sessions:
                raise RealtimeSessionError(
                    "Realtime session capacity has been reached",
                    code="realtime_overloaded",
                    status=429,
                )
            session_id = self._normalize_session_id(session_id)
            (
                agent,
                history,
                frozen_instructions,
                record,
                recovered_device_calls,
            ) = await self._prepare_agent(session_id)
            if not record:
                raise RealtimeSessionError(
                    "Realtime session was not found",
                    code="session_not_found",
                    status=404,
                )
            base_tools = self._base_tools(agent, record)
            device_binding = self._device_binding(device)
            self._require_device_persistence(agent, device_binding)
            frozen_tools = self._combine_tools(base_tools, device_binding)
            logical_config = self._logical_config(record)
            agent.model = logical_config.model
            session_config = logical_config.openai_session(
                instructions=frozen_instructions,
                tools=frozen_tools,
            )
            call = await self._call_client.create_call(offer_sdp, session_config)
            session_ref: dict[str, RealtimeVoiceSession] = {}

            async def handle_event(event):
                session = session_ref.get("session")
                if session is not None:
                    await session.handle_provider_event(event)

            sideband = self._sideband_factory(call.call_id, handle_event)
            session = RealtimeVoiceSession(
                session_id=session_id,
                agent=agent,
                config=logical_config,
                frozen_instructions=frozen_instructions,
                frozen_tools=frozen_tools,
                frozen_base_tools=base_tools,
                device_binding=device_binding,
                conversation_history=history,
                sideband=sideband,
                call_id=call.call_id,
                profile_name=self._profile_name,
            )
            session_ref["session"] = session
            try:
                await session.start(seed_history=bool(history))
                if recovered_device_calls:
                    await session.resume_recovered_device_turn(
                        recovered_device_calls,
                        outputs_seeded=bool(history),
                    )
            except Exception:
                await session.close(reason="renewal_failed", end_session=False)
                raise
            self._sessions[session_id] = session
            self._ensure_maintenance_task()
            return CreatedRealtimeSession(
                session=session,
                answer_sdp=call.answer_sdp,
                call_id=call.call_id,
            )

    async def close_session(
        self,
        session_id: str,
        *,
        reason: str = "client_closed",
    ) -> bool:
        async with self._lock:
            return await self._close_session_locked(session_id, reason=reason)

    async def _close_session_locked(
        self,
        session_id: str,
        *,
        reason: str,
    ) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.close(reason=reason, end_session=True)
            return True

        agent = await asyncio.to_thread(self._agent_factory, session_id)
        db = getattr(agent, "_session_db", None)
        if db is None:
            return False
        record = await asyncio.to_thread(db.get_realtime_session_state, session_id)
        if record is None:
            return False
        await asyncio.to_thread(db.delete_realtime_session_state, session_id)
        await asyncio.to_thread(db.end_session, session_id, reason)
        return True

    async def suspend_session(
        self,
        session_id: str,
        *,
        reason: str = "client_suspended",
    ) -> bool:
        """Close the provider call while retaining logical Hermes history."""

        async with self._lock:
            return await self._suspend_session_locked(session_id, reason=reason)

    async def _suspend_session_locked(
        self,
        session_id: str,
        *,
        reason: str,
    ) -> bool:
        session = self.get(session_id)
        if session is None:
            agent = await asyncio.to_thread(self._agent_factory, session_id)
            db = getattr(agent, "_session_db", None)
            if db is None:
                return False
            record = await asyncio.to_thread(
                db.get_realtime_session_state,
                session_id,
            )
            if record is None:
                return False
            if record.get("state") != "suspended":
                await asyncio.to_thread(
                    db.save_realtime_session_state,
                    session_id,
                    provider_call_id=str(record.get("provider_call_id") or ""),
                    provider_call_started_at=float(
                        record.get("provider_call_started_at") or 0
                    ),
                    state="suspended",
                    model=str(record.get("model") or ""),
                    voice=str(record.get("voice") or ""),
                    frozen_instructions=str(record.get("frozen_instructions") or ""),
                    frozen_tools=_normalize_frozen_tools(
                        record.get("frozen_tools"),
                    ),
                    frozen_base_tools=self._base_tools(agent, record),
                    device_context=record.get("device_context"),
                )
            return True

        if not session.can_suspend:
            raise RealtimeSessionError(
                "Realtime session is busy; wait for the current turn to settle",
                code="session_busy",
                status=409,
            )
        try:
            await session.suspend(reason=reason)
        except RealtimeProtocolError as exc:
            raise RealtimeSessionError(
                str(exc),
                code=exc.code,
                status=409,
            ) from exc
        if self._sessions.get(session_id) is session:
            self._sessions.pop(session_id, None)
        return True

    async def close_all(self, *, reason: str = "gateway_shutdown") -> None:
        maintenance, self._maintenance_task = self._maintenance_task, None
        if maintenance is not None:
            maintenance.cancel()
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        if sessions:
            await asyncio.gather(
                *(
                    session.close(reason=reason, end_session=False)
                    for session in sessions
                ),
                return_exceptions=True,
            )
        if maintenance is not None:
            await asyncio.gather(maintenance, return_exceptions=True)

    async def prune(self) -> None:
        async with self._lock:
            await self._prune_locked()

    async def _prune_locked(self) -> None:
        now = time.time()
        stale = [
            session_id
            for session_id, session in self._sessions.items()
            if session.closed
            or (
                session.idle_timeout_eligible
                and now - session.last_activity_at > self.config.idle_timeout_seconds
            )
        ]
        for session_id in stale:
            session = self._sessions.pop(session_id)
            await session.close(reason="idle_timeout", end_session=True)

    def _ensure_maintenance_task(self) -> None:
        if not self._sessions:
            return
        if self._maintenance_task is not None and not self._maintenance_task.done():
            return
        self._maintenance_task = asyncio.create_task(
            self._maintenance_loop(),
            name=f"realtime-maintenance-{self._profile_name or 'default'}",
        )

    async def _maintenance_loop(self) -> None:
        interval = max(5.0, min(30.0, self.config.idle_timeout_seconds / 2))
        try:
            while True:
                await asyncio.sleep(interval)
                await self.prune()
                if not self._sessions:
                    return
        finally:
            if self._maintenance_task is asyncio.current_task():
                self._maintenance_task = None

    def _check_creation_rate(self) -> None:
        now = time.monotonic()
        while self._creation_times and now - self._creation_times[0] >= 60:
            self._creation_times.popleft()
        if len(self._creation_times) >= self.config.max_creations_per_minute:
            raise RealtimeSessionError(
                "Too many realtime sessions were created recently",
                code="realtime_creation_rate_limited",
                status=429,
            )

    @staticmethod
    def _normalize_session_id(requested: Optional[str]) -> str:
        if requested is None or not str(requested).strip():
            return str(uuid.uuid4())
        session_id = str(requested).strip()
        if not _SESSION_ID_RE.fullmatch(session_id):
            raise RealtimeSessionError(
                "Requested session_id is invalid",
                code="invalid_session_id",
                status=400,
            )
        return session_id

    @staticmethod
    def _load_history(agent: Any, session_id: str) -> list[dict[str, Any]]:
        db = getattr(agent, "_session_db", None)
        if db is None:
            return []
        session_row = db.get_session(session_id)
        if session_row is None:
            return []
        if session_row.get("ended_at") is not None:
            raise RealtimeSessionError(
                "The requested Hermes session has already ended",
                code="session_ended",
                status=409,
            )
        return db.get_messages(session_id)
