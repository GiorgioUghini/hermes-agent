"""Admission, lookup, renewal, and cleanup for realtime voice sessions."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, replace
import logging
import re
import time
from typing import Any, Callable, Optional
import uuid

from gateway.realtime.openai_sideband import (
    OpenAIRealtimeCallClient,
    OpenAIRealtimeSideband,
)
from gateway.realtime.protocol import RealtimeCall, RealtimeProtocolError, RealtimeVoiceConfig
from gateway.realtime.session import RealtimeVoiceSession, prepare_realtime_agent


logger = logging.getLogger(__name__)

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")


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

    def _logical_config(
        self, record: Optional[dict[str, Any]]
    ) -> RealtimeVoiceConfig:
        """Keep model and voice stable across restart and provider-call rotation."""

        if not record:
            return self.config
        return replace(
            self.config,
            model=str(record.get("model") or self.config.model),
            voice=str(record.get("voice") or self.config.voice),
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
    ) -> tuple[Any, list[dict[str, Any]], str, Optional[dict[str, Any]]]:
        agent = await asyncio.to_thread(self._agent_factory, session_id)
        history = await asyncio.to_thread(self._load_history, agent, session_id)
        db = getattr(agent, "_session_db", None)
        record = (
            await asyncio.to_thread(db.get_realtime_session_state, session_id)
            if db is not None
            else None
        )
        frozen_instructions = await asyncio.to_thread(
            prepare_realtime_agent,
            agent,
            history,
            frozen_instructions=(
                str(record.get("frozen_instructions") or "") if record else None
            ),
        )
        return agent, history, frozen_instructions, record

    async def _recover_active_session(
        self, session_id: str
    ) -> RealtimeVoiceSession:
        agent, history, frozen_instructions, record = await self._prepare_agent(
            session_id
        )
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
        session = RealtimeVoiceSession(
            session_id=session_id,
            agent=agent,
            config=logical_config,
            frozen_instructions=frozen_instructions,
            frozen_tools=record.get("frozen_tools") or agent.tools or [],
            conversation_history=history,
            sideband=sideband,
            call_id=call_id,
            call_started_at=call_started_at,
            profile_name=self._profile_name,
        )
        session_ref["session"] = session
        try:
            await session.start(preserve_call_started_at=True)
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

            agent, history, frozen_instructions, prior_state = (
                await self._prepare_agent(session_id)
            )
            frozen_tools = (
                prior_state.get("frozen_tools")
                if prior_state and prior_state.get("frozen_tools")
                else agent.tools or []
            )
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
                conversation_history=history,
                sideband=sideband,
                call_id=call.call_id,
                profile_name=self._profile_name,
            )
            session_ref["session"] = session
            try:
                await session.start(seed_history=bool(history))
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
    ) -> CreatedRealtimeSession:
        session = self.get(session_id)
        if session is None:
            return await self._renew_persisted_session(session_id, offer_sdp)
        if not session.can_renew:
            raise RealtimeSessionError(
                "Realtime session is busy; wait for the current turn to settle",
                code="session_busy",
                status=409,
            )
        call = await self._call_client.create_call(
            offer_sdp, session.session_config()
        )
        sideband = self._sideband_factory(
            call.call_id, session.handle_provider_event
        )
        try:
            await session.replace_sideband(sideband=sideband, call_id=call.call_id)
        except RealtimeProtocolError as exc:
            await sideband.close()
            raise RealtimeSessionError(
                str(exc), code=exc.code, status=409
            ) from exc
        return CreatedRealtimeSession(
            session=session,
            answer_sdp=call.answer_sdp,
            call_id=call.call_id,
        )

    async def _renew_persisted_session(
        self,
        session_id: str,
        offer_sdp: str,
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
                call = await self._call_client.create_call(
                    offer_sdp, existing.session_config()
                )
                sideband = self._sideband_factory(
                    call.call_id, existing.handle_provider_event
                )
                await existing.replace_sideband(
                    sideband=sideband, call_id=call.call_id
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
            agent, history, frozen_instructions, record = (
                await self._prepare_agent(session_id)
            )
            if not record:
                raise RealtimeSessionError(
                    "Realtime session was not found",
                    code="session_not_found",
                    status=404,
                )
            frozen_tools = record.get("frozen_tools") or agent.tools or []
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
                conversation_history=history,
                sideband=sideband,
                call_id=call.call_id,
                profile_name=self._profile_name,
            )
            session_ref["session"] = session
            try:
                await session.start(seed_history=bool(history))
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
