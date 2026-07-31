"""OpenAI WebRTC call creation and server-side Realtime sideband transport."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, Awaitable, Callable, Mapping, Optional
from urllib.parse import quote, unquote_plus, urlsplit, urlunsplit

from gateway.realtime.protocol import (
    RealtimeCall,
    RealtimeProtocolError,
    extract_call_id,
)


logger = logging.getLogger(__name__)


def _sideband_call_url(websocket_url: str, call_id: str) -> str:
    """Add the call ID without discarding proxy-specific query parameters."""

    parsed = urlsplit(websocket_url)
    query_segments = []
    for segment in parsed.query.split("&") if parsed.query else []:
        encoded_name = segment.partition("=")[0]
        try:
            name = unquote_plus(encoded_name, errors="strict")
        except UnicodeDecodeError:
            name = ""
        if name != "call_id":
            query_segments.append(segment)
    query = "&".join(query_segments)
    if query and not query.endswith("&"):
        query += "&"
    query += f"call_id={quote(call_id, safe='')}"
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment)
    )


class OpenAIRealtimeError(RuntimeError):
    """An OpenAI Realtime HTTP or WebSocket operation failed."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "openai_realtime_error",
        status: Optional[int] = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.code = code
        self.status = status
        self.retryable = retryable


def _safe_error_text(value: Any, *, limit: int = 500) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return text[:limit] or "OpenAI Realtime request failed"


class OpenAIRealtimeCallClient:
    """Create a WebRTC call while keeping the standard API key server-side."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        call_url: Optional[str] = None,
        request_timeout_seconds: float = 20.0,
        safety_identifier: str = "",
        http_session: Any = None,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._call_url = call_url or f"{base_url.rstrip('/')}/realtime/calls"
        self._timeout = request_timeout_seconds
        self._safety_identifier = safety_identifier
        self._http_session = http_session

    async def create_call(
        self,
        offer_sdp: str,
        session_config: Mapping[str, Any],
    ) -> RealtimeCall:
        if not offer_sdp.strip():
            raise RealtimeProtocolError("SDP offer is empty", code="empty_sdp_offer")

        import aiohttp

        form = aiohttp.FormData()
        form.add_field("sdp", offer_sdp, content_type="application/sdp")
        form.add_field(
            "session",
            json.dumps(session_config, ensure_ascii=False, separators=(",", ":")),
            content_type="application/json",
        )
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._safety_identifier:
            headers["OpenAI-Safety-Identifier"] = self._safety_identifier
        timeout = aiohttp.ClientTimeout(total=self._timeout)

        owns_session = self._http_session is None
        client = self._http_session or aiohttp.ClientSession(timeout=timeout)
        try:
            async with client.post(
                self._call_url,
                data=form,
                headers=headers,
                timeout=timeout,
            ) as response:
                answer_sdp = await response.text()
                if response.status >= 400:
                    raise OpenAIRealtimeError(
                        _safe_error_text(answer_sdp),
                        code="call_creation_failed",
                        status=response.status,
                        retryable=response.status in {408, 409, 429}
                        or response.status >= 500,
                    )
                try:
                    call_id = extract_call_id(response.headers)
                except RealtimeProtocolError as exc:
                    raise OpenAIRealtimeError(
                        str(exc), code=exc.code, status=response.status
                    ) from exc
                if not answer_sdp.strip():
                    raise OpenAIRealtimeError(
                        "OpenAI returned an empty SDP answer",
                        code="empty_sdp_answer",
                        status=response.status,
                    )
                return RealtimeCall(answer_sdp=answer_sdp, call_id=call_id)
        finally:
            if owns_session:
                await client.close()


class OpenAIRealtimeSideband:
    """One serialized server-side WebSocket attached to an OpenAI call."""

    def __init__(
        self,
        api_key: str,
        call_id: str,
        event_handler: Callable[[Mapping[str, Any]], Awaitable[None] | None],
        *,
        websocket_url: str = "wss://api.openai.com/v1/realtime",
        connect_timeout_seconds: float = 10.0,
        connect_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    ):
        if not api_key or not call_id:
            raise ValueError("api_key and call_id are required")
        self.call_id = call_id
        self._api_key = api_key
        self._event_handler = event_handler
        self._websocket_url = websocket_url
        self._connect_timeout = connect_timeout_seconds
        self._connect_fn = connect_fn
        self._websocket: Any = None
        self._send_lock = asyncio.Lock()
        self._closed = False

    @property
    def connected(self) -> bool:
        return self._websocket is not None and not self._closed

    async def connect(self) -> None:
        if self.connected:
            return
        if self._closed:
            raise OpenAIRealtimeError(
                "Sideband is closed", code="sideband_closed"
            )
        if self._connect_fn is None:
            from websockets.asyncio.client import connect

            connect_fn = connect
        else:
            connect_fn = self._connect_fn

        url = _sideband_call_url(self._websocket_url, self.call_id)
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            self._websocket = await asyncio.wait_for(
                connect_fn(
                    url,
                    additional_headers=headers,
                    max_size=4 * 1024 * 1024,
                ),
                timeout=self._connect_timeout,
            )
        except Exception as exc:
            raise OpenAIRealtimeError(
                f"Could not attach OpenAI sideband: {_safe_error_text(exc)}",
                code="sideband_connect_failed",
                retryable=True,
            ) from exc

    async def send(self, event: Mapping[str, Any]) -> None:
        if not self.connected:
            raise OpenAIRealtimeError(
                "OpenAI sideband is not connected",
                code="sideband_not_connected",
                retryable=True,
            )
        encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        async with self._send_lock:
            try:
                await self._websocket.send(encoded)
            except Exception as exc:
                raise OpenAIRealtimeError(
                    f"OpenAI sideband send failed: {_safe_error_text(exc)}",
                    code="sideband_send_failed",
                    retryable=True,
                ) from exc

    async def receive_loop(self) -> None:
        if not self.connected:
            await self.connect()
        try:
            async for raw in self._websocket:
                try:
                    event = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Ignoring malformed OpenAI Realtime event for call %s",
                        self.call_id,
                    )
                    continue
                if not isinstance(event, Mapping):
                    continue
                result = self._event_handler(event)
                if inspect.isawaitable(result):
                    await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                raise OpenAIRealtimeError(
                    f"OpenAI sideband receive failed: {_safe_error_text(exc)}",
                    code="sideband_receive_failed",
                    retryable=True,
                ) from exc

    async def reconnect(self) -> None:
        """Replace a failed socket while retaining the same provider call."""

        if self._closed:
            raise OpenAIRealtimeError(
                "Sideband is closed", code="sideband_closed"
            )
        websocket, self._websocket = self._websocket, None
        if websocket is not None:
            try:
                await websocket.close()
            except Exception:
                logger.debug("Failed to close stale OpenAI sideband", exc_info=True)
        await self.connect()

    async def close(self) -> None:
        self._closed = True
        websocket, self._websocket = self._websocket, None
        if websocket is not None:
            try:
                await websocket.close()
            except Exception:
                logger.debug("Failed to close OpenAI sideband", exc_info=True)
