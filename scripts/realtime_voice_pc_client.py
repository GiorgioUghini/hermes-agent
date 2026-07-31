#!/usr/bin/env python3
"""Run a localhost-only browser client for the Hermes Realtime voice API."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping

DEFAULT_PORT = 8787
DELETE_TIMEOUT_SECONDS = 5.0
MAX_SDP_BYTES = 256 * 1024
MAX_UPSTREAM_BYTES = 1024 * 1024
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")
CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
HTML_PATH = Path(__file__).with_suffix(".html")


class ProxyError(RuntimeError):
    """Raised when the configured Hermes server cannot be proxied safely."""


@dataclass(frozen=True)
class UpstreamResponse:
    status: int
    body: bytes
    content_type: str


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent bearer credentials from following a redirect to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _is_loopback_hostname(hostname: str | None) -> bool:
    if not hostname:
        return False
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_hermes_url(raw_url: str) -> str:
    """Validate and normalize a Hermes API-server base URL."""

    value = raw_url.strip()
    if not value:
        raise ValueError("HERMES_REALTIME_URL is required.")
    try:
        parsed = urllib.parse.urlsplit(value)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("HERMES_REALTIME_URL is not a valid URL.") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("HERMES_REALTIME_URL must be an http(s) URL.")
    if parsed.username or parsed.password:
        raise ValueError("HERMES_REALTIME_URL must not contain credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("HERMES_REALTIME_URL must not contain a query or fragment.")
    if parsed.scheme == "http" and not _is_loopback_hostname(parsed.hostname):
        raise ValueError(
            "HERMES_REALTIME_URL must use HTTPS unless Hermes is on localhost."
        )
    path = parsed.path.rstrip("/")
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, path, "", "")
    )


def _safe_content_type(raw_value: str | None) -> str:
    if raw_value and raw_value.lower().startswith("application/json"):
        return "application/json"
    return "text/plain; charset=utf-8"


def _read_limited(stream, limit: int) -> bytes:
    body = stream.read(limit + 1)
    if len(body) > limit:
        raise ProxyError("Hermes returned an unexpectedly large response.")
    return body


def _cleanup_succeeded(status: int) -> bool:
    return 200 <= status < 300 or status == 404


class HermesProxy:
    """Small authenticated HTTP client for the remote Hermes API server."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 90.0,
    ) -> None:
        self.base_url = validate_hermes_url(base_url)
        self._api_key = api_key.strip()
        if not self._api_key:
            raise ValueError("HERMES_API_KEY is required.")
        if timeout_seconds <= 0:
            raise ValueError("The upstream timeout must be positive.")
        self._timeout_seconds = timeout_seconds
        handlers = [_NoRedirectHandler()]
        if _is_loopback_hostname(
            urllib.parse.urlsplit(self.base_url).hostname
        ):
            handlers.insert(0, urllib.request.ProxyHandler({}))
        self._opener = urllib.request.build_opener(*handlers)

    def capabilities(self) -> UpstreamResponse:
        return self._request("GET", "/v1/capabilities")

    def create_session(self, offer_sdp: bytes) -> UpstreamResponse:
        return self._request(
            "POST",
            "/v1/realtime/sessions",
            body=offer_sdp,
            content_type="application/sdp",
        )

    def delete_session(self, session_id: str) -> UpstreamResponse:
        if not SESSION_ID_RE.fullmatch(session_id):
            raise ValueError("The realtime session ID is invalid.")
        encoded_id = urllib.parse.quote(session_id, safe="")
        return self._request(
            "DELETE",
            f"/v1/realtime/sessions/{encoded_id}",
            timeout_seconds=DELETE_TIMEOUT_SECONDS,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout_seconds: float | None = None,
    ) -> UpstreamResponse:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "Hermes-Realtime-PC-Client/1",
        }
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(
                request,
                timeout=timeout_seconds or self._timeout_seconds,
            ) as response:
                return UpstreamResponse(
                    status=response.status,
                    body=_read_limited(response, MAX_UPSTREAM_BYTES),
                    content_type=_safe_content_type(
                        response.headers.get("Content-Type")
                    ),
                )
        except urllib.error.HTTPError as exc:
            with exc:
                return UpstreamResponse(
                    status=exc.code,
                    body=_read_limited(exc, MAX_UPSTREAM_BYTES),
                    content_type=_safe_content_type(
                        exc.headers.get("Content-Type")
                    ),
                )
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProxyError(
                "Could not reach the configured Hermes server."
            ) from exc


def load_html() -> bytes:
    try:
        return HTML_PATH.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Could not read {HTML_PATH.name}.") from exc


class VoiceClientServer(ThreadingHTTPServer):
    daemon_threads = False
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        proxy: HermesProxy,
        html: bytes,
    ) -> None:
        self.proxy = proxy
        self.html = html
        self._session_lock = threading.Lock()
        self._client_sessions: dict[str, str | None] = {}
        self._cancelled_clients: set[str] = set()
        self._active_sessions: set[str] = set()
        self._closing = False
        super().__init__(server_address, VoiceClientHandler)

    def begin_creation(self, client_id: str) -> bool:
        with self._session_lock:
            if (
                self._closing
                or client_id in self._client_sessions
                or client_id in self._cancelled_clients
            ):
                return False
            self._client_sessions[client_id] = None
            return True

    def register_created_session(self, client_id: str, session_id: str) -> bool:
        with self._session_lock:
            self._active_sessions.add(session_id)
            if (
                self._closing
                or client_id in self._cancelled_clients
                or client_id not in self._client_sessions
            ):
                self._client_sessions.pop(client_id, None)
                self._cancelled_clients.discard(client_id)
                return False
            self._client_sessions[client_id] = session_id
            return True

    def finish_failed_creation(self, client_id: str) -> None:
        with self._session_lock:
            self._client_sessions.pop(client_id, None)
            self._cancelled_clients.discard(client_id)

    def cancel_client(self, client_id: str) -> str | None:
        with self._session_lock:
            self._cancelled_clients.add(client_id)
            return self._client_sessions.pop(client_id, None)

    def forget_session(self, session_id: str) -> None:
        with self._session_lock:
            self._active_sessions.discard(session_id)
            for client_id, active_id in list(self._client_sessions.items()):
                if active_id == session_id:
                    self._client_sessions.pop(client_id, None)

    def begin_shutdown(self) -> None:
        with self._session_lock:
            self._closing = True
            self._cancelled_clients.update(self._client_sessions)

    def close_remote_sessions(
        self,
        *,
        exclude: set[str] | None = None,
    ) -> set[str]:
        with self._session_lock:
            session_ids = tuple(self._active_sessions.difference(exclude or ()))
        attempted = set(session_ids)
        for session_id in session_ids:
            try:
                response = self.proxy.delete_session(session_id)
            except (ProxyError, ValueError) as exc:
                print(
                    f"Warning: could not close Hermes session {session_id}: {exc}",
                    file=sys.stderr,
                )
                continue
            if _cleanup_succeeded(response.status):
                self.forget_session(session_id)
            else:
                print(
                    "Warning: Hermes rejected cleanup for "
                    f"{session_id} with HTTP {response.status}.",
                    file=sys.stderr,
                )
        return attempted


class VoiceClientHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "HermesRealtimePC"
    sys_version = ""

    @property
    def voice_server(self) -> VoiceClientServer:
        return self.server  # type: ignore[return-value]

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10.0)

    def do_GET(self) -> None:
        if not self._is_local_request():
            return
        path = urllib.parse.urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            self._send(
                200,
                self.voice_server.html,
                "text/html; charset=utf-8",
                html=True,
            )
            return
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        if path == "/api/ping":
            if not self._is_same_site_api_request():
                return
            self._send(200, b'{"ok":true}', "application/json")
            return
        if path == "/api/capabilities":
            if not self._is_same_site_api_request():
                return
            self._proxy(self.voice_server.proxy.capabilities)
            return
        self._json_error(404, "not_found", "Route not found.")

    def do_POST(self) -> None:
        if not self._is_local_request() or not self._is_same_site_api_request():
            return
        path = urllib.parse.urlsplit(self.path).path
        if path != "/api/session":
            self._json_error(404, "not_found", "Route not found.")
            return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
        if content_type.casefold() != "application/sdp":
            self._json_error(
                415,
                "unsupported_media_type",
                "Session creation requires application/sdp.",
            )
            return
        offer = self._read_request_body(MAX_SDP_BYTES)
        if offer is None:
            return
        try:
            decoded_offer = offer.decode("utf-8")
        except UnicodeDecodeError:
            decoded_offer = ""
        if "v=0" not in decoded_offer or "m=audio" not in decoded_offer:
            self._json_error(
                400,
                "invalid_sdp",
                "The SDP offer does not contain an audio media line.",
            )
            return
        client_id = self.headers.get("X-Hermes-Client-Id", "")
        if not CLIENT_ID_RE.fullmatch(client_id):
            self._json_error(
                400,
                "invalid_client_id",
                "A valid local client ID is required.",
            )
            return
        if not self.voice_server.begin_creation(client_id):
            self._json_error(
                409,
                "client_busy",
                "This local client already has a session request.",
            )
            return
        try:
            response = self.voice_server.proxy.create_session(offer)
        except (ProxyError, ValueError) as exc:
            self.voice_server.finish_failed_creation(client_id)
            self._json_error(502, "proxy_error", str(exc))
            return
        if not 200 <= response.status < 300:
            self.voice_server.finish_failed_creation(client_id)
            self._send(response.status, response.body, response.content_type)
            return
        session_id = self._session_id_from_response(response)
        if session_id is None:
            self.voice_server.finish_failed_creation(client_id)
            self._json_error(
                502,
                "invalid_upstream_response",
                "Hermes returned an invalid session response.",
            )
            return
        if not self.voice_server.register_created_session(client_id, session_id):
            self._close_registered_session(session_id)
            self._json_error(
                409,
                "client_closed",
                "The browser closed before the session was ready.",
            )
            return
        if not self._send(response.status, response.body, response.content_type):
            self._close_registered_session(session_id)

    def do_DELETE(self) -> None:
        if not self._is_local_request() or not self._is_same_site_api_request():
            return
        path = urllib.parse.urlsplit(self.path).path
        client_prefix = "/api/client/"
        if path.startswith(client_prefix):
            client_id = urllib.parse.unquote(path[len(client_prefix) :])
            if not CLIENT_ID_RE.fullmatch(client_id):
                self._json_error(
                    400,
                    "invalid_client_id",
                    "The local client ID is invalid.",
                )
                return
            session_id = self.voice_server.cancel_client(client_id)
            if session_id is None:
                self._send(204, b"", "application/json")
                return
            self._delete_registered_session(session_id)
            return
        prefix = "/api/session/"
        if not path.startswith(prefix):
            self._json_error(404, "not_found", "Route not found.")
            return
        session_id = urllib.parse.unquote(path[len(prefix) :])
        if not SESSION_ID_RE.fullmatch(session_id):
            self._json_error(
                400,
                "invalid_session_id",
                "The realtime session ID is invalid.",
            )
            return
        self._delete_registered_session(session_id)

    def do_OPTIONS(self) -> None:
        self._json_error(
            405,
            "cross_origin_not_allowed",
            "Cross-origin API access is not allowed.",
        )

    def _read_request_body(self, limit: int) -> bytes | None:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._json_error(
                411,
                "content_length_required",
                "Content-Length is required.",
            )
            return None
        try:
            length = int(raw_length)
        except ValueError:
            length = -1
        if length <= 0:
            self._json_error(400, "empty_body", "A non-empty SDP offer is required.")
            return None
        if length > limit:
            self._json_error(413, "sdp_too_large", "The SDP offer is too large.")
            return None
        try:
            body = self.rfile.read(length)
        except TimeoutError:
            self.close_connection = True
            return None
        if len(body) != length:
            self._json_error(
                400,
                "incomplete_body",
                "The SDP request body ended unexpectedly.",
            )
            return None
        return body

    def _is_local_request(self) -> bool:
        raw_host = self.headers.get("Host", "")
        try:
            hostname = urllib.parse.urlsplit(f"//{raw_host}").hostname
        except ValueError:
            hostname = None
        if not _is_loopback_hostname(hostname):
            self._json_error(
                403,
                "local_only",
                "This test client only accepts localhost requests.",
            )
            return False
        return True

    def _is_same_site_api_request(self) -> bool:
        if self.headers.get("Sec-Fetch-Site", "").casefold() == "cross-site":
            self._json_error(
                403,
                "cross_origin_not_allowed",
                "Cross-origin API access is not allowed.",
            )
            return False
        origin = self.headers.get("Origin")
        if origin:
            try:
                origin_host = urllib.parse.urlsplit(origin).hostname
            except ValueError:
                origin_host = None
            if not _is_loopback_hostname(origin_host):
                self._json_error(
                    403,
                    "cross_origin_not_allowed",
                    "Cross-origin API access is not allowed.",
                )
                return False
        return True

    def _proxy(self, operation) -> None:
        try:
            response = operation()
        except (ProxyError, ValueError) as exc:
            self._json_error(502, "proxy_error", str(exc))
            return
        self._send(response.status, response.body, response.content_type)

    @staticmethod
    def _session_id_from_response(
        response: UpstreamResponse,
    ) -> str | None:
        try:
            payload = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        session_id = payload.get("session_id") if isinstance(payload, dict) else None
        if not isinstance(session_id, str) or not SESSION_ID_RE.fullmatch(session_id):
            return None
        return session_id

    def _delete_registered_session(self, session_id: str) -> None:
        try:
            response = self.voice_server.proxy.delete_session(session_id)
        except (ProxyError, ValueError) as exc:
            self._json_error(502, "proxy_error", str(exc))
            return
        if _cleanup_succeeded(response.status):
            self.voice_server.forget_session(session_id)
        self._send(response.status, response.body, response.content_type)

    def _close_registered_session(self, session_id: str) -> None:
        try:
            response = self.voice_server.proxy.delete_session(session_id)
        except (ProxyError, ValueError):
            return
        if _cleanup_succeeded(response.status):
            self.voice_server.forget_session(session_id)

    def _json_error(
        self,
        status: int,
        code: str,
        message: str,
    ) -> None:
        body = json.dumps(
            {"error": {"code": code, "message": message}},
            separators=(",", ":"),
        ).encode("utf-8")
        self._send(status, body, "application/json")

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        html: bool = False,
    ) -> bool:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            if html:
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                    "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                    "media-src 'self' blob:; object-src 'none'; base-uri 'none'; "
                    "frame-ancestors 'none'",
                )
                self.send_header("Permissions-Policy", "microphone=(self)")
            self.end_headers()
            self.close_connection = True
            if body:
                self.wfile.write(body)
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[pc-client] {fmt % args}\n")


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _load_environment(
    environ: Mapping[str, str] = os.environ,
) -> tuple[str, str]:
    return (
        environ.get("HERMES_REALTIME_URL", ""),
        environ.get("HERMES_API_KEY", ""),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Open a localhost microphone client for Hermes Realtime."
    )
    parser.add_argument("--port", type=_port, default=DEFAULT_PORT)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the default browser automatically.",
    )
    args = parser.parse_args(argv)

    base_url, api_key = _load_environment()
    try:
        proxy = HermesProxy(base_url, api_key)
        os.environ.pop("HERMES_API_KEY", None)
        html = load_html()
        server = VoiceClientServer(("127.0.0.1", args.port), proxy, html)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Hermes Realtime PC client: {url}")
    print("The API key stays in this Python process. Press Ctrl+C to stop.")
    if not args.no_open:
        timer = threading.Timer(0.25, webbrowser.open_new_tab, args=(url,))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.begin_shutdown()
        attempted = server.close_remote_sessions()
        server.server_close()
        server.close_remote_sessions(exclude=attempted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
