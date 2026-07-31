"""Behavioral tests for the localhost Hermes Realtime voice client."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scripts import realtime_voice_pc_client as client

OFFER = b"v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"


class _FakeHermesServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        self.requests = []
        self.create_status = 200
        self.create_started = threading.Event()
        self.release_create = threading.Event()
        self.release_create.set()
        super().__init__(("127.0.0.1", 0), _FakeHermesHandler)


class _FakeHermesHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.requests.append(
            (self.command, self.path, dict(self.headers), b"")
        )
        if self.path != "/v1/capabilities":
            self.send_error(404)
            return
        self._send_json(
            200,
            {"features": {"realtime_voice": True}},
        )

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.requests.append(
            (self.command, self.path, dict(self.headers), body)
        )
        self.server.create_started.set()
        self.server.release_create.wait(timeout=3)
        if self.server.create_status != 200:
            self._send_json(
                self.server.create_status,
                {"error": {"message": "upstream rejected the offer"}},
            )
            return
        self._send_json(
            200,
            {
                "version": 1,
                "session_id": "rt_test-1",
                "call_id": "rtc_test",
                "sdp": "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
                "model": "gpt-realtime-2.1",
                "voice": "marin",
            },
        )

    def do_DELETE(self):
        self.server.requests.append(
            (self.command, self.path, dict(self.headers), b"")
        )
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        return


@contextmanager
def _running(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        if isinstance(server, client.VoiceClientServer):
            server.begin_shutdown()
            attempted = server.close_remote_sessions()
        else:
            attempted = set()
        server.server_close()
        if isinstance(server, client.VoiceClientServer):
            server.close_remote_sessions(exclude=attempted)
        thread.join(timeout=2)


def _request(url, *, method="GET", body=None, headers=None):
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers or {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, dict(exc.headers), exc.read()


@pytest.mark.parametrize(
    "url",
    [
        "",
        "ftp://example.com",
        "http://example.com",
        "https://user:pass@example.com",
        "https://example.com?token=value",
        "https://example.com/#fragment",
    ],
)
def test_validate_hermes_url_rejects_unsafe_values(url):
    with pytest.raises(ValueError):
        client.validate_hermes_url(url)


def test_validate_hermes_url_allows_https_and_loopback_http():
    assert (
        client.validate_hermes_url("https://example.com/profile/")
        == "https://example.com/profile"
    )
    assert (
        client.validate_hermes_url("http://127.0.0.1:8642/")
        == "http://127.0.0.1:8642"
    )


def test_local_server_proxies_capabilities_session_and_cleanup():
    api_key = "secret-api-server-key"
    with _running(_FakeHermesServer()) as upstream:
        proxy = client.HermesProxy(
            f"http://127.0.0.1:{upstream.server_port}",
            api_key,
        )
        local = client.VoiceClientServer(
            ("127.0.0.1", 0),
            proxy,
            client.load_html(),
        )
        with _running(local):
            base = f"http://127.0.0.1:{local.server_port}"

            status, headers, html = _request(f"{base}/")
            assert status == 200
            assert headers["Content-Security-Policy"]
            assert api_key.encode() not in html

            status, _, body = _request(f"{base}/api/capabilities")
            assert status == 200
            assert json.loads(body)["features"]["realtime_voice"] is True

            status, _, body = _request(
                f"{base}/api/session",
                method="POST",
                body=OFFER,
                headers={
                    "Content-Type": "application/sdp",
                    "X-Hermes-Client-Id": "browser-client-0001",
                },
            )
            assert status == 200
            assert json.loads(body)["session_id"] == "rt_test-1"

            status, _, body = _request(
                f"{base}/api/session/rt_test-1",
                method="DELETE",
            )
            assert status == 204
            assert body == b""

    assert [request[0:2] for request in upstream.requests] == [
        ("GET", "/v1/capabilities"),
        ("POST", "/v1/realtime/sessions"),
        ("DELETE", "/v1/realtime/sessions/rt_test-1"),
    ]
    for _method, _path, headers, _body in upstream.requests:
        assert headers["Authorization"] == f"Bearer {api_key}"
    assert upstream.requests[1][2]["Content-Type"] == "application/sdp"
    assert upstream.requests[1][3] == OFFER


def test_local_server_preserves_safe_upstream_error():
    with _running(_FakeHermesServer()) as upstream:
        upstream.create_status = 401
        proxy = client.HermesProxy(
            f"http://127.0.0.1:{upstream.server_port}",
            "test-key",
        )
        local = client.VoiceClientServer(("127.0.0.1", 0), proxy, b"test")
        with _running(local):
            status, headers, body = _request(
                f"http://127.0.0.1:{local.server_port}/api/session",
                method="POST",
                body=OFFER,
                headers={
                    "Content-Type": "application/sdp",
                    "X-Hermes-Client-Id": "browser-client-0002",
                },
            )

    assert status == 401
    assert headers["Content-Type"] == "application/json"
    assert json.loads(body)["error"]["message"] == "upstream rejected the offer"


def test_loopback_upstream_bypasses_ambient_http_proxy(monkeypatch):
    with _running(_FakeHermesServer()) as upstream, _running(
        _FakeHermesServer()
    ) as ambient_proxy:
        monkeypatch.setenv(
            "HTTP_PROXY",
            f"http://127.0.0.1:{ambient_proxy.server_port}",
        )
        monkeypatch.setenv(
            "http_proxy",
            f"http://127.0.0.1:{ambient_proxy.server_port}",
        )
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")
        proxy = client.HermesProxy(
            f"http://127.0.0.1:{upstream.server_port}",
            "must-not-reach-proxy",
        )

        response = proxy.capabilities()

    assert response.status == 200
    assert len(upstream.requests) == 1
    assert upstream.requests[0][2]["Authorization"] == (
        "Bearer must-not-reach-proxy"
    )
    assert ambient_proxy.requests == []


def test_client_close_during_creation_reclaims_remote_session():
    with _running(_FakeHermesServer()) as upstream:
        upstream.release_create.clear()
        proxy = client.HermesProxy(
            f"http://127.0.0.1:{upstream.server_port}",
            "test-key",
        )
        local = client.VoiceClientServer(("127.0.0.1", 0), proxy, b"test")
        with _running(local):
            base = f"http://127.0.0.1:{local.server_port}"
            result = []

            def create():
                result.append(
                    _request(
                        f"{base}/api/session",
                        method="POST",
                        body=OFFER,
                        headers={
                            "Content-Type": "application/sdp",
                            "X-Hermes-Client-Id": "browser-client-0003",
                        },
                    )
                )

            create_thread = threading.Thread(target=create)
            create_thread.start()
            assert upstream.create_started.wait(timeout=2)

            status, _, _ = _request(
                f"{base}/api/client/browser-client-0003",
                method="DELETE",
            )
            assert status == 204
            upstream.release_create.set()
            create_thread.join(timeout=3)

            assert result[0][0] == 409
            assert [request[0:2] for request in upstream.requests] == [
                ("POST", "/v1/realtime/sessions"),
                ("DELETE", "/v1/realtime/sessions/rt_test-1"),
            ]


def test_client_cancel_before_creation_prevents_remote_session():
    with _running(_FakeHermesServer()) as upstream:
        proxy = client.HermesProxy(
            f"http://127.0.0.1:{upstream.server_port}",
            "test-key",
        )
        local = client.VoiceClientServer(("127.0.0.1", 0), proxy, b"test")
        with _running(local):
            base = f"http://127.0.0.1:{local.server_port}"
            status, _, _ = _request(
                f"{base}/api/client/browser-client-0005",
                method="DELETE",
            )
            assert status == 204

            status, _, _ = _request(
                f"{base}/api/session",
                method="POST",
                body=OFFER,
                headers={
                    "Content-Type": "application/sdp",
                    "X-Hermes-Client-Id": "browser-client-0005",
                },
            )

            assert status == 409
            assert upstream.requests == []


def test_local_server_shutdown_closes_tracked_remote_session():
    with _running(_FakeHermesServer()) as upstream:
        proxy = client.HermesProxy(
            f"http://127.0.0.1:{upstream.server_port}",
            "test-key",
        )
        local = client.VoiceClientServer(("127.0.0.1", 0), proxy, b"test")
        with _running(local):
            status, _, _ = _request(
                f"http://127.0.0.1:{local.server_port}/api/session",
                method="POST",
                body=OFFER,
                headers={
                    "Content-Type": "application/sdp",
                    "X-Hermes-Client-Id": "browser-client-0006",
                },
            )
            assert status == 200
            assert [request[0:2] for request in upstream.requests] == [
                ("POST", "/v1/realtime/sessions"),
            ]

        assert [request[0:2] for request in upstream.requests] == [
            ("POST", "/v1/realtime/sessions"),
            ("DELETE", "/v1/realtime/sessions/rt_test-1"),
        ]


def test_local_server_rejects_cross_site_and_invalid_sdp_without_proxying():
    with _running(_FakeHermesServer()) as upstream:
        proxy = client.HermesProxy(
            f"http://127.0.0.1:{upstream.server_port}",
            "test-key",
        )
        local = client.VoiceClientServer(("127.0.0.1", 0), proxy, b"test")
        with _running(local):
            base = f"http://127.0.0.1:{local.server_port}/api/session"
            cross_site_status, _, _ = _request(
                base,
                method="POST",
                body=OFFER,
                headers={
                    "Content-Type": "application/sdp",
                    "Sec-Fetch-Site": "cross-site",
                    "X-Hermes-Client-Id": "browser-client-0004",
                },
            )
            invalid_status, _, _ = _request(
                base,
                method="POST",
                body=b"v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n",
                headers={
                    "Content-Type": "application/sdp",
                    "X-Hermes-Client-Id": "browser-client-0004",
                },
            )

    assert cross_site_status == 403
    assert invalid_status == 400
    assert upstream.requests == []
