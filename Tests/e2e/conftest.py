"""End-to-end checks against a running Feynt.

These drive the real app over HTTP: the protocols, the official client SDKs, cancellation,
and a model that actually loads and answers. They need a Feynt listening and a model on
disk, so they are not part of `swift test`.

Point them at a copy, not the Feynt you use: they load and unload models. By default they
talk to 127.0.0.1:19235.

    FEYNT_URL=http://127.0.0.1:19235 FEYNT_MODEL=uncensored-moe \
        uv run --with pytest --with mcp --with a2a-sdk pytest Tests/e2e
"""

import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import pytest

BASE = os.environ.get("FEYNT_URL", "http://127.0.0.1:19235").rstrip("/")
MODEL = os.environ.get("FEYNT_MODEL", "uncensored-moe")
VERSION = "2026-07-28"
META = {
    "io.modelcontextprotocol/protocolVersion": VERSION,
    "io.modelcontextprotocol/clientInfo": {"name": "feynt-e2e", "version": "1"},
    "io.modelcontextprotocol/clientCapabilities": {},
}
ESSAY = "Write a 1500-word essay about the history of rivers. Do not stop early."


def pytest_sessionstart(session):
    try:
        urllib.request.urlopen(BASE + "/.well-known/agent-card.json", timeout=5)
    except Exception as error:  # noqa: BLE001 - any failure means there is nothing to test
        pytest.exit(f"no Feynt at {BASE}: {error}", returncode=2)


def post(path, body, headers, timeout=900):
    request = urllib.request.Request(BASE + path, data=body, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
        status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    return status, (json.loads(raw) if raw else None)


def mcp(method, params=None, rid=1, modern=True, headers=None):
    params = dict(params or {})
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if modern:
        params["_meta"] = META
        h.update({"MCP-Protocol-Version": VERSION, "Mcp-Method": method})
        if method == "tools/call":
            h["Mcp-Name"] = params["name"]
    h.update(headers or {})
    body = {"jsonrpc": "2.0", "method": method, "params": params}
    if rid is not None:
        body["id"] = rid
    return post("/mcp", json.dumps(body).encode(), h)


def tool(name, arguments, modern=True):
    status, data = mcp("tools/call", {"name": name, "arguments": arguments}, modern=modern)
    assert status == 200, data
    return data["result"]


def a2a(method, params, rid=1):
    body = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}).encode()
    status, data = post("/a2a", body, {"Content-Type": "application/json", "A2A-Version": "1.0"})
    assert status == 200, data
    return data


def message(text, context=None, **fields):
    m = {"messageId": str(uuid.uuid4()), "role": "ROLE_USER", "parts": [{"text": text}]}
    if context:
        m["contextId"] = context
    m.update(fields)
    return m


def timed(fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    return time.time() - start, result


def short_answer():
    """A short MCP generation; how long it takes says whether the GPU was free."""
    return timed(tool, "generate", {"prompt": "Reply with exactly: ok", "max_tokens": 8, "model": MODEL})[0]


def send_then_hang_up(path, headers, body, after=1.5):
    """Starts a request and closes the socket mid-answer, the way an interrupted agent does."""
    url = urllib.parse.urlparse(BASE)
    sock = socket.create_connection((url.hostname, url.port))
    head = f"POST {path} HTTP/1.1\r\nHost: {url.hostname}\r\nContent-Length: {len(body)}\r\n"
    head += "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n"
    sock.sendall(head.encode() + body)
    time.sleep(after)
    sock.close()


@pytest.fixture(scope="session")
def warm():
    """Loads the model once, so no test measures a load as if it were a generation."""
    short_answer()
    return short_answer()
