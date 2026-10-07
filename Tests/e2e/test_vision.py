"""Images through every protocol, on a model that has a vision tower."""

import base64
import json
import struct
import zlib

from conftest import MODEL, a2a, message, mcp, post, tool


def solid_png(rgb, side=64):
    """A one-colour PNG, written by hand so the suite needs no imaging library."""
    raw = b"".join(b"\x00" + bytes(rgb) * side for _ in range(side))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


RED = base64.b64encode(solid_png((220, 20, 20))).decode()
QUESTION = "What colour fills this image? One word."


def openai(content, model=MODEL):
    body = json.dumps({"model": model, "max_tokens": 10, "temperature": 0,
                       "messages": [{"role": "user", "content": content}]}).encode()
    return post("/v1/chat/completions", body, {"Content-Type": "application/json"})


def test_openai_image_url(warm):
    status, data = openai([{"type": "text", "text": QUESTION},
                           {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{RED}"}}])
    assert status == 200, data
    assert "red" in data["choices"][0]["message"]["content"].lower()


def test_openai_image_without_text(warm):
    status, data = openai([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{RED}"}}])
    assert status == 200, data


def test_openai_refuses_a_remote_image_url():
    status, data = openai([{"type": "text", "text": "hi"},
                           {"type": "image_url", "image_url": {"url": "http://127.0.0.1/x.png"}}])
    assert status == 400 and "does not fetch" in data["error"]["message"]


def test_openai_refuses_something_that_is_not_an_image():
    junk = base64.b64encode(b"not a picture at all").decode()
    status, data = openai([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{junk}"}}])
    assert status == 400


def test_mcp_generate_with_an_image(warm):
    result = tool("generate", {"prompt": QUESTION, "max_tokens": 10, "model": MODEL, "tools": False,
                               "images": [{"data": RED, "mimeType": "image/png"}]})
    assert result["isError"] is False
    assert "red" in result["content"][0]["text"].lower()


def test_a2a_raw_image_part(warm):
    task = a2a("SendMessage", {"message": message(QUESTION, parts=[
        {"text": QUESTION}, {"raw": RED, "mediaType": "image/png"}]),
        "metadata": {"model": MODEL, "tools": False}})["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert "red" in task["artifacts"][0]["parts"][0]["text"].lower()


def test_a2a_refuses_a_url_part():
    error = a2a("SendMessage", {"message": message("hi", parts=[{"url": "http://x/y.png"}])})["error"]
    assert error["code"] == -32005
