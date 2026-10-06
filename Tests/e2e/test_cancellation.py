"""A stopped request has to free the GPU, not finish an answer nobody will read.

Each test starts a long generation, stops it, then times a short one: on a free machine
that answers from the prefix cache in well under a second, while behind an essay it waits
for the essay. The long one is timed once to prove it really is long.
"""

import json
import time

import pytest

from conftest import ESSAY, MODEL, META, a2a, message, send_then_hang_up, short_answer, timed, tool

SLACK = 3.0  # seconds above an idle short answer that still count as "the GPU was free"


@pytest.fixture(scope="module")
def essay_seconds(warm):
    seconds, _ = timed(tool, "generate", {"prompt": ESSAY, "max_tokens": 4000, "model": MODEL})
    assert seconds > 10, f"the essay took {seconds:.1f} s; too short to tell a stop from an end"
    return seconds


def test_a2a_cancel_stops_generation(warm, essay_seconds):
    task = a2a("SendMessage", {"message": message(ESSAY), "configuration": {"returnImmediately": True},
                               "metadata": {"model": MODEL, "maxTokens": 4000}})["result"]["task"]
    time.sleep(1.5)
    a2a("CancelTask", {"id": task["id"]})
    assert short_answer() < warm + SLACK


def test_mcp_hang_up_stops_generation(warm, essay_seconds):
    body = json.dumps({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {
        "name": "generate", "arguments": {"prompt": ESSAY, "max_tokens": 4000, "model": MODEL},
        "_meta": META}}).encode()
    send_then_hang_up("/mcp", {"Content-Type": "application/json", "MCP-Protocol-Version": "2026-07-28",
                               "Mcp-Method": "tools/call", "Mcp-Name": "generate"}, body)
    assert short_answer() < warm + SLACK


def test_openai_stream_hang_up_stops_generation(warm, essay_seconds):
    body = json.dumps({"model": MODEL, "stream": True, "max_tokens": 4000, "temperature": 0,
                       "messages": [{"role": "user", "content": ESSAY}]}).encode()
    send_then_hang_up("/v1/chat/completions", {"Content-Type": "application/json"}, body)
    assert short_answer() < warm + SLACK
