"""What an agent hands a local model: files by path, a schema, and patience for progress."""

import asyncio
import os
import tempfile

import pytest

from conftest import BASE, MODEL, tool

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_files_are_read_by_feynt(warm):
    result = tool("generate", {"model": MODEL, "max_tokens": 60, "tools": False,
                               "prompt": "What is the value of maxBytes? Digits only.",
                               "files": ["Sources/Feynt/Tools/WebFetch.swift"], "workspace": REPO})
    assert result["isError"] is False
    assert "2000000" in result["content"][0]["text"].replace("_", "").replace(",", "")
    assert result["structuredContent"]["toolCalls"] == []


def test_absolute_file_without_workspace(warm):
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write("12:00 started\n12:01 ERROR disk full on /var\n12:02 stopped\n")
    try:
        result = tool("generate", {"model": MODEL, "max_tokens": 40, "tools": False, "files": [f.name],
                                   "prompt": "Which error is in this log? A few words."})
        assert "disk" in result["content"][0]["text"].lower()
    finally:
        os.unlink(f.name)


def test_a_question_the_files_cannot_answer_is_flagged(warm):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("The office opens at 9.\n")
    try:
        result = tool("generate", {"model": MODEL, "max_tokens": 60, "tools": False, "files": [f.name],
                                   "prompt": "What is the CEO's phone number?"})
        assert result["structuredContent"]["insufficient"] is True, result["content"][0]["text"]
    finally:
        os.unlink(f.name)


def test_secret_files_are_refused():
    result = tool("generate", {"model": MODEL, "prompt": "hi", "files": [os.path.expanduser("~/.ssh/config")]})
    assert result["isError"] is True and "credential" in result["content"][0]["text"]


def test_json_schema_gives_parseable_json(warm):
    schema = {"type": "object", "required": ["name", "email", "amount"],
              "properties": {"name": {"type": "string"}, "email": {"type": "string"},
                             "amount": {"type": "number"}}}
    result = tool("generate", {"model": MODEL, "max_tokens": 120, "json_schema": schema,
                               "prompt": "Extract from: 'Maria Lopez (maria@example.com) wants a refund of 49.90 EUR.'"})
    value = result["structuredContent"]["json"]
    assert value["name"] == "Maria Lopez" and value["email"] == "maria@example.com"
    assert abs(value["amount"] - 49.9) < 0.01


def test_usage_is_reported(warm):
    result = tool("generate", {"model": MODEL, "max_tokens": 20, "tools": False, "prompt": "Say hi."})
    usage = result["structuredContent"]["usage"]
    assert usage["generatedTokens"] > 0 and usage["tokensPerSecond"] > 0 and usage["seconds"] > 0
    assert result["content"][1]["text"].startswith("[")


def test_list_models_helps_choose():
    models = tool("list_models", {})["structuredContent"]["models"]
    for m in models:
        assert m["bestFor"] and m["sizeGB"] > 0 and "vision" in m and "speculative" in m


def test_progress_notifications_arrive(warm):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    seen = []

    async def run():
        async with streamable_http_client(BASE + "/mcp") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                async def on_progress(progress, total, message):
                    seen.append(message)

                return await session.call_tool(
                    "generate", {"model": MODEL, "prompt": "Count from 1 to 300.", "max_tokens": 600,
                                 "tools": False}, progress_callback=on_progress)

    result = asyncio.run(run())
    assert result.is_error is False
    assert len(seen) >= 2, seen
