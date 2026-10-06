import json
import os
import urllib.error
import urllib.request

from conftest import BASE, MODEL, META, mcp, tool

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_discover_names_the_current_revision():
    status, data = mcp("server/discover")
    assert status == 200
    assert data["result"]["supportedVersions"] == ["2026-07-28"]
    assert "ttlMs" in data["result"]


def test_tools_are_listed():
    status, data = mcp("tools/list")
    assert status == 200
    assert [t["name"] for t in data["result"]["tools"]] == ["list_models", "load_model", "unload_model", "generate"]


def test_every_result_says_it_is_complete():
    status, data = mcp("tools/list")
    assert data["result"]["resultType"] == "complete"


def test_list_models_reports_every_catalog_model():
    result = tool("list_models", {})
    ids = [m["id"] for m in result["structuredContent"]["models"]]
    assert MODEL in ids
    assert all({"downloaded", "loaded", "active"} <= m.keys() for m in result["structuredContent"]["models"])


def test_generate_answers(warm):
    result = tool("generate", {"prompt": "Reply with exactly: OK", "max_tokens": 8, "model": MODEL})
    assert result["isError"] is False
    assert "OK" in result["content"][0]["text"]


def test_header_that_disagrees_with_the_body_is_refused():
    status, data = mcp("tools/list", headers={"Mcp-Method": "tools/call"})
    assert status == 400 and data["error"]["code"] == -32020


def test_unsupported_revision_is_refused_with_the_supported_list():
    params = {"_meta": dict(META, **{"io.modelcontextprotocol/protocolVersion": "2099-01-01"})}
    body = json.dumps({"jsonrpc": "2.0", "id": 9, "method": "tools/list", "params": params}).encode()
    request = urllib.request.Request(BASE + "/mcp", data=body, headers={
        "Content-Type": "application/json", "MCP-Protocol-Version": "2099-01-01", "Mcp-Method": "tools/list"})
    try:
        urllib.request.urlopen(request)
        raise AssertionError("accepted an unknown revision")
    except urllib.error.HTTPError as error:
        data = json.loads(error.read())
        assert error.code == 400 and data["error"]["code"] == -32022
        assert data["error"]["data"]["supported"] == ["2026-07-28"]


def test_unknown_method():
    status, data = mcp("nope/method")
    assert status == 404 and data["error"]["code"] == -32601


def test_notification_is_accepted_without_a_body():
    status, data = mcp("notifications/initialized", rid=None, modern=False)
    assert status == 202 and data is None


def test_a_website_origin_is_refused():
    request = urllib.request.Request(BASE + "/mcp", data=b'{"jsonrpc":"2.0","id":1,"method":"ping"}',
                                     headers={"Content-Type": "application/json", "Origin": "http://evil.example"})
    try:
        urllib.request.urlopen(request)
        raise AssertionError("a foreign Origin got through")
    except urllib.error.HTTPError as error:
        assert error.code == 403


def test_get_and_delete_are_not_allowed():
    for method in ("GET", "DELETE"):
        try:
            urllib.request.urlopen(urllib.request.Request(BASE + "/mcp", method=method))
            raise AssertionError(f"{method} answered")
        except urllib.error.HTTPError as error:
            assert error.code == 405


def test_legacy_handshake_and_call(warm):
    status, data = mcp("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "old", "version": "1"}}, modern=False)
    assert status == 200 and data["result"]["protocolVersion"] == "2025-06-18"
    status, data = mcp("tools/list", modern=False)
    assert status == 200 and "result" in data
    result = tool("generate", {"prompt": "Say hi", "max_tokens": 6, "model": MODEL}, modern=False)
    assert result["isError"] is False and result["content"][0]["text"]


def test_unknown_model_is_a_tool_error():
    result = tool("load_model", {"model": "orcarouter-nonexistent"})
    assert result["isError"] is True
    assert "unknown model" in result["content"][0]["text"]


def test_unload_and_load_again():
    first = tool("unload_model", {"model": MODEL})
    assert first["isError"] is False
    again = tool("unload_model", {"model": MODEL})
    assert again["structuredContent"]["unloaded"] is False
    listed = {m["id"]: m for m in tool("list_models", {})["structuredContent"]["models"]}
    assert listed[MODEL]["loaded"] is False and listed[MODEL]["active"] is False
    loaded = tool("load_model", {"model": MODEL})
    assert loaded["isError"] is False
    listed = {m["id"]: m for m in tool("list_models", {})["structuredContent"]["models"]}
    assert listed[MODEL]["loaded"] is True and listed[MODEL]["active"] is True


# The model's own tools, through MCP `generate`.

def test_generate_reads_the_workspace(warm):
    result = tool("generate", {
        "prompt": "What is the value of maxBytes in Sources/Feynt/Tools/WebFetch.swift? Digits only.",
        "max_tokens": 300, "model": MODEL, "workspace": REPO})
    assert result["isError"] is False
    assert result["structuredContent"]["toolCalls"], "the model answered without looking"
    assert "2000000" in result["content"][0]["text"].replace(",", "").replace("_", "")


def test_generate_without_tools_makes_no_calls(warm):
    result = tool("generate", {"prompt": "Reply with exactly: OK", "max_tokens": 8, "model": MODEL,
                               "workspace": REPO, "tools": False})
    assert result["structuredContent"]["toolCalls"] == []


def test_a_workspace_that_is_not_a_folder_is_an_error():
    result = tool("generate", {"prompt": "hi", "max_tokens": 8, "model": MODEL, "workspace": "/nonexistent/folder"})
    assert result["isError"] is True
    assert "not a folder" in result["content"][0]["text"]
