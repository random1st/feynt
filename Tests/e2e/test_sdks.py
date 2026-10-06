"""The official clients, because what people actually run is what has to work."""

import asyncio
import inspect
import uuid

import pytest

from conftest import BASE, MODEL

mcp = pytest.importorskip("mcp")
a2a_client = pytest.importorskip("a2a.client")


async def _mcp_session(handshake):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(BASE + "/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await handshake(session)
            tools = await session.list_tools()
            result = await session.call_tool(
                "generate", {"prompt": "Name one prime number. Digits only.", "max_tokens": 6, "model": MODEL})
            return [t.name for t in tools.tools], result


@pytest.mark.parametrize("path", ["discover", "initialize"])
def test_mcp_sdk(path, warm):
    async def handshake(session):
        return await (session.discover() if path == "discover" else session.initialize())

    names, result = asyncio.run(_mcp_session(handshake))
    assert names == ["list_models", "load_model", "unload_model", "generate"]
    assert result.is_error is False
    assert any(ch.isdigit() for ch in result.content[0].text)


@pytest.mark.parametrize("streaming", [False, True])
def test_a2a_sdk(streaming, warm):
    from a2a.client import ClientConfig, create_client
    from a2a.types import GetTaskRequest, Message, Part, Role, SendMessageRequest, TaskState

    async def run():
        made = create_client(BASE, client_config=ClientConfig(streaming=streaming))
        client = await made if inspect.isawaitable(made) else made
        request = SendMessageRequest(message=Message(
            message_id=str(uuid.uuid4()), role=Role.ROLE_USER,
            parts=[Part(text="Name the capital of France. One word.")]))
        text, state, task_id = "", None, None
        async for event in client.send_message(request):
            kind = event.WhichOneof("payload")
            if kind == "task":
                task_id, state = event.task.id, TaskState.Name(event.task.status.state)
                text += "".join(p.text for a in event.task.artifacts for p in a.parts)
            elif kind == "artifact_update":
                text += "".join(p.text for p in event.artifact_update.artifact.parts)
            elif kind == "status_update":
                state = TaskState.Name(event.status_update.status.state)
        got = await client.get_task(GetTaskRequest(id=task_id))
        await client.close()
        return text, state, TaskState.Name(got.status.state)

    text, state, stored = asyncio.run(run())
    assert state == stored == "TASK_STATE_COMPLETED"
    assert "Paris" in text
