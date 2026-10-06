import json
import time
import urllib.request

from conftest import BASE, MODEL, a2a, message


def test_agent_card():
    card = json.load(urllib.request.urlopen(BASE + "/.well-known/agent-card.json"))
    assert card["name"] == "Feynt"
    assert card["capabilities"]["streaming"] is True
    assert [s["id"] for s in card["skills"]] == ["local-generation"]


def test_send_message_completes_and_keeps_the_context(warm):
    first = a2a("SendMessage", {"message": message("My name is Roman. Reply with exactly: noted"),
                                "metadata": {"model": MODEL}})["result"]["task"]
    assert first["status"]["state"] == "TASK_STATE_COMPLETED"
    assert first["artifacts"][0]["parts"][0]["text"]
    second = a2a("SendMessage", {"message": message("What is my name? One word.", context=first["contextId"]),
                                 "metadata": {"model": MODEL}})["result"]["task"]
    assert "Roman" in second["artifacts"][0]["parts"][0]["text"]
    got = a2a("GetTask", {"id": second["id"], "historyLength": 1})["result"]
    assert got["status"]["state"] == "TASK_STATE_COMPLETED"
    assert len(got.get("history", [])) == 1


def test_streaming_ends_in_a_final_status(warm):
    body = json.dumps({"jsonrpc": "2.0", "id": 7, "method": "SendStreamingMessage",
                       "params": {"message": message("Count from 1 to 5, comma separated."),
                                  "metadata": {"model": MODEL}}}).encode()
    request = urllib.request.Request(BASE + "/a2a", data=body,
                                     headers={"Content-Type": "application/json", "A2A-Version": "1.0"})
    response = urllib.request.urlopen(request, timeout=900)
    assert response.headers["Content-Type"].startswith("text/event-stream")
    kinds, text, final = [], "", None
    for line in response:
        line = line.decode().strip()
        if not line.startswith("data:"):
            continue
        event = json.loads(line[5:])["result"]
        kind = next(iter(event))
        kinds.append(kind)
        if kind == "artifactUpdate":
            text += event[kind]["artifact"]["parts"][0]["text"]
        if kind == "statusUpdate":
            final = event[kind]["status"]["state"]
    assert kinds[0] == "task" and kinds[-1] == "statusUpdate"
    assert final == "TASK_STATE_COMPLETED"
    assert "1" in text and "5" in text


def test_cancel_a_running_task_and_not_a_finished_one(warm):
    running = a2a("SendMessage", {"message": message("Write a 400-word essay about rivers."),
                                  "configuration": {"returnImmediately": True},
                                  "metadata": {"model": MODEL}})["result"]["task"]
    assert running["status"]["state"] in ("TASK_STATE_SUBMITTED", "TASK_STATE_WORKING")
    time.sleep(1.5)
    assert a2a("CancelTask", {"id": running["id"]})["result"]["status"]["state"] == "TASK_STATE_CANCELED"
    time.sleep(1)
    assert a2a("GetTask", {"id": running["id"]})["result"]["status"]["state"] == "TASK_STATE_CANCELED"
    assert a2a("CancelTask", {"id": running["id"]})["error"]["code"] == -32002


def test_errors():
    assert a2a("GetTask", {"id": "nope"})["error"]["code"] == -32001
    assert a2a("message/send", {"message": message("hi")})["error"]["code"] == -32009
    assert a2a("SendMessage", {"message": dict(message("hi"), role="ROLE_AGENT")})["error"]["code"] == -32602
    assert a2a("SendMessage", {"message": dict(message("hi"), parts=[{"url": "http://x"}])})["error"]["code"] == -32005
    assert a2a("ListTasks", {})["error"]["code"] == -32004
