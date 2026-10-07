import Foundation

/// One A2A task as this server keeps it.
///
/// Every message starts a task and every task runs to a terminal state - this agent never
/// asks for input mid-task - so a task is a record of one request and its answer. A
/// conversation continues across tasks through `contextId`, which is where the history lives.
struct A2ATaskRecord {
    let id: String
    let contextId: String
    var state: String
    var statusMessage: [String: Any]?
    var answer: String?
    var history: [[String: Any]]
    var updated = Date()

    static let terminal: Set<String> = [
        "TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED",
    ]
    var isTerminal: Bool { Self.terminal.contains(state) }

    func json(historyLength: Int? = nil) -> [String: Any] {
        var status: [String: Any] = [
            "state": state, "timestamp": ISO8601DateFormatter().string(from: updated),
        ]
        if let statusMessage { status["message"] = statusMessage }
        var task: [String: Any] = ["id": id, "contextId": contextId, "status": status]
        if let answer {
            task["artifacts"] = [A2ATaskRecord.artifact(taskId: id, text: answer)]
        }
        let kept = historyLength.map { $0 <= 0 ? [] : Array(history.suffix($0)) } ?? history
        if !kept.isEmpty { task["history"] = kept }
        return task
    }

    static func artifact(taskId: String, text: String) -> [String: Any] {
        ["artifactId": "\(taskId)-answer", "name": "answer", "parts": [["text": text]]]
    }
}

/// Task and conversation state for the A2A endpoint. Bounded: an agent that never asks for a
/// task again leaves nothing behind but the most recent couple of hundred records.
@MainActor
final class A2ARegistry {
    var tasks: [String: A2ATaskRecord] = [:]
    var workers: [String: Task<Void, Never>] = [:]
    var contexts: [String: [EngineTurn]] = [:]
    private var contextOrder: [String] = []

    private static let taskLimit = 200
    private static let contextLimit = 50

    func store(_ task: A2ATaskRecord) {
        tasks[task.id] = task
        guard tasks.count > Self.taskLimit else { return }
        let oldest = tasks.values.filter(\.isTerminal).sorted { $0.updated < $1.updated }
        for task in oldest.prefix(tasks.count - Self.taskLimit) { tasks[task.id] = nil }
    }

    func remember(_ turns: [EngineTurn], in contextId: String) {
        contexts[contextId] = turns
        contextOrder.removeAll { $0 == contextId }
        contextOrder.append(contextId)
        while contextOrder.count > Self.contextLimit {
            contexts[contextOrder.removeFirst()] = nil
        }
    }
}

/// Feynt as an A2A agent, over the JSON-RPC binding of A2A 1.0.
///
/// Field names are camelCase and enums travel as their SCREAMING_SNAKE names, both by the
/// spec's ProtoJSON rule; methods are PascalCase. A client of the 0.3 revision uses
/// slash-separated method names (`message/send`) and is told which version this is rather
/// than that the method does not exist.
extension APIServer {
    func agentCard() -> [String: Any] {
        [
            "name": "Feynt",
            "description": "A language model running locally on this Mac. Private and free: "
                + "nothing sent here leaves the machine. Suits drafts, code, summaries and "
                + "second opinions; slower and less capable than a frontier model.",
            "supportedInterfaces": [
                [
                    "url": "http://127.0.0.1:\(settings.port)/a2a",
                    "protocolBinding": "JSONRPC",
                    "protocolVersion": "1.0",
                ]
            ],
            "version": appVersion,
            "documentationUrl": "https://github.com/random1st/feynt",
            "capabilities": ["streaming": true, "pushNotifications": false, "extendedAgentCard": false],
            "defaultInputModes": ["text/plain", "image/png", "image/jpeg", "image/webp"],
            "defaultOutputModes": ["text/plain"],
            "skills": [
                [
                    "id": "local-generation",
                    "name": "Local generation",
                    "description": "Answers a message with a local model. A conversation "
                        + "continues across messages that share a contextId. Pick the model "
                        + "with `metadata.model` (a catalog id such as `uncensored-moe`); "
                        + "without it the active model answers. The model may look things up "
                        + "with read-only tools: it fetches public pages, and reads and searches "
                        + "files in `metadata.workspace` when that names a folder. "
                        + "`metadata.tools: false` turns them off. Images are read when sent "
                        + "as `raw` parts with an image `mediaType`; `url` parts are not fetched.",
                    "tags": ["local", "private", "offline", "code", "text", "vision"],
                    "examples": [
                        "Write a Swift function that parses an ISO-8601 date.",
                        "Summarise this diff in three sentences.",
                    ],
                ]
            ],
        ]
    }

    func handleA2A(_ request: HTTPRequest, _ responder: HTTPResponder) {
        guard isAllowedOrigin(request) else {
            responder.sendJSON(
                status: 403, object: Self.rpcError(id: nil, code: -32600, message: "Origin not allowed"))
            return
        }
        guard let object = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any],
            let method = object["method"] as? String
        else {
            responder.sendJSON(
                status: 200, object: Self.rpcError(id: nil, code: -32700, message: "Invalid JSON payload"))
            return
        }
        let id = object["id"] ?? NSNull()
        let params = object["params"] as? [String: Any] ?? [:]

        if let version = request.headers["a2a-version"], !version.isEmpty, !version.hasPrefix("1.") {
            responder.sendJSON(
                status: 200,
                object: Self.rpcError(
                    id: id, code: -32009,
                    message: "A2A version \(version) is not supported; this agent speaks 1.0"))
            return
        }
        if method.contains("/") {
            responder.sendJSON(
                status: 200,
                object: Self.rpcError(
                    id: id, code: -32009,
                    message: "'\(method)' is an A2A 0.3 method; this agent speaks 1.0, where it is "
                        + "PascalCase (SendMessage, SendStreamingMessage, GetTask, CancelTask)"))
            return
        }

        switch method {
        case "SendMessage":
            Task { @MainActor in await self.a2aSend(id: id, params: params, responder: responder) }
        case "SendStreamingMessage":
            Task { @MainActor in await self.a2aStream(id: id, params: params, responder: responder) }
        case "GetTask":
            guard let taskId = params["id"] as? String, let task = a2a.tasks[taskId] else {
                responder.sendJSON(status: 200, object: Self.rpcError(id: id, code: -32001, message: "Task not found"))
                return
            }
            responder.sendJSON(
                status: 200,
                object: Self.rpcResult(id: id, task.json(historyLength: params["historyLength"] as? Int)))
        case "CancelTask":
            guard let taskId = params["id"] as? String, var task = a2a.tasks[taskId] else {
                responder.sendJSON(status: 200, object: Self.rpcError(id: id, code: -32001, message: "Task not found"))
                return
            }
            guard !task.isTerminal else {
                responder.sendJSON(
                    status: 200,
                    object: Self.rpcError(
                        id: id, code: -32002, message: "Task is already \(task.state) and cannot be canceled"))
                return
            }
            a2a.workers[taskId]?.cancel()
            a2a.workers[taskId] = nil
            task.state = "TASK_STATE_CANCELED"
            task.updated = Date()
            a2a.store(task)
            responder.sendJSON(status: 200, object: Self.rpcResult(id: id, task.json()))
        case "ListTasks", "SubscribeToTask":
            responder.sendJSON(
                status: 200,
                object: Self.rpcError(id: id, code: -32004, message: "\(method) is not supported by this agent"))
        case "GetExtendedAgentCard":
            responder.sendJSON(
                status: 200,
                object: Self.rpcError(id: id, code: -32007, message: "No extended agent card is configured"))
        default:
            if method.contains("PushNotification") {
                responder.sendJSON(
                    status: 200,
                    object: Self.rpcError(id: id, code: -32003, message: "Push notifications are not supported"))
            } else {
                responder.sendJSON(
                    status: 200, object: Self.rpcError(id: id, code: -32601, message: "Method not found"))
            }
        }
    }

    // MARK: - Sending

    /// What a SendMessage request asks for, validated once for both the blocking and the
    /// streaming method.
    private struct A2ARequest {
        let message: [String: Any]
        let text: String
        let images: [Data]
        let contextId: String
        let spec: ModelSpec
        let tools: LocalTools?
        let maxTokens: Int
        let returnImmediately: Bool
        let historyLength: Int?
    }

    private func parseSend(_ params: [String: Any]) -> Result<A2ARequest, A2AFault> {
        guard var message = params["message"] as? [String: Any] else {
            return .failure(A2AFault(-32602, "params.message is required"))
        }
        guard (message["role"] as? String) == "ROLE_USER" else {
            return .failure(A2AFault(-32602, "message.role must be ROLE_USER"))
        }
        let parts = message["parts"] as? [[String: Any]] ?? []
        let text = parts.compactMap { $0["text"] as? String }.joined(separator: "\n")
        // Images come as `raw` bytes with an image media type. A `url` part is refused, not
        // fetched: following a URL an agent names would let it reach the local network.
        var images: [Data] = []
        for part in parts where part["text"] == nil {
            guard let raw = part["raw"] as? String,
                (part["mediaType"] as? String ?? "image/").hasPrefix("image/")
            else {
                return .failure(A2AFault(
                    -32005, "only text parts and images as raw bytes are supported"))
            }
            do {
                images.append(try ImageInput.decode(base64: raw))
            } catch {
                return .failure(A2AFault(-32602, error.localizedDescription))
            }
        }
        guard !text.isEmpty || !images.isEmpty else {
            return .failure(A2AFault(-32005, "the message has no text and no image"))
        }
        if let taskId = message["taskId"] as? String, !taskId.isEmpty {
            guard let task = a2a.tasks[taskId] else { return .failure(A2AFault(-32001, "Task not found")) }
            if task.isTerminal {
                return .failure(A2AFault(
                    -32004,
                    "task \(taskId) is \(task.state); send without taskId and with its contextId to continue"))
            }
        }
        let contextId = (message["contextId"] as? String).flatMap { $0.isEmpty ? nil : $0 }
            ?? UUID().uuidString
        message["contextId"] = contextId
        let metadata = (params["metadata"] as? [String: Any]) ?? (message["metadata"] as? [String: Any]) ?? [:]
        let spec: ModelSpec
        do {
            spec = try self.spec(named: metadata["model"] as? String)
        } catch {
            return .failure(A2AFault(-32602, error.localizedDescription))
        }
        let configuration = params["configuration"] as? [String: Any] ?? [:]
        let tools: LocalTools?
        do {
            tools = try localTools(
                enabled: (metadata["tools"] as? Bool) ?? true, workspace: metadata["workspace"] as? String)
        } catch {
            return .failure(A2AFault(-32602, error.localizedDescription))
        }
        return .success(A2ARequest(
            message: message, text: text, images: images, contextId: contextId, spec: spec, tools: tools,
            maxTokens: min(max((metadata["maxTokens"] as? Int) ?? 2048, 1), 8192),
            returnImmediately: configuration["returnImmediately"] as? Bool ?? false,
            historyLength: configuration["historyLength"] as? Int))
    }

    /// Creates the task, starts the work, and returns the worker that finishes it.
    private func startTask(
        _ request: A2ARequest, onText: ((String) -> Void)? = nil
    ) -> (A2ATaskRecord, Task<Void, Never>) {
        let taskId = UUID().uuidString
        var userMessage = request.message
        userMessage["taskId"] = taskId
        if userMessage["messageId"] == nil { userMessage["messageId"] = UUID().uuidString }
        let task = A2ATaskRecord(
            id: taskId, contextId: request.contextId, state: "TASK_STATE_WORKING",
            history: [userMessage])
        a2a.store(task)

        let prior = a2a.contexts[request.contextId] ?? []
        let turns = prior + [EngineTurn(role: .user, content: request.text, images: request.images)]
        let worker = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let result = try await self.completeLocally(
                    turns: turns, spec: request.spec, maxTokens: request.maxTokens,
                    tools: request.tools, onText: onText)
                guard var done = self.a2a.tasks[taskId], !done.isTerminal else { return }
                // The transcript, not the last round: a streaming client has already been sent
                // every word, and the artifact it can fetch later should match what it saw.
                let reply: [String: Any] = [
                    "messageId": UUID().uuidString, "contextId": request.contextId, "taskId": taskId,
                    "role": "ROLE_AGENT", "parts": [["text": result.transcript]],
                ]
                done.state = "TASK_STATE_COMPLETED"
                done.answer = result.transcript
                done.history.append(reply)
                done.updated = Date()
                self.a2a.store(done)
                self.a2a.remember(turns + [EngineTurn(role: .assistant, content: result.text)],
                                  in: request.contextId)
            } catch {
                guard var failed = self.a2a.tasks[taskId], !failed.isTerminal else { return }
                if Task.isCancelled || error is CancellationError {
                    failed.state = "TASK_STATE_CANCELED"
                } else {
                    failed.state = "TASK_STATE_FAILED"
                    failed.statusMessage = [
                        "messageId": UUID().uuidString, "contextId": request.contextId, "taskId": taskId,
                        "role": "ROLE_AGENT", "parts": [["text": error.localizedDescription]],
                    ]
                }
                failed.updated = Date()
                self.a2a.store(failed)
            }
            self.a2a.workers[taskId] = nil
        }
        a2a.workers[taskId] = worker
        return (task, worker)
    }

    private func a2aSend(id: Any, params: [String: Any], responder: HTTPResponder) async {
        let request: A2ARequest
        switch parseSend(params) {
        case .failure(let fault):
            responder.sendJSON(status: 200, object: Self.rpcError(id: id, code: fault.code, message: fault.message))
            return
        case .success(let parsed):
            request = parsed
        }
        let (task, worker) = startTask(request)
        if !request.returnImmediately { await worker.value }
        let current = a2a.tasks[task.id] ?? task
        responder.sendJSON(
            status: 200,
            object: Self.rpcResult(id: id, ["task": current.json(historyLength: request.historyLength)]))
    }

    /// Streams the answer as artifact chunks between a first `task` event and a final status.
    /// Each SSE event is a whole JSON-RPC response whose result is one StreamResponse.
    private func a2aStream(id: Any, params: [String: Any], responder: HTTPResponder) async {
        let request: A2ARequest
        switch parseSend(params) {
        case .failure(let fault):
            responder.sendJSON(status: 200, object: Self.rpcError(id: id, code: fault.code, message: fault.message))
            return
        case .success(let parsed):
            request = parsed
        }

        func emit(_ result: [String: Any]) {
            guard let data = try? JSONSerialization.data(withJSONObject: Self.rpcResult(id: id, result)),
                let line = String(data: data, encoding: .utf8)
            else { return }
            responder.writeEvent("data: \(line)\n\n")
        }

        responder.beginEventStream()
        var taskId = ""
        var first = true
        let (task, worker) = startTask(request) { chunk in
            emit([
                "artifactUpdate": [
                    "taskId": taskId, "contextId": request.contextId,
                    "artifact": A2ATaskRecord.artifact(taskId: taskId, text: chunk),
                    "append": !first, "lastChunk": false,
                ]
            ])
            first = false
        }
        taskId = task.id
        emit(["task": task.json()])
        await worker.value

        let final = a2a.tasks[task.id] ?? task
        var status: [String: Any] = [
            "state": final.state, "timestamp": ISO8601DateFormatter().string(from: final.updated),
        ]
        if let message = final.statusMessage { status["message"] = message }
        emit(["statusUpdate": ["taskId": final.id, "contextId": final.contextId, "status": status]])
        responder.finish()
    }
}

struct A2AFault: Error {
    let code: Int
    let message: String
    init(_ code: Int, _ message: String) {
        self.code = code
        self.message = message
    }
}
