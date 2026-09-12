import Foundation
import SwiftUI

/// Cumulative counters served at `/metrics` — the same numbers the tray shows.
struct APIMetrics: Sendable, Equatable {
    var requests = 0
    var promptTokens = 0
    var completionTokens = 0
    private var decodeSeconds: Double = 0
    private var acceptedPerStepSum: Double = 0
    private var acceptedSamples = 0

    var meanDecodeTokensPerSecond: Double {
        decodeSeconds > 0 ? Double(completionTokens) / decodeSeconds : 0
    }
    var meanAcceptLength: Double {
        acceptedSamples > 0 ? acceptedPerStepSum / Double(acceptedSamples) : 0
    }

    mutating func record(_ stats: GenerationStats) {
        promptTokens += stats.promptTokens
        completionTokens += stats.generatedTokens
        if stats.tokensPerSecond > 0 {
            decodeSeconds += Double(stats.generatedTokens) / stats.tokensPerSecond
        }
        if stats.acceptedPerStep > 0 {
            acceptedPerStepSum += stats.acceptedPerStep
            acceptedSamples += 1
        }
    }
}

/// Serialises generations. The engine holds one model and the GPU is not shareable, so a
/// second request waits rather than interleaving.
actor GenerationGate {
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        if !busy {
            busy = true
            return
        }
        await withCheckedContinuation { waiting.append($0) }
    }

    func release() {
        if waiting.isEmpty {
            busy = false
        } else {
            waiting.removeFirst().resume()
        }
    }
}

/// OpenAI-compatible endpoint served from inside the app, so external clients keep working
/// after the Python server was dropped.
@MainActor
final class APIServer: ObservableObject, EngineLifecycleObserver {
    @Published private(set) var isRunning = false
    @Published private(set) var metrics = APIMetrics()
    @Published private(set) var lastError: String?

    private var http: HTTPServer?
    private let gate = GenerationGate()
    private unowned let engine: EngineController
    private let settings: AppSettings

    init(engine: EngineController, settings: AppSettings) {
        self.engine = engine
        self.settings = settings
    }

    var baseURL: String { "http://127.0.0.1:\(settings.port)/v1" }

    func start() {
        guard !isRunning else { return }
        let server = HTTPServer { [weak self] request, responder in
            Task { @MainActor in self?.route(request, responder) }
        }
        do {
            try server.start(port: UInt16(clamping: settings.port))
            http = server
            isRunning = true
            lastError = nil
            AppLog.write("api server listening on 127.0.0.1:\(settings.port)")
        } catch {
            lastError = error.localizedDescription
            AppLog.write("api server failed: \(error.localizedDescription)")
        }
    }

    func stop() {
        http?.stop()
        http = nil
        isRunning = false
    }

    /// Re-bind after a port change; silently does nothing when the server is stopped.
    func restartIfRunning() {
        guard isRunning else { return }
        stop()
        start()
    }

    func engineDidLoad() { start() }

    /// Deliberately not `stop()`. The idle timeout unloads the weights, which is the whole
    /// point of it, but a client that then connects should get an answer rather than a
    /// refused connection: `run(_:responder:)` reloads on demand. Tearing the listener down
    /// made the two features contradict each other - the endpoint a user copied out of the
    /// menu bar stopped existing five minutes after they last used it.
    func engineDidUnload() {}

    // MARK: - Routing

    private func route(_ request: HTTPRequest, _ responder: HTTPResponder) {
        let path = request.path.components(separatedBy: "?").first ?? request.path
        switch (request.method, path) {
        case ("GET", "/health"):
            responder.sendJSON(status: 200, object: healthPayload())
        case ("GET", "/metrics"):
            responder.sendJSON(status: 200, object: metricsPayload())
        case ("GET", "/v1/models"):
            responder.sendJSON(status: 200, object: modelsPayload())
        case ("POST", "/v1/chat/completions"):
            handleCompletion(request, responder)
        case ("OPTIONS", _):
            responder.send(status: 200, contentType: "text/plain", body: Data())
        default:
            responder.sendJSON(
                status: 404, object: ["error": ["message": "unknown route \(path)"]])
        }
    }

    private func healthPayload() -> [String: Any] {
        var payload: [String: Any] = [
            "model": engine.activeModel?.repo ?? settings.selectedModel.repo,
            "loaded": engine.loadedModels.map(\.repo),
            "mode": engine.stats.speculative ? "speculative" : "plain",
        ]
        switch engine.state {
        case .ready, .generating: payload["status"] = "ok"
        case .loading: payload["status"] = "loading"
        case .failed(let message):
            payload["status"] = "error"
            payload["error"] = message
        case .unloaded: payload["status"] = "no_model"
        }
        return payload
    }

    private func metricsPayload() -> [String: Any] {
        [
            "model": engine.activeModel?.repo ?? settings.selectedModel.repo,
            "mode": engine.stats.speculative ? "speculative" : "plain",
            "requests": metrics.requests,
            "prompt_tokens": metrics.promptTokens,
            "completion_tokens": metrics.completionTokens,
            "mean_decode_tokens_per_sec": metrics.meanDecodeTokensPerSecond,
            "mean_accept_len": metrics.meanAcceptLength,
        ]
    }

    private func modelsPayload() -> [String: Any] {
        let created = Int(Date().timeIntervalSince1970)
        let ids = ["local"] + ModelCatalog.all.filter(ModelResolver.isPresent).map(\.repo)
        return [
            "object": "list",
            "data": ids.map {
                ["id": $0, "object": "model", "created": created, "owned_by": "feynt"]
            },
        ]
    }

    // MARK: - Chat completions

    private func handleCompletion(_ request: HTTPRequest, _ responder: HTTPResponder) {
        guard let parsed = ChatRequest(body: request.body) else {
            responder.sendJSON(
                status: 400,
                object: [
                    "error": [
                        "message":
                            "no usable messages: expected a non-empty `messages` array whose "
                            + "entries carry text content, as a string or as typed parts"
                    ]
                ])
            return
        }
        guard let spec = resolveModel(parsed.model) else {
            responder.sendJSON(
                status: 404,
                object: [
                    "error": [
                        "message":
                            "unknown model '\(parsed.model ?? "")'; GET /v1/models lists what "
                            + "this server has",
                        "code": "model_not_found",
                    ]
                ])
            return
        }
        guard ModelResolver.isPresent(spec) else {
            responder.sendJSON(
                status: 404,
                object: [
                    "error": [
                        "message": "\(spec.title) is not downloaded", "code": "model_not_found",
                    ]
                ])
            return
        }
        metrics.requests += 1

        Task { [weak self] in
            guard let self else { return }
            // One generation at a time; everything else queues behind this.
            await self.gate.acquire()
            defer { Task { await self.gate.release() } }
            await self.run(parsed, spec: spec, responder: responder)
        }
    }

    /// Clients like pi name a model. With more than one resident the name decides which
    /// one answers, and an unknown name is an error rather than silently the current one:
    /// a typo would otherwise run on the wrong weights and nobody would know.
    private func resolveModel(_ name: String?) -> ModelSpec? {
        guard let name, !name.isEmpty, name != "local" else { return settings.selectedModel }
        let wanted = name.lowercased()
        return ModelCatalog.all.first { spec in
            [spec.id, spec.repo, spec.title, spec.directoryName].contains { $0.lowercased() == wanted }
                || spec.repo.lowercased().hasSuffix("/" + wanted)
        }
    }

    private func run(_ request: ChatRequest, spec: ModelSpec, responder: HTTPResponder) async {
        guard await engine.ensureLoaded(spec) else {
            responder.sendJSON(
                status: 503, object: ["error": ["message": "no model loaded"]])
            return
        }

        let options = GenerationOptions(
            maxTokens: request.maxTokens,
            temperature: request.temperature,
            thinking: request.thinking ?? settings.thinkingByDefault)

        let stream: AsyncStream<EngineEvent>
        do {
            stream = try await engine.generate(turns: request.turns, options: options)
        } catch {
            responder.sendJSON(
                status: 503, object: ["error": ["message": error.localizedDescription]])
            return
        }

        let id = "chatcmpl-\(UUID().uuidString.prefix(24))"
        let model = spec.repo
        var text = ""
        var reasoning = ""
        var stats = GenerationStats()

        if request.stream { responder.beginEventStream() }

        for await event in stream {
            switch event {
            case .text(let chunk):
                text += chunk
                if request.stream {
                    responder.writeEvent(
                        Self.sseChunk(id: id, model: model, delta: ["content": chunk]))
                }
            case .reasoning(let chunk):
                reasoning += chunk
                if request.stream {
                    responder.writeEvent(
                        Self.sseChunk(id: id, model: model, delta: ["reasoning_content": chunk]))
                }
            case .finished(let value):
                stats = value
            }
        }

        metrics.record(stats)
        engine.generationFinished(stats)

        if request.stream {
            responder.writeEvent(
                Self.sseChunk(id: id, model: model, delta: [:], finish: "stop"))
            responder.writeEvent("data: [DONE]\n\n")
            responder.finish()
        } else {
            var message: [String: Any] = ["role": "assistant", "content": text]
            // The existing client reads the thinking block from this field.
            if !reasoning.isEmpty { message["reasoning_content"] = reasoning }
            responder.sendJSON(
                status: 200,
                object: [
                    "id": id,
                    "object": "chat.completion",
                    "created": Int(Date().timeIntervalSince1970),
                    "model": model,
                    "choices": [["index": 0, "message": message, "finish_reason": "stop"]],
                    "usage": [
                        "prompt_tokens": stats.promptTokens,
                        "completion_tokens": stats.generatedTokens,
                        "total_tokens": stats.promptTokens + stats.generatedTokens,
                    ],
                ])
        }
    }

    private static func sseChunk(
        id: String, model: String, delta: [String: Any], finish: String? = nil
    ) -> String {
        let payload: [String: Any] = [
            "id": id,
            "object": "chat.completion.chunk",
            "created": Int(Date().timeIntervalSince1970),
            "model": model,
            "choices": [
                ["index": 0, "delta": delta, "finish_reason": finish as Any? ?? NSNull()]
            ],
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let json = String(data: data, encoding: .utf8)
        else { return "" }
        return "data: \(json)\n\n"
    }
}

/// Lenient decode: unknown fields are ignored rather than rejected, because clients send
/// plenty of OpenAI parameters this engine has no use for.
private struct ChatRequest {
    let turns: [EngineTurn]
    let maxTokens: Int
    let temperature: Float
    let stream: Bool
    let thinking: Bool?
    let model: String?

    /// OpenAI messages carry either a string or a list of typed parts, and real clients
    /// send both: pi puts its system prompt in a string and the user's turn in
    /// `[{"type": "text", "text": …}]`. Reading only the string form dropped that turn on
    /// the floor, leaving a conversation with a system prompt and nothing to answer — which
    /// the chat template rejects outright, so the request came back as a Jinja exception
    /// rather than as anything a client could act on.
    ///
    /// Non-text parts (images, audio) are skipped: this engine is text-only, and a caption
    /// invented for an image would be worse than an answer that ignores it.
    private static func text(from content: Any?) -> String? {
        if let text = content as? String { return text }
        guard let parts = content as? [[String: Any]] else { return nil }
        let pieces = parts.compactMap { part -> String? in
            guard part["type"] as? String == "text" else { return nil }
            return part["text"] as? String
        }
        return pieces.isEmpty ? nil : pieces.joined(separator: "\n")
    }

    init?(body: Data) {
        guard let root = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else {
            return nil
        }
        let rawMessages = root["messages"] as? [[String: Any]] ?? []
        turns = rawMessages.compactMap { entry in
            guard let content = Self.text(from: entry["content"]) else { return nil }
            let role = EngineTurn.Role(rawValue: entry["role"] as? String ?? "user") ?? .user
            return EngineTurn(role: role, content: content)
        }
        guard !turns.isEmpty else { return nil }

        maxTokens = (root["max_tokens"] as? Int) ?? (root["max_completion_tokens"] as? Int) ?? 4096
        // Greedy unless the client asks otherwise. The speculative loop verifies greedily,
        // so a sampling default sent every unqualified request down the slow path while the
        // server still reported that speculation was on.
        temperature = (root["temperature"] as? NSNumber).map { $0.floatValue } ?? 0
        stream = (root["stream"] as? Bool) ?? false
        model = root["model"] as? String
        let kwargs = root["chat_template_kwargs"] as? [String: Any]
        thinking = kwargs?["enable_thinking"] as? Bool
    }
}
