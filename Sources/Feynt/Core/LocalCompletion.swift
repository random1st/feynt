import Foundation

/// What the agent protocols need from the engine: one conversation in, one answer out.
///
/// MCP and A2A both arrive as "run this prompt on a local model", and both have to queue
/// behind the OpenAI route rather than beside it - the GPU holds one generation at a time,
/// and a second one interleaved with the first would slow both.
struct LocalCompletion {
    let text: String
    let model: ModelSpec
    let stats: GenerationStats
}

enum LocalCompletionError: LocalizedError {
    case unknownModel(String)
    case notDownloaded(ModelSpec)
    case loadFailed(ModelSpec)
    case engine(String)

    var errorDescription: String? {
        switch self {
        case .unknownModel(let name):
            return "unknown model '\(name)'; list_models shows what this server has"
        case .notDownloaded(let spec):
            return "\(spec.title) is not downloaded. Download it from Feynt's model window - "
                + "a download is 16-20 GB, so it is not something an agent starts on its own."
        case .loadFailed(let spec):
            return "\(spec.title) did not load; Feynt's log has the reason"
        case .engine(let message):
            return message
        }
    }
}

extension APIServer {
    /// Resolves a model name the way the OpenAI route does, and refuses one that is not on
    /// disk instead of letting the load fail somewhere less readable.
    func spec(named name: String?) throws -> ModelSpec {
        guard let spec = resolveModel(name) else {
            throw LocalCompletionError.unknownModel(name ?? "")
        }
        guard ModelResolver.isPresent(spec) else { throw LocalCompletionError.notDownloaded(spec) }
        return spec
    }

    /// Runs `turns` to completion on `spec`, behind the shared generation gate.
    ///
    /// `onText` sees each chunk as it is decoded, for the callers that stream. Cancelling the
    /// task that awaits this stops the generation itself: the engine's stream cancels its
    /// worker when nobody is reading it any more, so a cancelled A2A task frees the GPU
    /// rather than finishing an answer nobody will receive.
    func completeLocally(
        turns: [EngineTurn], spec: ModelSpec, maxTokens: Int,
        onText: ((String) -> Void)? = nil
    ) async throws -> LocalCompletion {
        await gate.acquire()
        defer { Task { await gate.release() } }
        try Task.checkCancellation()

        guard await engine.ensureLoaded(spec) else { throw LocalCompletionError.loadFailed(spec) }
        let stream: AsyncStream<EngineEvent>
        do {
            stream = try await engine.generate(
                turns: turns,
                options: GenerationOptions(
                    maxTokens: maxTokens, temperature: 0, thinking: false, tools: nil))
        } catch {
            throw LocalCompletionError.engine(error.localizedDescription)
        }

        var text = ""
        var stats = GenerationStats()
        for await event in stream {
            try Task.checkCancellation()
            switch event {
            case .text(let chunk):
                text += chunk
                onText?(chunk)
            case .finished(let finished):
                stats = finished
            case .reasoning, .toolCall:
                // Thinking is off and no tools are offered on this path, so neither should
                // arrive; if one does, the answer is still the text.
                break
            }
        }
        try Task.checkCancellation()
        return LocalCompletion(text: text, model: spec, stats: stats)
    }

    /// Runs `body` behind the generation gate. Loading and unloading go through it too: a
    /// load that lands in the middle of another request's generation would change which
    /// model is active under it.
    func exclusively<T>(_ body: () async throws -> T) async rethrows -> T {
        await gate.acquire()
        defer { Task { await gate.release() } }
        return try await body()
    }

    /// DNS-rebinding guard for the endpoints a browser could otherwise reach.
    ///
    /// The listener is loopback-only, but a page on a hostile site can resolve its own name
    /// to 127.0.0.1 and post to this port from inside the user's browser. Browsers always
    /// send `Origin` on such a request, so an Origin that is present and is not this machine
    /// is refused. No Origin at all is a non-browser client - Claude Code, curl, an agent -
    /// and is let through. MCP's Streamable HTTP transport makes this check a MUST.
    func isAllowedOrigin(_ request: HTTPRequest) -> Bool {
        guard let origin = request.headers["origin"], !origin.isEmpty, origin != "null" else {
            return true
        }
        guard let host = URLComponents(string: origin)?.host?.lowercased() else { return false }
        return host == "localhost" || host == "127.0.0.1" || host == "::1" || host == "[::1]"
    }

    /// The app's own version, for the identity both protocols ask a server to report.
    var appVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "dev"
    }
}
