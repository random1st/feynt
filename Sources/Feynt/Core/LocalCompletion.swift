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
    /// Every tool the model used on the way to `text`, in order.
    var toolCalls: [ToolUse] = []
    /// All text the model wrote, across the rounds a tool loop took - what a streaming caller
    /// already showed. `text` is the last round alone: the answer, without "let me check".
    var transcript: String = ""
}

/// One tool call as it ran: what was asked, and what came back.
struct ToolUse: Sendable {
    let call: EngineToolCall
    let result: String
    let isError: Bool
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
        turns: [EngineTurn], spec: ModelSpec, maxTokens: Int, tools: LocalTools? = nil,
        responseFormat: ResponseFormat? = nil,
        onText: ((String) -> Void)? = nil, onToolUse: ((ToolUse) -> Void)? = nil
    ) async throws -> LocalCompletion {
        await gate.acquire()
        defer { Task { await gate.release() } }
        try Task.checkCancellation()
        guard await engine.ensureLoaded(spec) else { throw LocalCompletionError.loadFailed(spec) }
        return try await Self.runToolLoop(
            turns: turns, spec: spec, maxTokens: maxTokens, tools: tools,
            responseFormat: responseFormat,
            generate: { [engine] turns, options in try await engine.generate(turns: turns, options: options) },
            onText: onText, onToolUse: onToolUse)
    }

    /// What a model says instead of guessing, and what the MCP result reports as
    /// `insufficient`, so an agent can tell "not in the material" from an answer.
    static let insufficientMarker = "INSUFFICIENT:"

    /// How many tool calls a single answer may make. Six is enough to list, grep, read two
    /// files and check a page; a model that needs more is lost rather than thorough, and on
    /// the last round it is asked to answer with what it has instead of being cut off.
    static let toolCallBudget = 6

    /// The loop shared by the chat, the MCP `generate` tool and A2A: generate, and when the
    /// model asks for tools, run them, hand back the results and generate again.
    ///
    /// The whole loop runs under one hold of the generation gate, so every round after the
    /// first reuses the prefix cache of the one before - a tool round costs its new tokens,
    /// not the conversation again.
    static func runToolLoop(
        turns initial: [EngineTurn], spec: ModelSpec, maxTokens: Int, tools: LocalTools?,
        thinking: Bool = false, responseFormat: ResponseFormat? = nil,
        generate: (_ turns: [EngineTurn], _ options: GenerationOptions) async throws -> AsyncStream<EngineEvent>,
        onText: ((String) -> Void)?, onReasoning: ((String) -> Void)? = nil,
        onToolUse: ((ToolUse) -> Void)?
    ) async throws -> LocalCompletion {
        var turns = initial
        if let tools, turns.first?.role != .system {
            turns.insert(EngineTurn(role: .system, content: Self.toolHint(tools)), at: 0)
        }
        var uses: [ToolUse] = []
        var transcript = ""
        var generated = 0
        var stats = GenerationStats()
        // Past the budget a model is offered no tools, and some ask anyway - the 35B-A3B
        // made 35 calls and returned an empty answer on 2026-10-07, because each refusal
        // only bought another round. It gets one nudge to answer; after that, the loop ends.
        var nudged = false

        while true {
            try Task.checkCancellation()
            let offered = (tools != nil && uses.count < toolCallBudget) ? tools?.definitions : nil
            let stream: AsyncStream<EngineEvent>
            do {
                stream = try await generate(
                    turns,
                    GenerationOptions(
                        maxTokens: maxTokens, temperature: 0, thinking: thinking, tools: offered,
                        responseFormat: responseFormat))
            } catch {
                throw LocalCompletionError.engine(error.localizedDescription)
            }

            var text = ""
            var calls: [EngineToolCall] = []
            for await event in stream {
                try Task.checkCancellation()
                switch event {
                case .text(let chunk):
                    text += chunk
                    onText?(chunk)
                case .toolCall(let call):
                    calls.append(call)
                case .finished(let finished):
                    stats = finished
                    generated += finished.generatedTokens
                case .reasoning(let chunk):
                    onReasoning?(chunk)
                }
            }
            try Task.checkCancellation()
            transcript += text

            guard let tools, !calls.isEmpty else {
                stats.generatedTokens = generated
                return LocalCompletion(
                    text: text, model: spec, stats: stats, toolCalls: uses, transcript: transcript)
            }
            if offered == nil && nudged {
                stats.generatedTokens = generated
                let answer = text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    ? "No answer: the model kept asking for tools after its \(toolCallBudget) "
                        + "calls were spent." : text
                return LocalCompletion(
                    text: answer, model: spec, stats: stats, toolCalls: uses, transcript: transcript)
            }
            turns.append(EngineTurn(role: .assistant, content: text, toolCalls: calls))
            for call in calls {
                // Past the budget a call is answered, not run, so every call the model made
                // still has a result to pair with - templates reject an unanswered one.
                let use: ToolUse
                if uses.count >= toolCallBudget {
                    use = ToolUse(
                        call: call,
                        result: "Tool budget spent; answer with what you have.", isError: true)
                } else if !tools.names.contains(call.name) {
                    use = ToolUse(call: call, result: "No such tool: \(call.name)", isError: true)
                } else {
                    let outcome = await tools.run(call)
                    use = ToolUse(call: call, result: outcome.text, isError: outcome.isError)
                }
                uses.append(use)
                onToolUse?(use)
                turns.append(EngineTurn(
                    role: .tool, content: use.result, toolCallID: call.id, toolName: call.name))
            }
            if offered == nil {
                nudged = true
                turns.append(EngineTurn(
                    role: .user,
                    content: "You have no tool calls left. Answer the question now from what the "
                        + "tools already returned."))
            }
        }
    }

    private static func toolHint(_ tools: LocalTools) -> String {
        // Small models read "only when needed" as permission to guess: asked where a value
        // is set in the repository, Qwen3.5-2B named a file that does not exist rather than
        // grep. So the hint says outright which questions need the tools.
        var hint = "You can call tools to look things up before you answer. When a question is "
            + "about the files in the working folder or about a web page, call the tools and "
            + "answer from what they return; never guess what a file or a page says. If they do "
            + "not contain the answer, reply `\(insufficientMarker)` followed by what is missing. "
            + "Answer anything else directly."
        if let workspace = tools.workspace {
            hint += " File tools work inside the folder \(workspace.root.path); paths are relative to it."
        }
        return hint
    }

    /// Runs `body` behind the generation gate. Loading and unloading go through it too: a
    /// load that lands in the middle of another request's generation would change which
    /// model is active under it.
    func exclusively<T>(_ body: () async throws -> T) async rethrows -> T {
        await gate.acquire()
        defer { Task { await gate.release() } }
        return try await body()
    }

    /// The tools for a request that may name a folder. A folder that does not exist is an
    /// error the caller sees, not a silent fall back to no file tools.
    func localTools(enabled: Bool, workspace path: String?) throws -> LocalTools? {
        guard enabled else { return nil }
        guard let path, !path.isEmpty else { return LocalTools(workspace: nil) }
        var isDirectory: ObjCBool = false
        let expanded = (path as NSString).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory),
            isDirectory.boolValue
        else { throw LocalCompletionError.engine("workspace \(path) is not a folder") }
        return LocalTools(workspace: Workspace(URL(fileURLWithPath: expanded)))
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
