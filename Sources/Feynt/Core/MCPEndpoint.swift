import Foundation
import MLXLMCommon

/// Feynt as an MCP server: the local model, and the models on disk, as tools any MCP host
/// can call - Claude Code, Codex, an IDE.
///
/// Speaks the 2026-07-28 revision, where every request carries its own protocol version and
/// client capabilities in `_meta` and there is no handshake, no session and no GET stream.
/// Clients of the earlier revisions open with `initialize` instead, and most hosts in use
/// today still do, so that path is answered too - without minting a session, which the
/// earlier revisions allowed a server to omit.
///
/// Every response is a single JSON object. The transport also allows an SSE stream per
/// request, but nothing here has progress worth streaming to a tool caller: `generate`
/// returns when the answer is complete, the way a tool result has to be read anyway.
extension APIServer {
    static let mcpCurrentVersion = "2026-07-28"
    /// Initialization-era revisions, newest first. The first is offered to a client asking
    /// for one this list does not have.
    static let mcpLegacyVersions = ["2025-11-25", "2025-06-18", "2025-03-26"]

    func handleMCP(_ request: HTTPRequest, _ responder: HTTPResponder) {
        guard isAllowedOrigin(request) else {
            responder.sendJSON(
                status: 403, object: Self.rpcError(id: nil, code: -32600, message: "Origin not allowed"))
            return
        }
        guard let object = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any]
        else {
            responder.sendJSON(
                status: 400, object: Self.rpcError(id: nil, code: -32700, message: "Parse error"))
            return
        }
        guard let method = object["method"] as? String else {
            responder.sendJSON(
                status: 400,
                object: Self.rpcError(id: object["id"], code: -32600, message: "Invalid Request: no method"))
            return
        }
        // A notification - `notifications/initialized` from an older client is the usual
        // one - is accepted and answered with nothing.
        guard let id = object["id"], !(id is NSNull) else {
            responder.send(status: 202, contentType: "text/plain", body: Data())
            return
        }
        let params = object["params"] as? [String: Any] ?? [:]
        let requested = (params["_meta"] as? [String: Any])?["io.modelcontextprotocol/protocolVersion"]
            as? String

        if let requested {
            guard requested == Self.mcpCurrentVersion else {
                responder.sendJSON(
                    status: 400,
                    object: Self.rpcError(
                        id: id, code: -32022, message: "Unsupported protocol version",
                        data: ["supported": [Self.mcpCurrentVersion], "requested": requested]))
                return
            }
            if let mismatch = Self.mcpHeaderMismatch(request, method: method, params: params) {
                responder.sendJSON(
                    status: 400, object: Self.rpcError(id: id, code: -32020, message: mismatch))
                return
            }
        }

        let modern = requested != nil
        // A client that sends a progress token on a tool call gets the answer as an SSE
        // stream: progress notifications while the model works, then the result.
        let progressToken = (params["_meta"] as? [String: Any])?["progressToken"]
        let progress = progressToken.flatMap { token -> MCPProgress? in
            guard method == "tools/call",
                (request.headers["accept"] ?? "").contains("text/event-stream")
            else { return nil }
            return MCPProgress(token: token, responder: responder)
        }
        let task = Task { @MainActor in
            progress?.start()
            let (status, payload) = await self.dispatchMCP(
                method: method, params: params, id: id, modern: modern, progress: progress)
            if let progress {
                progress.finish(with: payload)
            } else {
                responder.sendJSON(status: status, object: payload)
            }
        }
        // A client that hangs up no longer wants the answer. Cancelling the task stops the
        // generation itself - `completeLocally` reads the engine's stream, and the stream
        // ends the decode loop when nobody reads it - so the GPU is free for whoever is next.
        responder.onPeerClosed = { task.cancel() }
    }

    private func dispatchMCP(
        method: String, params: [String: Any], id: Any, modern: Bool, progress: MCPProgress? = nil
    ) async -> (Int, [String: Any]) {
        switch method {
        case "initialize":
            let asked = params["protocolVersion"] as? String ?? ""
            let version = Self.mcpLegacyVersions.contains(asked) ? asked : Self.mcpLegacyVersions[0]
            return (200, Self.mcpResult(id: id, [
                "protocolVersion": version,
                "capabilities": ["tools": ["listChanged": false]],
                "serverInfo": mcpServerInfo,
                "instructions": Self.mcpInstructions,
            ]))
        case "server/discover":
            return (200, Self.mcpResult(id: id, [
                "supportedVersions": [Self.mcpCurrentVersion],
                "capabilities": ["tools": ["listChanged": false]],
                "instructions": Self.mcpInstructions,
                "ttlMs": 3_600_000,
                "cacheScope": "public",
                "_meta": ["io.modelcontextprotocol/serverInfo": mcpServerInfo],
            ]))
        case "ping":
            return (200, Self.mcpResult(id: id, [:]))
        case "tools/list":
            // The tool set is fixed for the life of the app, so a client may cache it for
            // as long as it likes; an hour keeps a long-lived host from never re-reading it.
            return (200, Self.mcpResult(id: id, [
                "tools": Self.mcpTools, "ttlMs": 3_600_000, "cacheScope": "public",
            ]))
        case "tools/call":
            let name = params["name"] as? String ?? ""
            let arguments = params["arguments"] as? [String: Any] ?? [:]
            guard Self.mcpTools.contains(where: { $0["name"] as? String == name }) else {
                return (200, Self.rpcError(id: id, code: -32602, message: "Unknown tool: \(name)"))
            }
            return (200, Self.mcpResult(id: id, await callTool(name, arguments, progress: progress)))
        default:
            // The current revision asks for 404 so a client can tell an unknown method on a
            // modern server from a server that has no modern endpoint at all.
            return (modern ? 404 : 200,
                    Self.rpcError(id: id, code: -32601, message: "Method not found: \(method)"))
        }
    }

    // MARK: - Tools

    private static let mcpInstructions = """
        Feynt runs language models locally on this Mac; nothing sent here leaves the machine. \
        `generate` answers a prompt on a local model - free and private, and slower and less \
        capable than a frontier model, so it suits drafts, boilerplate, summaries and second \
        opinions. `list_models` shows what is on disk and what is in memory; `load_model` makes \
        one active and `unload_model` frees its memory. Loading a model takes a few seconds; \
        generating with one that is not loaded loads it first.
        """

    private static let mcpTools: [[String: Any]] = [
        [
            "name": "list_models",
            "title": "List local models",
            "description": "The models Feynt offers, whether each is downloaded, loaded into "
                + "memory, and which one is active.",
            "inputSchema": ["type": "object", "properties": [String: Any]()],
            "annotations": ["readOnlyHint": true, "openWorldHint": false],
        ],
        [
            "name": "load_model",
            "title": "Load a model",
            "description": "Load a downloaded model into memory and make it the active one. "
                + "Other loaded models stay resident. Does not download: a model that is not "
                + "on disk has to be downloaded from Feynt's model window first.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "model": [
                        "type": "string",
                        "description": "Catalog id or repository, as list_models reports it.",
                    ]
                ],
                "required": ["model"],
            ],
            "annotations": [
                "readOnlyHint": false, "destructiveHint": false, "idempotentHint": true,
                "openWorldHint": false,
            ],
        ],
        [
            "name": "unload_model",
            "title": "Unload a model",
            "description": "Free a loaded model's memory. Without `model`, unloads the active "
                + "one. The files stay on disk.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "model": [
                        "type": "string",
                        "description": "Catalog id or repository; omit for the active model.",
                    ]
                ],
            ],
            "annotations": [
                "readOnlyHint": false, "destructiveHint": false, "idempotentHint": true,
                "openWorldHint": false,
            ],
        ],
        [
            "name": "generate",
            "title": "Generate with a local model",
            "description": "Answer a prompt with a local model on this Mac. Private and free; "
                + "suits drafts, boilerplate, summaries and second opinions. The model can look "
                + "things up on its own - fetch a public page, and read or search files in a "
                + "workspace folder you name - but it does not write or run anything. It reads "
                + "images passed in `images`. Pass `files` to have Feynt read files for it, and "
                + "`json_schema` for an answer guaranteed to parse. A model that cannot answer from "
                + "what it was given says `INSUFFICIENT:`, flagged as `insufficient`. Greedy decoding.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "prompt": ["type": "string", "description": "What to answer."],
                    "system": ["type": "string", "description": "Optional system prompt."],
                    "model": [
                        "type": "string",
                        "description": "Catalog id or repository; omit for the active model.",
                    ],
                    "max_tokens": [
                        "type": "integer", "minimum": 1, "maximum": 8192,
                        "description": "Upper bound on the answer's length. Default 1024.",
                    ],
                    "workspace": [
                        "type": "string",
                        "description": "Absolute path of a folder the model may read with its "
                            + "own read_file, list_files and grep tools while answering.",
                    ],
                    "tools": [
                        "type": "boolean",
                        "description": "Let the model use its read-only tools (web_fetch, and the "
                            + "file tools when a workspace is given). Default true.",
                    ],
                    "files": [
                        "type": "array", "items": ["type": "string"],
                        "description": "Paths Feynt reads and hands to the model with the prompt - "
                            + "text files, and images by their bytes. Absolute, or relative to "
                            + "`workspace`. Keeps the contents out of your own context; prefer it "
                            + "to asking the model to go and find them.",
                    ],
                    "json_schema": [
                        "type": "object",
                        "description": "A JSON Schema the answer must follow, enforced token by "
                            + "token; the result also carries the parsed value in "
                            + "`structuredContent.json`. Turns the model's tools off.",
                    ],
                    "images": [
                        "type": "array",
                        "description": "Pictures for the model to look at, each base64 in "
                            + "`data` with its `mimeType`, the shape of MCP image content. "
                            + "Inline only; URLs are not fetched.",
                        "items": [
                            "type": "object",
                            "properties": [
                                "data": ["type": "string"], "mimeType": ["type": "string"],
                            ],
                            "required": ["data"],
                        ],
                    ],
                ],
                "required": ["prompt"],
            ],
            // Read-only, but not closed-world: with tools on, the model may fetch a public page.
            "annotations": ["readOnlyHint": true, "openWorldHint": true],
        ],
    ]

    private var mcpServerInfo: [String: Any] {
        ["name": "feynt", "title": "Feynt", "version": appVersion]
    }

    private func callTool(
        _ name: String, _ arguments: [String: Any], progress: MCPProgress? = nil
    ) async -> [String: Any] {
        do {
            switch name {
            case "list_models":
                let models: [[String: Any]] = ModelCatalog.all.map { spec in
                    let loaded = engine.loadedModels.contains { $0.id == spec.id }
                    return [
                        "id": spec.id, "title": spec.title, "repo": spec.repo,
                        "downloaded": ModelResolver.isPresent(spec),
                        "loaded": loaded,
                        // `activeModel` outlives an unload - the controller keeps it as the
                        // model to come back to - so on its own it would call a model active
                        // that is not in memory, and an agent would expect `generate` to be
                        // instant on it.
                        "active": loaded && engine.activeModel?.id == spec.id,
                        "bestFor": spec.bestFor,
                        "sizeGB": (Double(spec.totalApproximateBytes) / 1e8).rounded() / 10,
                        "contextTokens": spec.contextLimit ?? Self.contextLength(of: spec) as Any,
                        "vision": Self.hasVision(spec),
                        "speculative": spec.drafterRepo != nil,
                    ]
                }
                let lines = models.map { m -> String in
                    var flags: [String] = []
                    if m["active"] as? Bool == true { flags.append("active") }
                    if m["loaded"] as? Bool == true { flags.append("loaded") }
                    flags.append(m["downloaded"] as? Bool == true ? "downloaded" : "not downloaded")
                    return "\(m["id"]!) - \(m["title"]!), \(m["sizeGB"]!) GB "
                        + "(\(flags.joined(separator: ", "))): \(m["bestFor"]!)"
                }
                return Self.toolResult(lines.joined(separator: "\n"), structured: ["models": models])

            case "load_model":
                guard let name = arguments["model"] as? String, !name.isEmpty else {
                    return Self.toolError("`model` is required")
                }
                let spec = try spec(named: name)
                let loaded = await exclusively { await engine.ensureLoaded(spec) }
                guard loaded else { throw LocalCompletionError.loadFailed(spec) }
                return Self.toolResult(
                    "\(spec.title) is loaded and active.",
                    structured: ["model": spec.repo, "active": true])

            case "unload_model":
                let target: ModelSpec
                if let name = arguments["model"] as? String, !name.isEmpty {
                    guard let spec = resolveModel(name) else {
                        throw LocalCompletionError.unknownModel(name)
                    }
                    target = spec
                } else if let active = engine.activeModel {
                    target = active
                } else {
                    return Self.toolResult("No model is loaded.", structured: ["unloaded": NSNull()])
                }
                let wasLoaded = engine.loadedModels.contains { $0.id == target.id }
                await exclusively { await engine.unload(target) }
                return Self.toolResult(
                    wasLoaded ? "\(target.title) is unloaded." : "\(target.title) was not loaded.",
                    structured: ["model": target.repo, "unloaded": wasLoaded])

            case "generate":
                guard let prompt = arguments["prompt"] as? String, !prompt.isEmpty else {
                    return Self.toolError("`prompt` is required")
                }
                let spec = try spec(named: arguments["model"] as? String)
                let maxTokens = min(max((arguments["max_tokens"] as? Int) ?? 1024, 1), 8192)
                let format = try Self.responseFormat(arguments["json_schema"])
                // The folder serves two things - the model's file tools and the paths in
                // `files` - so it is checked once, whether or not the tools are on.
                let workspace = try localTools(
                    enabled: true, workspace: arguments["workspace"] as? String)?.workspace
                // A schema-bound answer is JSON from its first token, so the model cannot
                // also call tools; it works from the prompt and the files.
                let toolsOn = (arguments["tools"] as? Bool) ?? true
                let tools = toolsOn && format == nil ? LocalTools(workspace: workspace) : nil

                var turns: [EngineTurn] = []
                if let system = arguments["system"] as? String, !system.isEmpty {
                    turns.append(EngineTurn(role: .system, content: system))
                }
                var images = try (arguments["images"] as? [[String: Any]] ?? []).map {
                    try ImageInput.decode(base64: $0["data"] as? String ?? "")
                }
                var content = prompt
                let paths = arguments["files"] as? [String] ?? []
                if !paths.isEmpty {
                    let loaded = try FileInputs.load(paths, workspace: workspace)
                    images += loaded.images
                    if !loaded.text.isEmpty {
                        content = loaded.text + "\n\n" + prompt + "\n\nAnswer from the files above. If "
                            + "they do not contain the answer, reply `\(Self.insufficientMarker)` "
                            + "followed by what is missing."
                    }
                }
                turns.append(EngineTurn(role: .user, content: content, images: images))

                progress?.note("generating with \(spec.title)")
                let started = Date()
                var firstToken: Date?
                let result = try await completeLocally(
                    turns: turns, spec: spec, maxTokens: maxTokens, tools: tools,
                    responseFormat: format,
                    onText: { chunk in
                        if firstToken == nil { firstToken = Date() }
                        progress?.generated(chunk)
                    },
                    onToolUse: { use in progress?.note("tool \(use.call.name)") })
                let seconds = Date().timeIntervalSince(started)
                // The word is enough: small models drop the colon ("INSUFFICIENT").
                let insufficient = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    .uppercased().hasPrefix("INSUFFICIENT")
                let usage: [String: Any] = [
                    "promptTokens": result.stats.promptTokens,
                    "cachedPromptTokens": result.stats.cachedPromptTokens,
                    "generatedTokens": result.stats.generatedTokens,
                    "seconds": (seconds * 100).rounded() / 100,
                    "timeToFirstTokenSeconds": firstToken.map {
                        ($0.timeIntervalSince(started) * 100).rounded() / 100
                    } as Any? ?? NSNull(),
                    "tokensPerSecond": (result.stats.tokensPerSecond * 10).rounded() / 10,
                ]
                var structured: [String: Any] = [
                    "text": result.text,
                    "model": spec.id,
                    "repo": spec.repo,
                    "insufficient": insufficient,
                    "toolCalls": result.toolCalls.map {
                        ["name": $0.call.name, "arguments": $0.call.argumentsJSON, "isError": $0.isError]
                    },
                    "usage": usage,
                ]
                if format != nil {
                    structured["json"] = (try? JSONSerialization.jsonObject(
                        with: Data(result.text.utf8), options: [.fragmentsAllowed])) ?? NSNull()
                }
                // The numbers go in a second content block, so the first stays exactly what
                // the model wrote - JSON an agent can parse as it is.
                let footer = "[\(spec.id) · \(result.stats.promptTokens) prompt + "
                    + "\(result.stats.generatedTokens) generated tokens · "
                    + String(format: "%.1f s · %.0f tok/s", seconds, result.stats.tokensPerSecond)
                    + (result.toolCalls.isEmpty ? "" : " · \(result.toolCalls.count) tool calls")
                    + (insufficient ? " · insufficient" : "") + "]"
                return [
                    "content": [["type": "text", "text": result.text], ["type": "text", "text": footer]],
                    "structuredContent": structured, "isError": false,
                ]

            default:
                return Self.toolError("Unknown tool: \(name)")
            }
        } catch {
            // A tool that ran and failed reports it in the result, where the model calling it
            // can read the reason and react; a JSON-RPC error would reach only the host.
            return Self.toolError(error.localizedDescription)
        }
    }

    // MARK: - Wire helpers

    /// `json_schema` as a JSON Schema object, wrapped the way the OpenAI-shaped decoder
    /// vendored from mac-mlx expects it.
    static func responseFormat(_ raw: Any?) throws -> ResponseFormat? {
        guard let schema = raw as? [String: Any] else { return nil }
        let data = try JSONSerialization.data(withJSONObject: [
            "type": "json_schema", "json_schema": ["schema": schema],
        ])
        do {
            return try ResponseFormatDecoder.decode(JSONDecoder().decode(JSONValue.self, from: data))
        } catch {
            throw LocalCompletionError.engine("json_schema: \(error.localizedDescription)")
        }
    }

    private static func checkpointConfig(_ spec: ModelSpec) -> [String: Any]? {
        guard let directory = ModelResolver.installedLocation(for: spec),
            let data = try? Data(contentsOf: directory.appending(component: "config.json"))
        else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    private static func contextLength(of spec: ModelSpec) -> Int? {
        guard let config = checkpointConfig(spec) else { return nil }
        let text = config["text_config"] as? [String: Any] ?? config
        return text["max_position_embeddings"] as? Int
    }

    private static func hasVision(_ spec: ModelSpec) -> Bool {
        checkpointConfig(spec)?["vision_config"] != nil
    }

    private static func toolResult(_ text: String, structured: [String: Any]) -> [String: Any] {
        ["content": [["type": "text", "text": text]], "structuredContent": structured, "isError": false]
    }

    private static func toolError(_ message: String) -> [String: Any] {
        ["content": [["type": "text", "text": message]], "isError": true]
    }

    static func rpcResult(id: Any, _ result: [String: Any]) -> [String: Any] {
        ["jsonrpc": "2.0", "id": id, "result": result]
    }

    /// An MCP result. 2026-07-28 makes `resultType` mandatory on every result, so a client
    /// can tell a finished answer from an `InputRequiredResult` asking it for something;
    /// the official Python SDK rejects a result without it. Nothing here ever asks the
    /// client for input, so every result is complete. Clients of earlier revisions ignore
    /// the field. Kept out of `rpcResult` because A2A wraps its Task in that one.
    static func mcpResult(id: Any, _ result: [String: Any]) -> [String: Any] {
        var result = result
        result["resultType"] = "complete"
        return rpcResult(id: id, result)
    }

    static func rpcError(
        id: Any?, code: Int, message: String, data: [String: Any]? = nil
    ) -> [String: Any] {
        var error: [String: Any] = ["code": code, "message": message]
        if let data { error["data"] = data }
        return ["jsonrpc": "2.0", "id": id ?? NSNull(), "error": error]
    }

    /// The current revision mirrors the method, the tool name and the protocol version into
    /// headers so a proxy can route on them, and requires the server to reject a request
    /// whose headers disagree with its body - otherwise a gateway could be shown one call and
    /// the server run another.
    private static func mcpHeaderMismatch(
        _ request: HTTPRequest, method: String, params: [String: Any]
    ) -> String? {
        let headers = request.headers
        guard headers["mcp-protocol-version"] == mcpCurrentVersion else {
            return "Header mismatch: MCP-Protocol-Version must be \(mcpCurrentVersion)"
        }
        guard headers["mcp-method"] == method else {
            return "Header mismatch: Mcp-Method must match the body's method '\(method)'"
        }
        if ["tools/call", "resources/read", "prompts/get"].contains(method) {
            let body = (params["name"] as? String) ?? (params["uri"] as? String) ?? ""
            guard let header = headers["mcp-name"].map(decodeHeaderValue), header == body else {
                return "Header mismatch: Mcp-Name must match the body's name '\(body)'"
            }
        }
        return nil
    }

    /// `=?base64?...?=` is how a client carries a value that is not plain ASCII.
    private static func decodeHeaderValue(_ value: String) -> String {
        guard value.hasPrefix("=?base64?"), value.hasSuffix("?="),
            value.count >= "=?base64??=".count
        else { return value }
        let encoded = String(value.dropFirst("=?base64?".count).dropLast("?=".count))
        guard let data = Data(base64Encoded: encoded), let text = String(data: data, encoding: .utf8)
        else { return value }
        return text
    }
}
