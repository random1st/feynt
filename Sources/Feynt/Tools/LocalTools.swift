import Foundation

/// The tools a local model may call on its own: look things up, nothing more.
///
/// Read-only on purpose. A model this size handles "find where X is defined and summarise it"
/// well and multi-step edits badly, and every tool here can run without asking anyone - which
/// matters, because when Feynt answers another agent over MCP or A2A there is no human on the
/// other end to approve a write or a shell command. The file tools are confined to one folder
/// (`Workspace`); without one, only `web_fetch` is offered.
struct LocalTools: Sendable {
    let workspace: Workspace?

    /// What the model is told it has, in the OpenAI shape the chat template renders.
    var definitions: [[String: any Sendable]] {
        var tools: [[String: any Sendable]] = []
        if workspace != nil {
            tools.append(Self.function(
                "read_file",
                "Read a text file from the working folder. Returns numbered lines.",
                properties: [
                    "path": ["type": "string", "description": "Path relative to the working folder."],
                    "offset": ["type": "integer", "description": "First line to read, from 1."],
                    "limit": ["type": "integer", "description": "How many lines. Default 400."],
                ],
                required: ["path"]))
            tools.append(Self.function(
                "list_files",
                "List files in the working folder matching a glob such as `**/*.swift`.",
                properties: [
                    "pattern": ["type": "string", "description": "Glob; `**` crosses folders. Default `**/*`."]
                ],
                required: []))
            tools.append(Self.function(
                "grep",
                "Search file contents in the working folder with a regular expression. Returns "
                    + "file:line: text for each match.",
                properties: [
                    "pattern": ["type": "string", "description": "Regular expression."],
                    "glob": ["type": "string", "description": "Only files matching this glob."],
                    "ignore_case": ["type": "boolean"],
                ],
                required: ["pattern"]))
        }
        tools.append(Self.function(
            "web_fetch",
            "Fetch a public web page and return its text.",
            properties: ["url": ["type": "string", "description": "http or https URL."]],
            required: ["url"]))
        return tools
    }

    var names: Set<String> {
        Set(definitions.compactMap { ($0["function"] as? [String: any Sendable])?["name"] as? String })
    }

    /// Runs one call. A failure is returned as text for the model to read, never thrown: the
    /// model asked for something, and "that file is outside the folder" is an answer it can
    /// act on, where an exception would only end the turn.
    func run(_ call: EngineToolCall) async -> (text: String, isError: Bool) {
        let arguments = (try? JSONSerialization.jsonObject(with: Data(call.argumentsJSON.utf8)))
            as? [String: Any] ?? [:]
        do {
            let text: String
            switch call.name {
            case "read_file": text = try readFile(arguments)
            case "list_files": text = try listFiles(arguments)
            case "grep": text = try grep(arguments)
            case "web_fetch": text = try await WebFetch.fetch(arguments["url"] as? String ?? "")
            default: return ("No such tool: \(call.name)", true)
            }
            return (Self.clip(text), false)
        } catch {
            return (error.localizedDescription, true)
        }
    }

    /// The longest tool result the model sees. A whole file or page would crowd the
    /// conversation out of a context that also has to hold the answer.
    static let resultLimit = 16_000

    static func clip(_ text: String) -> String {
        guard text.count > resultLimit else { return text }
        return String(text.prefix(resultLimit)) + "\n[truncated: \(text.count - resultLimit) more characters]"
    }

    // MARK: - Files

    private struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    private func requireWorkspace() throws -> Workspace {
        guard let workspace else { throw Failure(message: "No working folder is set") }
        return workspace
    }

    private func readFile(_ arguments: [String: Any]) throws -> String {
        let workspace = try requireWorkspace()
        guard let path = arguments["path"] as? String, !path.isEmpty else {
            throw Failure(message: "`path` is required")
        }
        let url = try workspace.resolve(path)
        let data = try Data(contentsOf: url)
        guard !Self.looksBinary(data), let text = String(data: data, encoding: .utf8) else {
            throw Failure(message: "\(workspace.display(url)) is not a UTF-8 text file")
        }
        let lines = text.components(separatedBy: "\n")
        let offset = max((arguments["offset"] as? Int) ?? 1, 1)
        let limit = min(max((arguments["limit"] as? Int) ?? 400, 1), 2000)
        guard offset <= lines.count else {
            return "\(workspace.display(url)) has \(lines.count) lines; offset \(offset) is past the end"
        }
        let slice = lines[(offset - 1) ..< min(offset - 1 + limit, lines.count)]
        let body = slice.enumerated().map { "\(offset + $0.offset)\t\($0.element)" }.joined(separator: "\n")
        let more = offset - 1 + limit < lines.count
            ? "\n[\(lines.count - (offset - 1 + limit)) more lines; use offset to continue]" : ""
        return body + more
    }

    private func listFiles(_ arguments: [String: Any]) throws -> String {
        let workspace = try requireWorkspace()
        let pattern = (arguments["pattern"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "**/*"
        let matcher = try Glob(pattern)
        var found: [String] = []
        for url in Self.files(in: workspace) where matcher.matches(workspace.display(url)) {
            found.append(workspace.display(url))
            if found.count >= 300 { break }
        }
        if found.isEmpty { return "No files match \(pattern)" }
        let suffix = found.count >= 300 ? "\n[stopped at 300; narrow the pattern]" : ""
        return found.sorted().joined(separator: "\n") + suffix
    }

    private func grep(_ arguments: [String: Any]) throws -> String {
        let workspace = try requireWorkspace()
        guard let pattern = arguments["pattern"] as? String, !pattern.isEmpty else {
            throw Failure(message: "`pattern` is required")
        }
        let options: NSRegularExpression.Options = (arguments["ignore_case"] as? Bool) == true
            ? [.caseInsensitive] : []
        let regex: NSRegularExpression
        do { regex = try NSRegularExpression(pattern: pattern, options: options) } catch {
            throw Failure(message: "Not a valid regular expression: \(pattern)")
        }
        let matcher = try (arguments["glob"] as? String).flatMap { $0.isEmpty ? nil : $0 }.map(Glob.init)

        var hits: [String] = []
        scan: for url in Self.files(in: workspace) {
            let relative = workspace.display(url)
            if let matcher, !matcher.matches(relative) { continue }
            guard let data = try? Data(contentsOf: url), data.count <= 1_000_000,
                !Self.looksBinary(data), let text = String(data: data, encoding: .utf8)
            else { continue }
            for (number, line) in text.components(separatedBy: "\n").enumerated() {
                let range = NSRange(line.startIndex..., in: line)
                guard regex.firstMatch(in: line, range: range) != nil else { continue }
                hits.append("\(relative):\(number + 1): \(line.prefix(300))")
                if hits.count >= 200 { break scan }
            }
        }
        if hits.isEmpty { return "No matches for \(pattern)" }
        let suffix = hits.count >= 200 ? "\n[stopped at 200 matches; narrow the pattern or glob]" : ""
        return hits.joined(separator: "\n") + suffix
    }

    /// Directories nobody wants searched: build output and dependency caches would drown a
    /// result in generated files and make every grep slow.
    private static let skipped: Set<String> = [
        ".git", ".build", "build", "node_modules", "DerivedData", ".venv", "venv", "__pycache__",
        ".swiftpm", "dist", "target",
    ]

    private static func files(in workspace: Workspace) -> [URL] {
        guard let walker = FileManager.default.enumerator(
            at: workspace.root, includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
            options: [.skipsPackageDescendants])
        else { return [] }
        var result: [URL] = []
        for case let url as URL in walker {
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey])
            if values?.isDirectory == true {
                if skipped.contains(url.lastPathComponent) { walker.skipDescendants() }
                continue
            }
            // A symlink resolving out of the folder is skipped rather than followed.
            if values?.isRegularFile == true, workspace.contains(url.resolvingSymlinksInPath()),
                Workspace.secret(url.resolvingSymlinksInPath()) == nil
            {
                result.append(url)
            }
            if result.count >= 20_000 { break }
        }
        return result
    }

    private static func looksBinary(_ data: Data) -> Bool {
        data.prefix(8192).contains(0)
    }

    private static func function(
        _ name: String, _ description: String, properties: [String: any Sendable], required: [String]
    ) -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": name, "description": description,
                "parameters": ["type": "object", "properties": properties, "required": required]
                    as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }
}

/// `*` within a path segment, `**` across segments, `?` one character.
struct Glob {
    private let regex: NSRegularExpression

    init(_ pattern: String) throws {
        var out = "^"
        var inBraces = false
        var characters = Array(pattern)[...]
        while let c = characters.popFirst() {
            switch c {
            case "*":
                if characters.first == "*" {
                    characters.removeFirst()
                    if characters.first == "/" { characters.removeFirst(); out += "(?:.*/)?" } else { out += ".*" }
                } else {
                    out += "[^/]*"
                }
            case "?": out += "[^/]"
            // `{swift,py}` is alternation, as in a shell. Models write it unprompted -
            // `**/*.{swift,py,ts}` - and with the braces taken literally such a glob matched
            // nothing, so a search that would have found the answer came back empty.
            case "{" where !inBraces: inBraces = true; out += "(?:"
            case "}" where inBraces: inBraces = false; out += ")"
            case "," where inBraces: out += "|"
            case ".", "(", ")", "+", "|", "^", "$", "{", "}", "[", "]", "\\": out += "\\\(c)"
            default: out.append(c)
            }
        }
        regex = try NSRegularExpression(pattern: out + "$")
    }

    func matches(_ path: String) -> Bool {
        regex.firstMatch(in: path, range: NSRange(path.startIndex..., in: path)) != nil
    }
}
