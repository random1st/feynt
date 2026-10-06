import DFlashKit
import Foundation

/// One headless turn through the real engine stack, behind `Feynt --generate <id> [prompt]`.
///
/// Added when LFM2.5 joined the catalog. A new architecture has to be checked against the
/// reference runtime — same prompt, greedy, compare the text and the rate — and doing that
/// through the chat window proves nothing about the headless path the API server uses.
/// Two tools that are hard to confuse: one obviously right for a weather question, one
/// obviously wrong. A single-tool probe cannot tell "the model can call" from "the model
/// calls whatever it is handed". Written as JSON because that is the shape a client sends
/// and the shape the template reads — building it as nested Swift dictionaries would only
/// prove that a different spelling also works.
private let probeToolsJSON = """
    [{"type": "function", "function": {"name": "get_weather",
      "description": "Current weather for a city.",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                     "required": ["city"]}}},
     {"type": "function", "function": {"name": "send_email",
      "description": "Send an email. Use only when explicitly asked to send one.",
      "parameters": {"type": "object", "properties": {"to": {"type": "string"},
                     "body": {"type": "string"}}, "required": ["to", "body"]}}}]
    """

private var probeTools: [[String: any Sendable]]? {
    (try? JSONSerialization.jsonObject(with: Data(probeToolsJSON.utf8)))
        as? [[String: any Sendable]]
}

@MainActor func generateOnce(spec: ModelSpec, prompt: String) async {
    let engine = AppState.shared.engine
    let started = Date()
    guard await engine.ensureLoaded(spec) else {
        print("load failed: \(engine.state)")
        exit(1)
    }
    print(String(format: "loaded %@ in %.1fs, speculative: %@", spec.repo,
                 Date().timeIntervalSince(started), engine.speculative ? "yes" : "no"))

    // `--chat` drives the real ChatStore - the object behind the chat window - so the parts
    // a tool loop adds there can be checked without clicking: one bubble per round, a tool
    // row between them, and the calls going back with the history on the next message.
    // `--follow-up <text>` sends a second message in the same conversation.
    if CommandLine.arguments.contains("--chat") {
        let settings = AppState.shared.settings
        settings.selectedModelID = spec.id
        let chat = ChatStore(engine: engine, settings: settings)
        chat.toolsEnabled = true
        chat.workspace = CommandLine.arguments.firstIndex(of: "--workspace")
            .flatMap { CommandLine.arguments.count > $0 + 1 ? CommandLine.arguments[$0 + 1] : nil }
            .map { Workspace(URL(fileURLWithPath: $0)) }
        let followUp = CommandLine.arguments.firstIndex(of: "--follow-up")
            .flatMap { CommandLine.arguments.count > $0 + 1 ? CommandLine.arguments[$0 + 1] : nil }
        for question in [prompt] + (followUp.map { [$0] } ?? []) {
            chat.draft = question
            chat.send()
            while chat.isStreaming { try? await Task.sleep(for: .milliseconds(200)) }
        }
        for message in chat.messages {
            switch message.role {
            case .tool:
                let head = message.text.split(separator: "\n", maxSplits: 1).first.map(String.init) ?? ""
                print("  [tool \(message.toolName ?? "?") \(message.toolArguments ?? "")] -> \(message.isError ? "ERROR " : "")\(head.prefix(90))")
            default:
                let calls = message.toolCalls.isEmpty ? "" : " (asked for \(message.toolCalls.map(\.name).joined(separator: ", ")))"
                let text = message.text.replacingOccurrences(of: "\n", with: " ").prefix(140)
                print("\(message.role.rawValue)\(calls): \(text)")
            }
        }
        if let error = chat.errorMessage { print("ERROR: \(error)") }
        exit(0)
    }

    // `--local-tools` runs the model with Feynt's own read-only tools and the same loop the
    // chat, MCP and A2A use, so the loop can be checked against the real model without a UI.
    // `--workspace <dir>` gives the file tools a folder; without it only web_fetch is offered.
    if CommandLine.arguments.contains("--local-tools") {
        let workspace = CommandLine.arguments.firstIndex(of: "--workspace")
            .flatMap { CommandLine.arguments.count > $0 + 1 ? CommandLine.arguments[$0 + 1] : nil }
            .map { Workspace(URL(fileURLWithPath: $0)) }
        do {
            let result = try await APIServer.runToolLoop(
                turns: [EngineTurn(role: .user, content: prompt)], spec: spec, maxTokens: 1024,
                tools: LocalTools(workspace: workspace),
                generate: { turns, options in try await engine.generate(turns: turns, options: options) },
                onText: nil,
                onToolUse: { use in
                    let args = use.call.argumentsJSON.prefix(120)
                    let head = use.result.split(separator: "\n", maxSplits: 1).first.map(String.init) ?? ""
                    print("TOOL \(use.call.name) \(args) -> \(use.isError ? "ERROR " : "")\(head.prefix(140))")
                })
            print("--- answer ---")
            print(result.text)
            print("--- \(result.toolCalls.count) tool calls · \(result.stats.generatedTokens) tokens ---")
        } catch {
            print("loop failed: \(error.localizedDescription)")
            exit(1)
        }
        exit(0)
    }

    // `--with-tools` hands the model a small toolbox, so the run also proves the parts a
    // plain prompt cannot: that the chat template accepts tools, that the model asks for one
    // in its own dialect, and that the parser turns that back into a call.
    let tools: [[String: any Sendable]]? =
        CommandLine.arguments.contains("--with-tools") ? probeTools : nil

    // A model that reasons before answering spends the budget on thinking, so a fixed 200
    // measures the reasoning rather than the reply. `--max-tokens` gives a run room.
    let maxTokens =
        CommandLine.arguments.firstIndex(of: "--max-tokens")
        .flatMap { index -> Int? in
            guard CommandLine.arguments.count > index + 1 else { return nil }
            return Int(CommandLine.arguments[index + 1])
        } ?? 200

    do {
        let stream = try await engine.generate(
            turns: [EngineTurn(role: .user, content: prompt)],
            options: GenerationOptions(
                maxTokens: maxTokens, temperature: 0, thinking: false, tools: tools))
        var text = "", reasoning = ""
        for await event in stream {
            switch event {
            case .text(let chunk): text += chunk
            case .reasoning(let chunk): reasoning += chunk
            case .toolCall(let call):
                print("TOOL CALL \(call.name) \(call.argumentsJSON)")
            case .finished(let stats):
                print("--- reasoning (\(reasoning.count) chars) ---")
                print(String(reasoning.prefix(400)))
                print("--- answer ---")
                print(text)
                // Accepted tokens per round is the number that decides whether a drafter
                // earns its memory, so a measuring run has to print it rather than leave it
                // to be inferred from the rate.
                let accepted = stats.speculative
                    ? String(format: " · %.2f accepted/round", stats.acceptedPerStep) : ""
                print(String(
                    format: "--- %d tokens · %.1f tok/s · prompt %d at %.1f tok/s%@ ---",
                    stats.generatedTokens, stats.tokensPerSecond,
                    stats.promptTokens, stats.promptTokensPerSecond, accepted))
                // What a round is spent on. Speculation here pays only from about 1.9
                // accepted drafts, because the round costs several plain forwards - so
                // the split between drafting and verifying is the number that decides
                // where to spend effort, and it was not observable before.
                if stats.speculative && stats.roundCount > 0 {
                    let round = stats.draftSeconds + stats.verifySeconds
                        + stats.rollbackSeconds
                    let plainStep = stats.plainTokens > 0
                        ? stats.plainSeconds / Double(stats.plainTokens) : 0
                    let inForwards = plainStep > 0
                        ? String(format: ", round %.2f plain forwards",
                                 round / Double(stats.roundCount) / plainStep)
                        : ", no plain steps to compare against"
                    // `DFLASH_PROFILE=1` splits the drafter's own pass. It forces the
                    // graph between phases, so the total inflates - read the shares, not
                    // the seconds.
                    if DFlashDraftModel.profiling {
                        let p = DFlashDraftModel.profile
                        let sum = p.backboneSeconds + p.headSeconds + p.topKSeconds
                            + p.selectorSeconds + p.walkSeconds
                        if sum > 0 {
                            print(String(
                                format: "--- draft pass over %d calls: backbone %.0f%% ·"
                                    + " head %.0f%% · top-k %.0f%% · selector %.0f%% ·"
                                    + " walk %.0f%% (%.1f ms per call) ---",
                                p.calls,
                                100 * p.backboneSeconds / sum, 100 * p.headSeconds / sum,
                                100 * p.topKSeconds / sum, 100 * p.selectorSeconds / sum,
                                100 * p.walkSeconds / sum,
                                1000 * sum / Double(max(p.calls, 1))))
                        }
                    }
                    print(String(
                        format: "--- %d rounds · draft %.2fs (%.0f%%) · verify %.2fs (%.0f%%)"
                            + " · rollback %.2fs (%.0f%%)%@ ---",
                        stats.roundCount,
                        stats.draftSeconds, 100 * stats.draftSeconds / round,
                        stats.verifySeconds, 100 * stats.verifySeconds / round,
                        stats.rollbackSeconds, 100 * stats.rollbackSeconds / round,
                        inForwards))
                }
            }
        }
        exit(0)
    } catch {
        print("generate failed: \(error.localizedDescription)")
        exit(1)
    }
}
