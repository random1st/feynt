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
