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

    do {
        let stream = try await engine.generate(
            turns: [EngineTurn(role: .user, content: prompt)],
            options: GenerationOptions(
                maxTokens: 200, temperature: 0, thinking: false, tools: tools))
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
                print(String(
                    format: "--- %d tokens · %.1f tok/s · prompt %d at %.1f tok/s ---",
                    stats.generatedTokens, stats.tokensPerSecond,
                    stats.promptTokens, stats.promptTokensPerSecond))
            }
        }
        exit(0)
    } catch {
        print("generate failed: \(error.localizedDescription)")
        exit(1)
    }
}
