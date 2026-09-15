import Foundation

/// One turn of the conversation as the engine sees it.
///
/// An agent loop needs two turns a plain chat does not: the assistant turn that asked for
/// tools, and the `tool` turn carrying what running them returned. Both have to go back
/// into the prompt verbatim, or the model re-asks for a call it already made.
struct EngineTurn: Sendable, Equatable {
    enum Role: String, Sendable { case system, user, assistant, tool }
    let role: Role
    let content: String
    /// Calls this assistant turn asked for, as `(name, JSON arguments)`; empty otherwise.
    var toolCalls: [EngineToolCall] = []
    /// For a `.tool` turn: the call it answers, and the tool's name.
    var toolCallID: String? = nil
    var toolName: String? = nil
}

/// A tool call in the shape both the API and the chat template want. Arguments stay as the
/// raw JSON text the model produced: re-encoding them through a Swift dictionary loses key
/// order and turns integers into doubles, and clients compare these strings.
struct EngineToolCall: Sendable, Equatable {
    let id: String
    let name: String
    let argumentsJSON: String
}

/// Streamed output. Reasoning is kept separate from the answer so the UI can dim and
/// collapse it instead of mixing it into the text.
enum EngineEvent: Sendable {
    case text(String)
    case reasoning(String)
    /// A tool call the model asked for. Emitted instead of the text that encoded it, so a
    /// client never sees `<|tool_call_start|>` leak into the answer.
    case toolCall(EngineToolCall)
    case finished(GenerationStats)
}

/// Counters produced by the generation loop itself — there is no metrics endpoint any more.
struct GenerationStats: Sendable, Equatable {
    var tokensPerSecond: Double = 0
    var promptTokensPerSecond: Double = 0
    var generatedTokens: Int = 0
    var promptTokens: Int = 0
    /// Mean draft tokens accepted per speculative round; 0 when speculation is off.
    var acceptedPerStep: Double = 0
    /// Prompt tokens served from the prefix cache instead of being prefilled.
    var cachedPromptTokens: Int = 0
    var speculative: Bool = false
}

/// Per-request knobs. A struct rather than loose parameters so the API server can honour
/// OpenAI fields without widening the protocol every time a client sends a new one.
struct GenerationOptions: Sendable {
    var maxTokens: Int = 4096
    /// Greedy by default. The DFlash loop only verifies greedily, so any other default
    /// silently routed every request that did not name a temperature onto the slow path —
    /// speculation was never engaged and the menu bar still claimed it was.
    var temperature: Float = 0
    var thinking: Bool = false
    /// OpenAI-shaped tool definitions, passed straight to the chat template. The model has
    /// to be told what exists before it can ask for it, and every model spells that
    /// differently — the template owns the spelling, not this app.
    var tools: [[String: any Sendable]]? = nil
}

enum EngineError: LocalizedError {
    case notLoaded
    case modelMissing(String)
    case loadFailed(String)

    var errorDescription: String? {
        switch self {
        case .notLoaded:
            return "No model is loaded."
        case .modelMissing(let path):
            return "Model directory not found: \(path)"
        case .loadFailed(let reason):
            return "Could not load the model: \(reason)"
        }
    }
}

/// The seam between the UI and whatever runs the weights.
///
/// v1 is ``MLXEngine`` (MLX Swift in-process, MTP speculative decoding). A faster
/// block-diffusion drafter can replace the implementation behind this protocol without the
/// wizard, menu bar or chat knowing about it.
protocol InferenceEngine: AnyObject, Sendable {
    /// Load weights. `drafterDirectory` is optional: without a usable drafter the engine
    /// still generates, just without speculation.
    func load(modelDirectory: URL, drafterDirectory: URL?) async throws

    /// Release the weights and hand the memory back to the OS.
    func unload() async

    var isLoaded: Bool { get async }

    /// Whether the loaded pair actually speculates (a drafter can be present but unusable).
    var isSpeculative: Bool { get async }

    func generate(
        turns: [EngineTurn], options: GenerationOptions
    ) async throws -> AsyncStream<EngineEvent>
}
