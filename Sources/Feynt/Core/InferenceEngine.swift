import Foundation

/// One turn of the conversation as the engine sees it.
struct EngineTurn: Sendable, Equatable {
    enum Role: String, Sendable { case system, user, assistant }
    let role: Role
    let content: String
}

/// Streamed output. Reasoning is kept separate from the answer so the UI can dim and
/// collapse it instead of mixing it into the text.
enum EngineEvent: Sendable {
    case text(String)
    case reasoning(String)
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
