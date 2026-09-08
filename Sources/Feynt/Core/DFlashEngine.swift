import DFlashKit
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// In-process engine with DFlash 2 speculation.
///
/// The drafter proposes a whole block per forward and the target verifies it in a single
/// pass, which lands around 4.7 accepted tokens per round on Qwen3.8-27B where the MTP path
/// this replaced managed roughly half that.
///
/// The target is loaded once and shared with an ``MLXEngine`` instance: that engine serves
/// every request the greedy DFlash loop cannot (see ``generate(turns:options:)``) without a
/// second copy of 16 GB of weights becoming resident.
actor DFlashEngine: InferenceEngine {
    private let fallback: MLXEngine
    private var context: ModelContext?
    private var generator: DFlashSpeculativeGenerator?
    /// EOS ids for the loaded target, resolved once at load time.
    private var stopTokens: Set<Int> = []


    init(fallback: MLXEngine = MLXEngine()) {
        self.fallback = fallback
    }

    var isLoaded: Bool { context != nil }
    /// A drafter can be present but unusable, so this reports whether the loop is really wired.
    var isSpeculative: Bool { generator != nil }

    // MARK: - Loading

    func load(modelDirectory: URL, drafterDirectory: URL?) async throws {
        guard ModelResolver.isInstalled(modelDirectory) else {
            throw EngineError.modelMissing(modelDirectory.path)
        }
        await unload()

        let loaded: ModelContext
        do {
            loaded = try await loadModel(from: modelDirectory, using: #huggingFaceTokenizerLoader())
        } catch {
            throw EngineError.loadFailed(error.localizedDescription)
        }
        AppLog.write("loaded target \(modelDirectory.lastPathComponent)")

        context = loaded
        stopTokens = Self.stopTokens(for: loaded)
        await fallback.adopt(context: loaded)
        generator = Self.makeGenerator(context: loaded, drafterDirectory: drafterDirectory)
    }

    /// Never fatal. Speculation is a speed feature, so an unsupported target or a broken
    /// drafter costs tokens per second and nothing else — the model still answers.
    private static func makeGenerator(
        context: ModelContext, drafterDirectory: URL?
    ) -> DFlashSpeculativeGenerator? {
        guard let drafterDirectory else {
            AppLog.write("drafter skipped: none configured; speculation off")
            return nil
        }
        guard ModelResolver.isInstalled(drafterDirectory) else {
            AppLog.write("drafter skipped: no weights at \(drafterDirectory.path)")
            return nil
        }
        // MoE targets are Qwen35Model subclasses, so they resolve through `languageModel`
        // rather than a direct cast.
        guard
            let target = context.model as? Qwen35TextModel
                ?? (context.model as? Qwen35Model)?.languageModel
        else {
            AppLog.write("target \(type(of: context.model)) is not Qwen3.5-family; speculation off")
            return nil
        }
        do {
            let drafter = try DFlashDraftModel.load(directory: drafterDirectory)
            AppLog.write(
                "loaded drafter \(drafterDirectory.lastPathComponent), "
                    + "block \(drafter.configuration.blockSize)")
            return DFlashSpeculativeGenerator(target: target, drafter: drafter)
        } catch {
            AppLog.write("drafter unusable (\(error.localizedDescription)); speculation off")
            return nil
        }
    }

    /// Every source the MLX loop consults, so a stop the fallback honours is honoured here too.
    private static func stopTokens(for context: ModelContext) -> Set<Int> {
        var ids = context.configuration.eosTokenIds
        if let eos = context.tokenizer.eosTokenId { ids.insert(eos) }
        for token in context.configuration.extraEOSTokens {
            if let id = context.tokenizer.convertTokenToId(token) { ids.insert(id) }
        }
        return ids
    }

    // MARK: - Unloading

    func unload() async {
        guard context != nil || generator != nil else { return }
        context = nil
        generator = nil
        stopTokens = []
        // The fallback holds the same context; both references have to go before the buffers do.
        await fallback.unload()
        MLX.Memory.clearCache()
        AppLog.write("unloaded model, GPU cache cleared")
    }

    // MARK: - Generation

    func generate(
        turns: [EngineTurn], options: GenerationOptions
    ) async throws -> AsyncStream<EngineEvent> {
        guard let context else { throw EngineError.notLoaded }

        // The DFlash loop accepts a draft when it matches the target's argmax; speculative
        // sampling is not implemented yet, so anything but greedy has to take the MLX path or
        // the requested temperature would be silently ignored.
        guard let generator, options.temperature <= 0 else {
            return try await fallback.generate(turns: turns, options: options)
        }

        let messages = turns.map { turn -> Chat.Message in
            switch turn.role {
            case .system: return .system(turn.content)
            case .user: return .user(turn.content)
            case .assistant: return .assistant(turn.content)
            }
        }
        // Qwen's chat template reads `enable_thinking`; off by default because with thinking
        // on the model can spend the whole budget reasoning and return an empty answer.
        let userInput = UserInput(
            chat: messages, additionalContext: ["enable_thinking": options.thinking])
        let input = try await context.processor.prepare(input: userInput)
        let prompt = input.text.tokens.asArray(Int.self)

        let stops = stopTokens
        let tokenizer = context.tokenizer
        let raw = generator.stream(
            prompt: prompt, maximumTokens: options.maxTokens, stopTokens: stops)

        return AsyncStream<EngineEvent> { continuation in
            let task = Task {
                var splitter = ThinkingSplitter()
                // Incremental decode: re-decoding the whole array per token is quadratic, and
                // a token is often half a UTF-8 character, which a per-token `decode` mangles.
                var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
                for await event in raw {
                    switch event {
                    case .token(let id):
                        // The loop emits the stop token itself; it is a control marker, not text.
                        guard !stops.contains(id) else { continue }
                        detokenizer.append(token: id)
                        if let chunk = detokenizer.next() {
                            for item in splitter.consume(chunk) { continuation.yield(item) }
                        }
                    case .finished(let statistics):
                        for item in splitter.finish() { continuation.yield(item) }
                        continuation.yield(
                            .finished(Self.stats(from: statistics, promptTokens: prompt.count)))
                    case .failed(let reason):
                        AppLog.write("dflash generation failed: \(reason)")
                        for item in splitter.finish() { continuation.yield(item) }
                        continuation.yield(.text("\n[generation failed: \(reason)]"))
                        continuation.yield(.finished(GenerationStats()))
                    }
                }
                for item in splitter.finish() { continuation.yield(item) }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private static func stats(
        from statistics: DFlashGenerationStatistics, promptTokens: Int
    ) -> GenerationStats {
        var stats = GenerationStats()
        stats.generatedTokens = statistics.tokens
        stats.promptTokens = promptTokens
        // Prefill is excluded from tok/s so the number stays comparable with the MLX path's,
        // which reports prompt and decode rates separately for the same reason.
        let decodeSeconds = statistics.seconds - statistics.prefillSeconds
        if decodeSeconds > 0 {
            stats.tokensPerSecond = Double(statistics.tokens) / decodeSeconds
        }
        if statistics.prefillSeconds > 0 {
            stats.promptTokensPerSecond = Double(promptTokens) / statistics.prefillSeconds
        }
        stats.acceptedPerStep = statistics.meanAcceptedPerRound
        stats.speculative = true
        return stats
    }
}
