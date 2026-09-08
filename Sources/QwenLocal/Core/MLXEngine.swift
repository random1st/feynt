import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// In-process engine: MLX Swift runs the weights inside this app, so there is no server,
/// no port and no child process to supervise.
///
/// Speculative decoding uses the MTP drafter path that mlx-swift-lm ships for Qwen3.5-class
/// models. When no usable drafter is present the engine degrades to ordinary single-token
/// generation instead of refusing to run.
actor MLXEngine: InferenceEngine {
    private var context: ModelContext?
    private var drafter: (any MTPDrafterModel)?
    private var registeredDrafterTypes = false

    /// `blockSize - 1` tokens are drafted per round plus the bonus token from the previous
    /// verify; 4 is the value mlx-vlm's example configs use.
    private let draftBlockSize = 4

    var isLoaded: Bool { context != nil }
    var isSpeculative: Bool { drafter != nil }

    // MARK: - Loading

    func load(modelDirectory: URL, drafterDirectory: URL?) async throws {
        guard ModelResolver.isInstalled(modelDirectory) else {
            throw EngineError.modelMissing(modelDirectory.path)
        }
        await unload()

        let tokenizerLoader = #huggingFaceTokenizerLoader()
        do {
            context = try await loadModel(from: modelDirectory, using: tokenizerLoader)
        } catch {
            throw EngineError.loadFailed(error.localizedDescription)
        }
        AppLog.write("loaded target \(modelDirectory.lastPathComponent)")

        if let drafterDirectory {
            drafter = await loadDrafter(directory: drafterDirectory, tokenizerLoader: tokenizerLoader)
        }
    }

    /// A drafter failure is never fatal: speculation is a speed optimisation, so we log the
    /// reason and continue without it.
    private func loadDrafter(
        directory: URL, tokenizerLoader: any TokenizerLoader
    ) async -> (any MTPDrafterModel)? {
        guard ModelResolver.isInstalled(directory) else {
            AppLog.write("drafter skipped: no weights at \(directory.path)")
            return nil
        }
        if !registeredDrafterTypes {
            await Qwen35TextMTPRegistration.register()
            registeredDrafterTypes = true
        }
        do {
            let drafterContext = try await MTPDrafterModelFactory.shared.load(
                from: directory, using: tokenizerLoader)
            AppLog.write("loaded drafter \(directory.lastPathComponent)")
            return drafterContext.model
        } catch {
            AppLog.write("drafter unsupported (\(error.localizedDescription)); speculation off")
            return nil
        }
    }

    // MARK: - Unloading

    /// Drop every reference and return the buffers to the OS. In-process we control this
    /// directly — no process exit needed.
    func unload() async {
        guard context != nil || drafter != nil else { return }
        context = nil
        drafter = nil
        MLX.Memory.clearCache()
        AppLog.write("unloaded model, GPU cache cleared")
    }

    // MARK: - Generation

    func generate(
        turns: [EngineTurn], options: GenerationOptions
    ) async throws -> AsyncStream<EngineEvent> {
        guard let context else { throw EngineError.notLoaded }

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
        let parameters = GenerateParameters(
            maxTokens: options.maxTokens, temperature: options.temperature, topP: 0.8)

        let raw: AsyncStream<Generation>
        if let drafter {
            raw = try MLXLMCommon.generate(
                input: input, parameters: parameters, context: context,
                mtpDrafter: drafter, blockSize: draftBlockSize)
        } else {
            raw = try MLXLMCommon.generate(
                input: input, parameters: parameters, context: context)
        }

        let speculative = drafter != nil
        return AsyncStream<EngineEvent> { continuation in
            let task = Task {
                var splitter = ThinkingSplitter()
                for await item in raw {
                    switch item {
                    case .chunk(let text):
                        for event in splitter.consume(text) { continuation.yield(event) }
                    case .info(let info):
                        for event in splitter.finish() { continuation.yield(event) }
                        continuation.yield(
                            .finished(Self.stats(from: info, speculative: speculative)))
                    case .toolCall, .rejectedToolCall:
                        break
                    }
                }
                for event in splitter.finish() { continuation.yield(event) }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private static func stats(
        from info: GenerateCompletionInfo, speculative: Bool
    ) -> GenerationStats {
        var stats = GenerationStats()
        stats.generatedTokens = info.generationTokenCount
        stats.promptTokens = info.totalPromptTokenCount
        stats.tokensPerSecond = info.generateTime > 0 ? info.tokensPerSecond : 0
        stats.promptTokensPerSecond = info.promptTime > 0 ? info.promptTokensPerSecond : 0
        stats.speculative = speculative
        if let telemetry = info.speculativeDecodingTelemetry {
            stats.acceptedPerStep = telemetry.meanAcceptedDraftTokensPerRound
        }
        return stats
    }
}
