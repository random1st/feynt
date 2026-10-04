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

    /// Recent prompts kept resident so a conversation is prefilled once rather than once
    /// per turn. Memory spent deliberately: every turn re-sends the whole history, and
    /// re-reading the weights for a prompt the model has already seen is the longest pause
    /// in the app. Four conversations, capped so they cannot crowd out the weights.
    private var prefixCache = PrefixCache(slots: 4, byteLimit: 12 << 30)

    init(fallback: MLXEngine = MLXEngine()) {
        self.fallback = fallback
    }

    var isLoaded: Bool { context != nil }
    /// A drafter can be present but unusable, so this reports whether the loop is really wired.
    var isSpeculative: Bool { generator != nil }

    // MARK: - Loading

    func load(modelDirectory: URL, drafterDirectory: URL?, drafterQuantizationBits: Int?)
        async throws
    {
        guard ModelResolver.isInstalled(modelDirectory) else {
            throw EngineError.modelMissing(modelDirectory.path)
        }
        await unload()

        var loaded: ModelContext
        do {
            loaded = try await loadModel(from: modelDirectory, using: #huggingFaceTokenizerLoader())
        } catch {
            throw EngineError.loadFailed(error.localizedDescription)
        }
        // Same reason as in ``MLXEngine``: a directory load leaves the tool-call format
        // unset, and the fallback engine adopts this very context.
        loaded.configuration.toolCallFormat = ToolBridge.format(
            for: loaded, directory: modelDirectory)
        AppLog.write("loaded target \(modelDirectory.lastPathComponent)"
            + ", tool calls: \(loaded.configuration.toolCallFormat?.rawValue ?? "none")")

        context = loaded
        stopTokens = Self.stopTokens(for: loaded)
        await fallback.adopt(context: loaded)
        // A snapshot is only valid for the model it was taken from.
        prefixCache = PrefixCache(slots: 4, byteLimit: 12 << 30)
        generator = Self.makeGenerator(
            context: loaded, drafterDirectory: drafterDirectory,
            quantizeBits: drafterQuantizationBits, prefixCache: prefixCache)
    }

    /// How many tokens a round may draft. `nil` - the default - lets the generator grow
    /// the width itself and lets its own gate decide whether to draft at all.
    ///
    /// The width that pays depends on how long the context is, and no single number wins.
    /// Measured on Ornith, alternating order, four runs each:
    ///
    ///                      3.2k of context        29-token prompt
    ///     pinned to 7      78-84 tok/s  3.54      136-178 tok/s   6.26 accepted per round
    ///     grown by itself  62-72        3.00      218-262        11.55
    ///     pinned to 15     48-61        2.64
    ///
    /// One tile of the small-M kernel is eight rows, so a block wider than seven pays for
    /// a second read of the weights per round. Over 3.2k tokens of context that second
    /// read costs more than the extra drafts return, and narrowing the block is worth
    /// about 20%. On a short prompt the same narrowing throws away half the acceptance
    /// and costs about 35%. So the decision belongs to the context length, which neither
    /// this cap nor the generator's gate - it watches acceptance only - currently reads.
    /// Until that lands upstream the product keeps the generator's own behaviour.
    ///
    /// `FEYNT_DRAFT_CAP` pins a width for measuring. It exists because the obvious way to
    /// get a plain baseline - remove the drafter - swaps this loop for the library's and
    /// compares two implementations rather than two widths.
    private static var draftCap: Int? {
        ProcessInfo.processInfo.environment["FEYNT_DRAFT_CAP"].flatMap(Int.init)
    }

    /// Spends a round's verified rows on a tree of candidates instead of one chain.
    /// Off in the library and off here; `FEYNT_TREE=1` turns it on for a measuring run.
    /// It is a knob rather than a setting for the same reason the cap above is: the shape
    /// that pays depends on the workload, and nothing has measured which way that goes on
    /// this model yet.
    private static var treeSpeculation: Bool {
        ProcessInfo.processInfo.environment["FEYNT_TREE"] == "1"
    }

    /// Quantises the drafter's linear layers after loading. A measuring knob, like the two
    /// above, and the one that decides an open question: on an A3B MoE the target reads
    /// only ~2 GB of its 4-bit weights per token, so a bf16 drafter of 0.8-1.0 GB is
    /// roughly half of what it is drafting for. On the dense 27B the same drafter is a
    /// quarter. If that ratio is what sinks the better drafter here, four bits should
    /// show it: the selector is left alone either way, the loader only takes linears.
    private static func drafterBits(_ fromCatalog: Int?) -> Int? {
        ProcessInfo.processInfo.environment["FEYNT_DRAFTER_BITS"].flatMap(Int.init) ?? fromCatalog
    }

    /// Never fatal. Speculation is a speed feature, so an unsupported target or a broken
    /// drafter costs tokens per second and nothing else — the model still answers.
    private static func makeGenerator(
        context: ModelContext, drafterDirectory: URL?, quantizeBits: Int?,
        prefixCache: PrefixCache
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
            let bits = Self.drafterBits(quantizeBits)
            let drafter = try DFlashDraftModel.load(
                directory: drafterDirectory, quantizeBits: bits)
            let generator = DFlashSpeculativeGenerator(
                target: target, drafter: drafter, maximumDraftTokens: Self.draftCap,
                prefixCache: prefixCache, treeSpeculation: Self.treeSpeculation)
            // The round's shape decides how many target forwards a reply costs, so it
            // belongs in the log next to the block width rather than being inferred
            // from the tokens per second afterwards.
            AppLog.write(
                "loaded drafter \(drafterDirectory.lastPathComponent), "
                    + "block \(drafter.configuration.blockSize), "
                    + "cap \(generator.cap)\(generator.adaptiveWidth ? " adaptive" : " pinned"), "
                    + (bits.map { "drafter \($0)-bit, " } ?? "")
                    + "round \(generator.treeSpeculation ? "tree" : "chain")")
            return generator
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
        // The snapshots hold GPU buffers of their own; leaving them behind would defeat the
        // point of unloading.
        prefixCache.clear()
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

        let messages = ToolBridge.messages(from: turns)
        // Qwen's chat template reads `enable_thinking`; off by default because with thinking
        // on the model can spend the whole budget reasoning and return an empty answer.
        let userInput = UserInput(
            chat: messages, tools: options.tools,
            additionalContext: ["enable_thinking": options.thinking])
        let input = try await context.processor.prepare(input: userInput)
        let prompt = input.text.tokens.asArray(Int.self)

        let stops = stopTokens
        let tokenizer = context.tokenizer
        // The loop yields raw token ids, so the tool-call syntax arrives here as ordinary
        // text and has to be pulled back out - the same processor the MLX path gets from the
        // library, in the dialect resolved at load time. Without it a request carrying tools
        // fell through to plain decoding, which cost this path the prefix cache on exactly
        // the requests an agent makes: the same 3187-token conversation re-sent took 2.96s
        // every single time, against 5.26s cold and 1.00s warm now.
        //
        // The trade is real and worth naming. Speculation on a tool-carrying request lands
        // around 1.6 accepted per round on prose, where a block of 16 does not pay for
        // itself: sustained decode falls from 64-68 to 47-49 tok/s. Prefill dominates an
        // agent's turn - a call is thirty tokens against three thousand of context - so
        // skipping it wins up to roughly 500 generated tokens, and past that the two paths
        // draw level. If a workload ever lives out there, the gate belongs on measured
        // acceptance rather than on the presence of tools.
        let toolFormat = context.configuration.toolCallFormat ?? .json
        let tools = options.tools
        let raw = generator.stream(
            prompt: prompt, maximumTokens: options.maxTokens, stopTokens: stops)

        return AsyncStream<EngineEvent> { continuation in
            let task = Task {
                var splitter = ThinkingSplitter()
                // Incremental decode: re-decoding the whole array per token is quadratic, and
                // a token is often half a UTF-8 character, which a per-token `decode` mangles.
                var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
                let toolCalls = ToolCallProcessor(format: toolFormat, tools: tools)

                /// Processor outputs as engine events, in the order the model emitted them:
                /// response text still goes through the reasoning splitter, a parsed call is
                /// handed over as a call, and a tool-call-shaped output that did not parse is
                /// logged rather than leaked - as text it is protocol noise, as a call a lie.
                func events(from outputs: [ToolCallProcessor.Output]) -> [EngineEvent] {
                    var result: [EngineEvent] = []
                    for output in outputs {
                        switch output {
                        case .response(let text):
                            result.append(contentsOf: splitter.consume(text))
                        case .toolCall(let call):
                            result.append(.toolCall(ToolBridge.engineCall(from: call)))
                        case .rejectedToolCall(let rejected):
                            AppLog.write("rejected tool call: \(rejected)")
                        }
                    }
                    return result
                }

                for await event in raw {
                    switch event {
                    case .token(let id):
                        // The loop emits the stop token itself; it is a control marker, not text.
                        guard !stops.contains(id) else { continue }
                        detokenizer.append(token: id)
                        if let chunk = detokenizer.next() {
                            for item in events(from: toolCalls.processChunkOutputs(chunk)) {
                                continuation.yield(item)
                            }
                        }
                    case .finished(let statistics):
                        // A call framed by EOS is only complete once the stream ends, so the
                        // processor is drained before the splitter is.
                        for item in events(from: toolCalls.processEOSOutputs()) {
                            continuation.yield(item)
                        }
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
        stats.cachedPromptTokens = statistics.reusedPromptTokens
        stats.speculative = true
        return stats
    }
}
