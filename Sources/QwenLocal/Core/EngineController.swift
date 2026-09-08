import Combine
import Foundation
import SwiftUI

/// Implemented by the API server so the endpoint tracks model residency.
@MainActor
protocol EngineLifecycleObserver: AnyObject {
    func engineDidLoad()
    func engineDidUnload()
}

/// Owns the engine lifecycle and everything the UI observes about it.
@MainActor
final class EngineController: ObservableObject {
    enum State: Equatable {
        case unloaded
        case loading
        case ready
        case generating
        case failed(String)

        var isBusy: Bool { self == .loading || self == .generating }
    }

    @Published private(set) var state: State = .unloaded
    @Published private(set) var activeModel: ModelSpec?
    @Published private(set) var stats = GenerationStats()
    /// Live tok/s during a stream; falls back to the last completed run's rate.
    @Published private(set) var liveTokensPerSecond: Double = 0
    @Published private(set) var speculative = false

    private let engine: any InferenceEngine
    private let settings: AppSettings
    private var idleTimer: Timer?
    private var lastActivity = Date()

    /// The API endpoint follows the model: it comes up when weights are resident and goes
    /// down when they are not, so a client never talks to an engine that cannot answer.
    weak var lifecycle: EngineLifecycleObserver?

    init(engine: any InferenceEngine, settings: AppSettings) {
        self.engine = engine
        self.settings = settings
        startIdleTimer()
    }

    // MARK: - Lifecycle

    func load(_ spec: ModelSpec) async {
        guard let modelDirectory = ModelResolver.installedLocation(for: spec) else {
            state = .failed("Модель \(spec.title) не найдена на диске")
            return
        }
        state = .loading
        activeModel = spec
        let drafterDirectory = ModelResolver.installedLocation(for: spec.drafter)
        do {
            try await engine.load(modelDirectory: modelDirectory, drafterDirectory: drafterDirectory)
            speculative = await engine.isSpeculative
            state = .ready
            touch()
            lifecycle?.engineDidLoad()
        } catch {
            state = .failed(error.localizedDescription)
            activeModel = nil
        }
    }

    func unload() async {
        await engine.unload()
        speculative = false
        liveTokensPerSecond = 0
        if case .failed = state {} else { state = .unloaded }
        lifecycle?.engineDidUnload()
    }

    /// Switch models: the old weights go first so two 16 GB models are never resident at once.
    func switchTo(_ spec: ModelSpec) async {
        await unload()
        await load(spec)
    }

    /// Load on demand — the chat calls this so the first message just works.
    func ensureLoaded() async -> Bool {
        if state == .ready || state == .generating { return true }
        await load(settings.selectedModel)
        return state == .ready
    }

    // MARK: - Generation

    func generate(
        turns: [EngineTurn], options: GenerationOptions
    ) async throws -> AsyncStream<EngineEvent> {
        touch()
        state = .generating
        return try await engine.generate(turns: turns, options: options)
    }

    /// Options for a UI-initiated turn; the API server builds its own from the request.
    ///
    /// Greedy on purpose: the DFlash loop accepts a draft only when it matches the target's
    /// argmax, so it has no speculative sampler yet and any temperature above zero would
    /// silently fall back to plain decoding — the chat would show "DFlash-спекуляция" while
    /// running at half the speed. Raise this once sampling lands in the drafter.
    func uiOptions(thinking: Bool) -> GenerationOptions {
        GenerationOptions(maxTokens: settings.maxTokens, temperature: 0, thinking: thinking)
    }

    func generationFinished(_ stats: GenerationStats?) {
        if let stats {
            self.stats = stats
            liveTokensPerSecond = stats.tokensPerSecond
        }
        if state == .generating { state = .ready }
        touch()
    }

    func reportLiveRate(_ rate: Double) {
        liveTokensPerSecond = rate
    }

    // MARK: - Idle unload

    /// In-process we free the ~16 GB by dropping the model references and clearing the MLX
    /// cache — no process exit involved. The timer only counts quiet time: any load or
    /// generation activity resets it, so a long answer can never look idle.
    private func startIdleTimer() {
        idleTimer = Timer.scheduledTimer(withTimeInterval: 15, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.checkIdle() }
        }
    }

    private func checkIdle() {
        let timeout = settings.idleTimeout
        guard timeout != AppSettings.idleNever else { return }
        guard state == .ready else { return }
        guard Date().timeIntervalSince(lastActivity) >= Double(timeout) else { return }
        AppLog.write("idle \(timeout)s — unloading")
        Task { await unload() }
    }

    private func touch() {
        lastActivity = Date()
    }

    // MARK: - Presentation helpers

    var statusIcon: String {
        switch state {
        case .unloaded: return "moon.zzz"
        case .loading: return "hourglass"
        case .ready: return "bolt.circle"
        case .generating: return "waveform"
        case .failed: return "exclamationmark.triangle"
        }
    }

    var statusText: String {
        switch state {
        case .unloaded: return "выгружена"
        case .loading: return "загрузка…"
        case .ready: return "готова"
        case .generating: return "генерация"
        case .failed(let message): return "ошибка: \(message)"
        }
    }

    /// Menu-bar title: live speed while the model is resident, nothing when it is not.
    var menuBarTitle: String {
        switch state {
        case .ready, .generating:
            return liveTokensPerSecond > 0 ? String(format: "%.0f tok/s", liveTokensPerSecond) : ""
        default:
            return ""
        }
    }
}
