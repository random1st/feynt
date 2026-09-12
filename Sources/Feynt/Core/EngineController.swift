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

    /// State, speed and mode describe the active model only; the others sit resident and
    /// silent until they are picked.
    @Published private(set) var state: State = .unloaded
    @Published private(set) var activeModel: ModelSpec?
    @Published private(set) var stats = GenerationStats()
    /// Live tok/s during a stream; falls back to the last completed run's rate.
    @Published private(set) var liveTokensPerSecond: Double = 0
    @Published private(set) var speculative = false
    /// Everything resident, in load order.
    @Published private(set) var loadedModels: [ModelSpec] = []

    /// One engine per resident model. Each holds its own weights, drafter and prefix cache,
    /// so a switch touches nothing the other model is using.
    private struct Resident {
        let spec: ModelSpec
        let engine: any InferenceEngine
        let speculative: Bool
        var lastActivity: Date
    }

    private var residents: [String: Resident] = [:]
    /// Loads in flight. The chat's first message and an API request can ask for the same
    /// model within the same second; without this each would build its own engine, the
    /// second would replace the first in the pool, and 20 GB would sit resident with
    /// nobody holding a reference to unload it.
    private var loads: [String: Task<Void, Never>] = [:]
    private let makeEngine: () -> any InferenceEngine
    private let settings: AppSettings
    private var idleTimer: Timer?

    /// The API endpoint follows the model: it comes up when weights are resident and goes
    /// down when they are not, so a client never talks to an engine that cannot answer.
    weak var lifecycle: EngineLifecycleObserver?

    init(makeEngine: @escaping () -> any InferenceEngine, settings: AppSettings) {
        self.makeEngine = makeEngine
        self.settings = settings
        startIdleTimer()
    }

    // MARK: - Lifecycle

    /// Loads a model or, when it is already resident, just makes it the active one. The
    /// selection in settings follows from here so every surface that switches - chat, model
    /// window, menu bar, API - shares one rule about what "switch" means.
    func load(_ spec: ModelSpec) async {
        if residents[spec.id] != nil {
            activate(spec)
            return
        }
        if let pending = loads[spec.id] {
            await pending.value
            if residents[spec.id] != nil { activate(spec) }
            return
        }
        let task = Task { await performLoad(spec) }
        loads[spec.id] = task
        await task.value
        loads[spec.id] = nil
    }

    private func performLoad(_ spec: ModelSpec) async {
        guard let modelDirectory = ModelResolver.installedLocation(for: spec) else {
            state = .failed("\(spec.title) is not on disk")
            return
        }
        state = .loading
        activeModel = spec
        settings.selectedModelID = spec.id
        let drafterDirectory = ModelResolver.installedLocation(for: spec.drafter)
        let engine = makeEngine()
        do {
            try await engine.load(modelDirectory: modelDirectory, drafterDirectory: drafterDirectory)
            let isSpeculative = await engine.isSpeculative
            residents[spec.id] = Resident(
                spec: spec, engine: engine, speculative: isSpeculative, lastActivity: Date())
            loadedModels.append(spec)
            speculative = isSpeculative
            state = .ready
            touch(spec)
            lifecycle?.engineDidLoad()
        } catch {
            state = .failed(error.localizedDescription)
            activeModel = nil
        }
    }

    private func activate(_ spec: ModelSpec) {
        guard let resident = residents[spec.id] else { return }
        activeModel = spec
        settings.selectedModelID = spec.id
        speculative = resident.speculative
        state = .ready
        stats = GenerationStats()
        liveTokensPerSecond = 0
        touch(spec)
    }

    func unload(_ spec: ModelSpec) async {
        guard let resident = residents.removeValue(forKey: spec.id) else { return }
        await resident.engine.unload()
        loadedModels.removeAll { $0.id == spec.id }
        if activeModel?.id == spec.id {
            speculative = false
            liveTokensPerSecond = 0
            if case .failed = state {} else { state = .unloaded }
        }
        if residents.isEmpty {
            lifecycle?.engineDidUnload()
        }
    }

    /// Unloads the active model; the others stay.
    func unload() async {
        guard let active = activeModel else { return }
        await unload(active)
    }

    func unloadAll() async {
        for spec in loadedModels {
            await unload(spec)
        }
    }

    /// Switching used to unload first so two 16 GB models were never resident at once. The
    /// machine has the memory, and a second resident model turns the switch back into an
    /// instant; what nobody uses the idle timer frees.
    func switchTo(_ spec: ModelSpec) async {
        await load(spec)
    }

    /// Unload-then-load of one model, for the state a message will not fix.
    func reload(_ spec: ModelSpec) async {
        await unload(spec)
        await load(spec)
    }

    /// Load on demand — the chat calls this so the first message just works.
    func ensureLoaded() async -> Bool {
        await ensureLoaded(settings.selectedModel)
    }

    func ensureLoaded(_ spec: ModelSpec) async -> Bool {
        if activeModel?.id == spec.id, state == .ready || state == .generating { return true }
        await load(spec)
        return state == .ready && activeModel?.id == spec.id
    }

    // MARK: - Generation

    func generate(
        turns: [EngineTurn], options: GenerationOptions
    ) async throws -> AsyncStream<EngineEvent> {
        guard let active = activeModel, let resident = residents[active.id] else {
            throw EngineError.notLoaded
        }
        touch(active)
        state = .generating
        return try await resident.engine.generate(turns: turns, options: options)
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
        if let active = activeModel { touch(active) }
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
        let now = Date()
        for resident in residents.values {
            guard now.timeIntervalSince(resident.lastActivity) >= Double(timeout) else { continue }
            // The active model is touched at every generation, so a stale timestamp there
            // is only possible mid-work; leave it alone until it goes quiet.
            if resident.spec.id == activeModel?.id, state.isBusy { continue }
            AppLog.write("idle \(timeout)s — unloading \(resident.spec.title)")
            Task { await unload(resident.spec) }
        }
    }

    private func touch(_ spec: ModelSpec) {
        residents[spec.id]?.lastActivity = Date()
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
        case .unloaded: return "unloaded"
        case .loading: return "loading…"
        case .ready: return "ready"
        case .generating: return "generating"
        case .failed(let message): return "error: \(message)"
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
