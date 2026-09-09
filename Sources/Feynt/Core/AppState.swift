import AppKit
import Foundation
import SwiftUI

/// Single root object wired into every view. A shared instance exists so the app delegate
/// can unload the model on quit without threading references through the scene graph.
@MainActor
final class AppState: ObservableObject {
    static let shared = AppState()

    let settings: AppSettings
    let engine: EngineController
    let downloader: ModelDownloader
    let chat: ChatStore
    let api: APIServer

    private init() {
        let settings = AppSettings()
        // DFlash 2 drives generation; the MLXEngine handed to it is the fallback for the
        // requests its greedy loop does not serve, and it shares the same loaded weights.
        let engine = EngineController(
            engine: DFlashEngine(fallback: MLXEngine()), settings: settings)
        let api = APIServer(engine: engine, settings: settings)
        self.settings = settings
        self.engine = engine
        self.api = api
        self.downloader = ModelDownloader()
        self.chat = ChatStore(engine: engine, settings: settings)
        engine.lifecycle = api
        // Listening from launch, not from the first load: a client should be able to wake a
        // configured model by asking for a completion, the same way it wakes one after the
        // idle timeout has unloaded it.
        if settings.wizardCompleted {
            api.start()
        }
    }

    /// The two catalog models plus the shared drafter, in the order the wizard downloads them.
    func missingArtifacts(for spec: ModelSpec) -> [ModelSpec] {
        [spec, spec.drafter].filter { !ModelResolver.isPresent($0) }
    }

    /// Switching stops the current model first, downloads what is missing, then loads the
    /// other one. Lives here because two surfaces offer it - the chat header and the model
    /// window - and a second copy would be a second set of rules about what "switch" means.
    func switchModel(to spec: ModelSpec) {
        let missing = missingArtifacts(for: spec)
        settings.selectedModelID = spec.id
        guard !missing.isEmpty else {
            Task { await engine.switchTo(spec) }
            return
        }
        downloader.download(missing) { success in
            guard success else { return }
            Task { await self.engine.switchTo(spec) }
        }
    }

    func finishWizard(with spec: ModelSpec) {
        settings.selectedModelID = spec.id
        settings.wizardCompleted = true
        // Launch raised the app to a regular one so the wizard could take focus; drop back
        // so the Dock icon does not outlive the setup it existed for.
        NSApp.setActivationPolicy(.accessory)
    }
}
