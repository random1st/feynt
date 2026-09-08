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
    }

    /// The two catalog models plus the shared drafter, in the order the wizard downloads them.
    func missingArtifacts(for spec: ModelSpec) -> [ModelSpec] {
        [spec, spec.drafter].filter { !ModelResolver.isPresent($0) }
    }

    func finishWizard(with spec: ModelSpec) {
        settings.selectedModelID = spec.id
        settings.wizardCompleted = true
        // Launch raised the app to a regular one so the wizard could take focus; drop back
        // so the Dock icon does not outlive the setup it existed for.
        NSApp.setActivationPolicy(.accessory)
    }
}
