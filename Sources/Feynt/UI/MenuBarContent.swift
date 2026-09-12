import AppKit
import SwiftUI

enum Pasteboard {
    static func copy(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }
}

/// Menu-bar menu: state, speed, model control, log, idle timeout, quit.
struct MenuBarContent: View {
    @ObservedObject var engine: EngineController
    @ObservedObject var settings: AppSettings
    @ObservedObject var api: APIServer
    @Environment(\.openWindow) private var openWindow
    @EnvironmentObject private var state: AppState

    var body: some View {
        Text("\(settings.selectedModel.title) — \(engine.statusText)")

        Text(engine.speculative ? "Mode: speculative decoding" : "Mode: plain decoding")

        if engine.stats.tokensPerSecond > 0 {
            Text(String(format: "Speed: %.1f tok/s", engine.stats.tokensPerSecond))
            Text(String(format: "Accepted per round: %.2f", engine.stats.acceptedPerStep))
        }

        Divider()

        Text(api.isRunning ? "Port \(settings.port)" : "Port \(settings.port) — server stopped")

        Button("Copy URL") { Pasteboard.copy(api.baseURL) }

        let installed = ModelCatalog.all.filter { ModelResolver.isPresent($0) }
        let missing = ModelCatalog.all.filter { !ModelResolver.isPresent($0) }
        let downloading = state.downloader.isDownloading

        Menu("Copy model path") {
            ForEach(installed) { spec in
                if let path = ModelResolver.installedLocation(for: spec)?.path {
                    Button(spec.title) { Pasteboard.copy(path) }
                }
            }
        }
        .disabled(installed.isEmpty)

        Divider()

        if downloading {
            Text("Downloading — \(state.downloader.detail)")
        }

        // The menu bar is where the model gets picked: it is the surface that is always
        // there. Load lists only what is on disk and Download only what is not, so each
        // item says exactly what the click will cost - a switch, or gigabytes. Download
        // stops at the download; the model is loaded when it is picked under Load.
        // Re-download does not belong here; it lives in the model window next to the
        // path and size it acts on.
        Menu("Load model") {
            ForEach(installed) { spec in
                let loaded =
                    settings.selectedModelID == spec.id
                    && (engine.state == .ready || engine.state.isBusy)
                Button {
                    state.switchModel(to: spec)
                } label: {
                    if loaded {
                        Label(spec.title, systemImage: "checkmark")
                    } else {
                        Text(spec.title)
                    }
                }
                .disabled(loaded || engine.state.isBusy || downloading)
            }
        }
        .disabled(installed.isEmpty)

        if !missing.isEmpty {
            Menu("Download model") {
                ForEach(missing) { spec in
                    let artifacts = state.missingArtifacts(for: spec)
                    let bytes = artifacts.reduce(Int64(0)) { $0 + $1.approximateBytes }
                    Button("\(spec.title) — \(Paths.formatBytes(bytes))") {
                        state.downloader.download(artifacts) { _ in }
                    }
                    .disabled(downloading)
                }
            }
        }

        if engine.state == .ready || engine.state.isBusy {
            Button("Unload model") {
                Task { await engine.unload() }
            }
            .disabled(engine.state == .generating)
        }

        Button("Open chat") {
            NSApp.activate(ignoringOtherApps: true)
            openWindow(id: MainWindowID.value)
        }

        Button("Open log") {
            Paths.ensureDirectory(Paths.logDirectory)
            NSWorkspace.shared.open(Paths.logFile)
        }

        Menu("Unload when idle") {
            ForEach(AppSettings.idleChoices, id: \.self) { value in
                Button {
                    settings.idleTimeout = value
                } label: {
                    if settings.idleTimeout == value {
                        Label(AppSettings.idleLabel(value), systemImage: "checkmark")
                    } else {
                        Text(AppSettings.idleLabel(value))
                    }
                }
            }
        }

        Divider()

        Button("Quit") {
            NSApp.terminate(nil)
        }
        .keyboardShortcut("q")
    }
}

enum MainWindowID {
    static let value = "feynt.main"
}
