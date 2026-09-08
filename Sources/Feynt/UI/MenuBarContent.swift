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

    var body: some View {
        Text("\(settings.selectedModel.title) — \(engine.statusText)")

        Text(engine.speculative ? "Mode: DFlash speculation" : "Mode: plain decoding")

        if engine.stats.tokensPerSecond > 0 {
            Text(String(format: "Speed: %.1f tok/s", engine.stats.tokensPerSecond))
            Text(String(format: "Accepted per round: %.2f", engine.stats.acceptedPerStep))
        }

        Divider()

        Text(api.isRunning ? "Port \(settings.port)" : "Port \(settings.port) — server stopped")

        Button("Copy URL") { Pasteboard.copy(api.baseURL) }

        if let path = ModelResolver.installedLocation(for: settings.selectedModel)?.path {
            Button("Copy model path") { Pasteboard.copy(path) }
        }

        Divider()

        if engine.state == .unloaded {
            Button("Load model") {
                Task { await engine.load(settings.selectedModel) }
            }
        } else {
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
