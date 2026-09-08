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

        Text(engine.speculative ? "Режим: MTP-спекуляция" : "Режим: обычная генерация")

        if engine.stats.tokensPerSecond > 0 {
            Text(String(format: "Скорость: %.1f tok/s", engine.stats.tokensPerSecond))
            Text(String(format: "Принято за шаг: %.2f", engine.stats.acceptedPerStep))
        }

        Divider()

        Text(api.isRunning ? "Порт \(settings.port)" : "Порт \(settings.port) — сервер остановлен")

        Button("Копировать URL") { Pasteboard.copy(api.baseURL) }

        if let path = ModelResolver.installedLocation(for: settings.selectedModel)?.path {
            Button("Копировать путь модели") { Pasteboard.copy(path) }
        }

        Divider()

        if engine.state == .unloaded {
            Button("Загрузить модель") {
                Task { await engine.load(settings.selectedModel) }
            }
        } else {
            Button("Выгрузить модель") {
                Task { await engine.unload() }
            }
            .disabled(engine.state == .generating)
        }

        Button("Открыть чат") {
            NSApp.activate(ignoringOtherApps: true)
            openWindow(id: MainWindowID.value)
        }

        Button("Открыть лог") {
            Paths.ensureDirectory(Paths.logDirectory)
            NSWorkspace.shared.open(Paths.logFile)
        }

        Menu("Выгружать после простоя") {
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

        Button("Выйти") {
            NSApp.terminate(nil)
        }
        .keyboardShortcut("q")
    }
}

enum MainWindowID {
    static let value = "qwenlocal.main"
}
