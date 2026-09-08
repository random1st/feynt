import AppKit
import SwiftUI

@main
struct QwenLocalApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var state = AppState.shared

    var body: some Scene {
        Window("QwenLocal", id: MainWindowID.value) {
            RootView(settings: state.settings)
                .environmentObject(state)
        }
        .defaultSize(width: 780, height: 560)

        MenuBarExtra {
            MenuBarContent(engine: state.engine, settings: state.settings, api: state.api)
                .environmentObject(state)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: state.engine.statusIcon)
                if !state.engine.menuBarTitle.isEmpty {
                    Text(state.engine.menuBarTitle)
                }
            }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        Paths.ensureDirectory(Paths.logDirectory)
        Paths.ensureDirectory(Paths.modelsRoot)
        AppLog.write("app launched")
    }

    /// Quitting exits the process, which returns the weights' memory to the OS; the explicit
    /// unload is belt and braces so the log records a clean shutdown.
    func applicationWillTerminate(_ notification: Notification) {
        AppLog.write("app terminating")
        Task { @MainActor in await AppState.shared.engine.unload() }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }
}
