import AppKit
import SwiftUI

extension Notification.Name {
    static let feyntShowFirstRunWindow = Notification.Name("feynt.showFirstRunWindow")
}

@main
struct FeyntApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var state = AppState.shared
    /// The app is `LSUIElement`, so its window scene is never shown on its own. Without
    /// opening it here a first-time user launches the app and sees nothing but a menu-bar
    /// glyph, with the setup wizard behind a window nobody asked for.
    @Environment(\.openWindow) private var openWindow


    var body: some Scene {
        Window("Feynt", id: MainWindowID.value) {
            RootView(settings: state.settings)
                .environmentObject(state)
        }
        .defaultSize(width: 780, height: 560)


        MenuBarExtra {
            MenuBarContent(engine: state.engine, settings: state.settings, api: state.api)
                .environmentObject(state)
                .task {
                    // The menu-bar scene is alive from launch, unlike the window, so this is
                    // where the delegate's first-run signal can be received.
                    for await _ in NotificationCenter.default.notifications(
                        named: .feyntShowFirstRunWindow)
                    {
                        openWindow(id: MainWindowID.value)
                    }
                }
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
        // Without this an uncaught Objective-C exception unwinds silently in a release
        // build: no crash report, no log line, just a window that is gone.
        NSSetUncaughtExceptionHandler { exception in
            AppLog.write("uncaught exception: \(exception.name.rawValue) — \(exception.reason ?? "")")
        }

        guard !AppState.shared.settings.wizardCompleted else { return }
        // A menu-bar-only app cannot show a window or take focus until it is briefly a
        // regular app; it drops back to accessory once setup finishes, so the Dock icon
        // does not outlive the wizard.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        NotificationCenter.default.post(name: .feyntShowFirstRunWindow, object: nil)
    }

    /// Quitting exits the process, which returns the weights' memory to the OS; the explicit
    /// unload is belt and braces so the log records a clean shutdown.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        // The app once exited with nothing in the log and no crash report, which left the
        // cause unknowable. Recording who asked settles that next time.
        AppLog.write("termination requested")
        return .terminateNow
    }

    func applicationWillTerminate(_ notification: Notification) {
        AppLog.write("app terminating")
        Task { @MainActor in await AppState.shared.engine.unload() }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }
}
