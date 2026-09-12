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

        // `Feynt --download <model-id>` runs the wizard's download inside the real bundle -
        // same code, same signature, same entitlements, same network stack - and prints what
        // happens. Added because a download that fails only on someone else's machine cannot
        // be diagnosed from a test binary that shares none of those things.
        if let index = CommandLine.arguments.firstIndex(of: "--download") {
            let id = CommandLine.arguments.count > index + 1 ? CommandLine.arguments[index + 1] : ""
            runDownloadSelfTest(modelID: id)
            return
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
        Task { @MainActor in await AppState.shared.engine.unloadAll() }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    /// The diagnostic behind `--download`: resolve the catalog entry, fetch whatever is
    /// missing, print each step, and exit with a status a script can read.
    @MainActor private func runDownloadSelfTest(modelID: String) {
        guard let spec = ModelCatalog.model(id: modelID) else {
            print("no such model: \(modelID)")
            print("available: \(ModelCatalog.all.map(\.id).joined(separator: ", "))")
            exit(2)
        }
        let state = AppState.shared
        let missing = state.missingArtifacts(for: spec)
        print("model: \(spec.repo)")
        print("missing: \(missing.map(\.repo).joined(separator: ", "))")
        guard !missing.isEmpty else {
            print("already installed at \(ModelResolver.installedLocation(for: spec)?.path ?? "?")")
            exit(0)
        }
        state.downloader.download(missing) { success in
            print(success ? "download finished" : "download failed: \(state.downloader.errorMessage ?? "no reason given")")
            exit(success ? 0 : 1)
        }
        // The downloader reports on the main actor, so the run loop has to keep turning.
        Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { _ in
            let downloader = AppState.shared.downloader
            guard downloader.isDownloading else { return }
            print("  \(downloader.detail)")
        }
    }
}
