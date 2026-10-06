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

    init() {
        Self.yieldToRunningInstance()
    }

    /// Exits if another Feynt is already running, and brings that one forward instead.
    ///
    /// Two copies fight over the port - the second fails to listen and sits in the menu bar
    /// looking alive - and each loads its own weights, which is 20 GB twice. Launch Services
    /// already refuses a second launch of the same bundle; this catches the other way in, a
    /// second bundle with the same identifier (a build next to the installed app). Checked
    /// here, before the app state exists, so the loser never binds the port or loads a model.
    ///
    /// Only an instance launched earlier wins, so two started at the same moment do not both
    /// leave. `--generate` and `--download` are diagnostics that print and exit, and run beside
    /// the app on purpose.
    private static func yieldToRunningInstance() {
        let headless = CommandLine.arguments.contains { $0 == "--generate" || $0 == "--download" }
        guard !headless, let id = Bundle.main.bundleIdentifier else { return }
        let me = NSRunningApplication.current
        let mine = me.launchDate ?? Date()
        let earlier = NSRunningApplication.runningApplications(withBundleIdentifier: id).first {
            guard $0.processIdentifier != me.processIdentifier, !$0.isTerminated else { return false }
            let theirs = $0.launchDate ?? .distantPast
            return theirs < mine || (theirs == mine && $0.processIdentifier < me.processIdentifier)
        }
        guard let earlier else { return }
        AppLog.write("another Feynt is running (pid \(earlier.processIdentifier)); exiting")
        earlier.activate()
        exit(0)
    }


    var body: some Scene {
        Window("Feynt", id: MainWindowID.value) {
            RootView(settings: state.settings)
                .environmentObject(state)
        }
        .defaultSize(width: 780, height: 560)


        MenuBarExtra {
            MenuBarContent(
                engine: state.engine, settings: state.settings, api: state.api,
                updates: state.updates)
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

        // `Feynt --generate <model-id> <prompt>` is the same diagnostic one step further:
        // load through the real engine stack and answer once, printing tok/s. Added when
        // LFM2.5 joined the catalog — a new architecture has to be checked against the
        // reference runtime (same prompt, greedy, compare text and rate), and doing that
        // through the UI proves nothing about the headless path the API server uses.
        if let index = CommandLine.arguments.firstIndex(of: "--generate") {
            let args = CommandLine.arguments
            let id = args.count > index + 1 ? args[index + 1] : ""
            let prompt = args.count > index + 2 ? args[index + 2] : "Reply with exactly: ok"
            runGenerateSelfTest(modelID: id, prompt: prompt)
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

    /// The diagnostic behind `--generate`: load the named catalog model through the engine
    /// controller the app itself uses, answer one prompt greedily, print the counters.
    @MainActor private func runGenerateSelfTest(modelID: String, prompt: String) {
        guard let spec = ModelCatalog.model(id: modelID) else {
            print("no such model: \(modelID)")
            print("available: \(ModelCatalog.all.map(\.id).joined(separator: ", "))")
            exit(2)
        }
        Task { @MainActor in await generateOnce(spec: spec, prompt: prompt) }
    }
}
