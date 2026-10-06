import AppKit
import Combine
import Foundation
import UserNotifications

/// A Feynt newer than the running one, as GitHub reports it.
struct AvailableUpdate: Equatable {
    let version: String
    let page: URL
}

/// Tells the user when a newer Feynt is out. It does not install anything.
///
/// Feynt is distributed as a Homebrew cask, and an app that replaced itself would leave
/// Homebrew believing the old version is installed: `brew upgrade` would then reinstall what
/// is already there, or downgrade it. So this only notices and points - the notification says
/// `brew upgrade --cask feynt`, and the menu item opens the release page for anyone who
/// installed from the DMG.
///
/// One request to GitHub's public releases API, shortly after launch and then once a day.
/// Nothing about the machine or its use is sent; the request carries only the app's version
/// in its User-Agent, which GitHub asks every client to set.
@MainActor
final class UpdateChecker: NSObject, ObservableObject, UNUserNotificationCenterDelegate {
    @Published private(set) var available: AvailableUpdate?

    private let settings: AppSettings
    private var timer: Timer?
    private var subscription: AnyCancellable?

    private static let latestRelease = URL(
        string: "https://api.github.com/repos/random1st/feynt/releases/latest")!
    private static let interval: TimeInterval = 24 * 60 * 60
    /// The first check waits for the launch to settle - the model load and the listener
    /// matter more in those seconds than a version number does.
    private static let firstCheckDelay: TimeInterval = 15
    private static let notifiedKey = "notifiedUpdateVersion"

    init(settings: AppSettings) {
        self.settings = settings
        super.init()
    }

    var currentVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0"
    }

    /// Starts the daily check, and follows the menu's toggle from then on.
    func start() {
        UNUserNotificationCenter.current().delegate = self
        subscription = settings.$checkForUpdates.removeDuplicates().sink { [weak self] enabled in
            Task { @MainActor in self?.schedule(enabled) }
        }
    }

    private func schedule(_ enabled: Bool) {
        timer?.invalidate()
        timer = nil
        guard enabled else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.firstCheckDelay) { [weak self] in
            Task { @MainActor in await self?.check(userInitiated: false) }
        }
        timer = Timer.scheduledTimer(withTimeInterval: Self.interval, repeats: true) { [weak self] _ in
            Task { @MainActor in await self?.check(userInitiated: false) }
        }
    }

    /// Asks GitHub for the latest release. A check the user asked for always answers, in a
    /// window - "up to date" included, and a failure too; a scheduled one stays silent unless
    /// there is something new, and then notifies once per version.
    func check(userInitiated: Bool) async {
        // A build from source has no release to compare against; telling it that every
        // release is newer would be a notification a day for nothing.
        guard currentVersion.split(separator: ".").allSatisfy({ Int($0) != nil }) else {
            if userInitiated {
                alert("Not a release build", "Version \(currentVersion) is not a release, so there is nothing to compare.")
            }
            return
        }
        let latest: AvailableUpdate
        do {
            latest = try await Self.fetchLatest(userAgent: "Feynt/\(currentVersion)")
        } catch {
            AppLog.write("update check failed: \(error.localizedDescription)")
            if userInitiated {
                alert("Could not check for updates", "GitHub did not answer: \(error.localizedDescription)")
            }
            return
        }

        guard Self.isVersion(latest.version, newerThan: currentVersion) else {
            available = nil
            if userInitiated {
                alert("Feynt is up to date", "Version \(currentVersion) is the latest release.")
            }
            return
        }
        available = latest
        AppLog.write("update available: \(latest.version) (running \(currentVersion))")

        if userInitiated {
            offer(latest)
        } else if UserDefaults.standard.string(forKey: Self.notifiedKey) != latest.version {
            UserDefaults.standard.set(latest.version, forKey: Self.notifiedKey)
            await notify(latest)
        }
    }

    func openReleasePage() {
        guard let available else { return }
        NSWorkspace.shared.open(available.page)
    }

    // MARK: - GitHub

    private struct Release: Decodable {
        let tagName: String
        let htmlURL: URL
        enum CodingKeys: String, CodingKey {
            case tagName = "tag_name"
            case htmlURL = "html_url"
        }
    }

    /// `releases/latest` already skips drafts and pre-releases, so a release cut but not yet
    /// published is never announced.
    private static func fetchLatest(userAgent: String) async throws -> AvailableUpdate {
        var request = URLRequest(url: latestRelease, timeoutInterval: 20)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey: "HTTP \(code)"])
        }
        let release = try JSONDecoder().decode(Release.self, from: data)
        let version = release.tagName.hasPrefix("v") ? String(release.tagName.dropFirst()) : release.tagName
        return AvailableUpdate(version: version, page: release.htmlURL)
    }

    /// Numeric, component by component, so 0.10.0 is newer than 0.9.9 - a string comparison
    /// would get that backwards. A component that is not a number counts as zero, so this
    /// alone would call every release newer than a "dev" build; `check` refuses to run for a
    /// version that is not purely numeric, which is what keeps a build from source quiet.
    static func isVersion(_ candidate: String, newerThan current: String) -> Bool {
        func parts(_ v: String) -> [Int] { v.split(separator: ".").map { Int($0) ?? 0 } }
        let a = parts(candidate), b = parts(current)
        for i in 0 ..< max(a.count, b.count) {
            let x = i < a.count ? a[i] : 0, y = i < b.count ? b[i] : 0
            if x != y { return x > y }
        }
        return false
    }

    // MARK: - Telling the user

    private func notify(_ update: AvailableUpdate) async {
        let center = UNUserNotificationCenter.current()
        guard (try? await center.requestAuthorization(options: [.alert, .sound])) == true else {
            // Notifications refused: the menu item still says so.
            return
        }
        let content = UNMutableNotificationContent()
        content.title = "Feynt \(update.version) is available"
        content.body = "You have \(currentVersion). Update with: brew upgrade --cask feynt"
        content.userInfo = ["page": update.page.absoluteString]
        let request = UNNotificationRequest(
            identifier: "feynt.update.\(update.version)", content: content, trigger: nil)
        try? await center.add(request)
    }

    private func offer(_ update: AvailableUpdate) {
        NSApp.activate(ignoringOtherApps: true)
        let panel = NSAlert()
        panel.messageText = "Feynt \(update.version) is available"
        panel.informativeText = "You have \(currentVersion). Update with: brew upgrade --cask feynt"
        panel.addButton(withTitle: "Open release page")
        panel.addButton(withTitle: "Later")
        if panel.runModal() == .alertFirstButtonReturn {
            NSWorkspace.shared.open(update.page)
        }
    }

    private func alert(_ title: String, _ text: String) {
        NSApp.activate(ignoringOtherApps: true)
        let panel = NSAlert()
        panel.messageText = title
        panel.informativeText = text
        panel.runModal()
    }

    // MARK: - UNUserNotificationCenterDelegate

    /// Shown as a banner even while Feynt is frontmost - a menu-bar app is "active" often
    /// enough that the default, silent delivery would swallow most of these.
    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter, willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        [.banner, .sound]
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse
    ) async {
        guard let page = response.notification.request.content.userInfo["page"] as? String,
            let url = URL(string: page)
        else { return }
        await MainActor.run { _ = NSWorkspace.shared.open(url) }
    }
}
