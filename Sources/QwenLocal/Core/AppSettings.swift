import Foundation
import SwiftUI

/// User-visible preferences, persisted in `UserDefaults`.
@MainActor
final class AppSettings: ObservableObject {
    /// Sentinel for "никогда" — the idle timer is disabled.
    static let idleNever = 0
    static let idleChoices = [60, 300, 900, 1800, 3600, idleNever]

    private let defaults = UserDefaults.standard

    @Published var port: Int { didSet { defaults.set(port, forKey: "port") } }
    @Published var idleTimeout: Int { didSet { defaults.set(idleTimeout, forKey: "idleTimeout") } }
    @Published var thinkingByDefault: Bool {
        didSet { defaults.set(thinkingByDefault, forKey: "thinkingByDefault") }
    }
    @Published var maxTokens: Int { didSet { defaults.set(maxTokens, forKey: "maxTokens") } }
    @Published var selectedModelID: String {
        didSet { defaults.set(selectedModelID, forKey: "selectedModelID") }
    }
    @Published var wizardCompleted: Bool {
        didSet { defaults.set(wizardCompleted, forKey: "wizardCompleted") }
    }

    init() {
        port = (defaults.object(forKey: "port") as? Int) ?? 19234
        idleTimeout = (defaults.object(forKey: "idleTimeout") as? Int) ?? 300
        thinkingByDefault = defaults.bool(forKey: "thinkingByDefault")
        maxTokens = (defaults.object(forKey: "maxTokens") as? Int) ?? 2048
        selectedModelID = defaults.string(forKey: "selectedModelID") ?? ModelCatalog.uncensored.id
        wizardCompleted = defaults.bool(forKey: "wizardCompleted")
    }

    var selectedModel: ModelSpec {
        ModelCatalog.model(id: selectedModelID) ?? ModelCatalog.uncensored
    }

    static func idleLabel(_ seconds: Int) -> String {
        switch seconds {
        case idleNever: return "никогда"
        case ..<3600: return "\(seconds / 60) мин"
        default: return "\(seconds / 3600) ч"
        }
    }
}
