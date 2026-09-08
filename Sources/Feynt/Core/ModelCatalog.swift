import Foundation

/// A model the app offers, together with the drafter that makes it fast.
///
/// The catalog is closed on purpose, but the rule is not "few models" - it is that every
/// entry has a matching DFlash drafter. A target without one runs at plain decode speed,
/// which is the one thing this app exists to avoid, so listing it would sell a promise the
/// app cannot keep for that row.
struct ModelSpec: Identifiable, Hashable {
    let id: String
    let title: String
    let subtitle: String
    let repo: String
    let approximateBytes: Int64
    /// Repo of the DFlash drafter trained against this target.
    let drafterRepo: String
    let drafterApproximateBytes: Int64

    var directoryName: String {
        repo.split(separator: "/").last.map(String.init) ?? repo
    }

    /// The drafter as its own downloadable spec.
    var drafter: ModelSpec {
        ModelSpec(
            id: "\(id).drafter",
            title: "Drafter for \(title)",
            subtitle: "DFlash",
            repo: drafterRepo,
            approximateBytes: drafterApproximateBytes,
            drafterRepo: drafterRepo,
            drafterApproximateBytes: drafterApproximateBytes)
    }
}

enum ModelCatalog {
    private static let dflash2_27B = "incoai/Qwen3.8-27B-DFlash2"

    static let uncensored = ModelSpec(
        id: "uncensored",
        title: "Qwen3.8-27B Uncensored",
        subtitle: "uncensored",
        repo: "orcarouter/Qwen3.8-27B-Uncensored-MLX",
        approximateBytes: 16_000_000_000,
        drafterRepo: dflash2_27B,
        drafterApproximateBytes: 3_700_000_000)

    static let stock = ModelSpec(
        id: "stock",
        title: "Qwen3.8-27B",
        subtitle: "stock",
        repo: "mlx-community/Qwen3.8-27B-4bit",
        approximateBytes: 16_000_000_000,
        drafterRepo: dflash2_27B,
        drafterApproximateBytes: 3_700_000_000)

    static let qwen36 = ModelSpec(
        id: "qwen36-27b",
        title: "Qwen3.6-27B",
        subtitle: "previous generation",
        repo: "mlx-community/Qwen3.6-27B-4bit",
        approximateBytes: 16_000_000_000,
        drafterRepo: "z-lab/Qwen3.6-27B-DFlash",
        drafterApproximateBytes: 3_700_000_000)

    static let qwen35 = ModelSpec(
        id: "qwen35-27b",
        title: "Qwen3.5-27B",
        subtitle: "previous generation",
        repo: "mlx-community/Qwen3.5-27B-4bit",
        approximateBytes: 16_000_000_000,
        drafterRepo: "z-lab/Qwen3.5-27B-DFlash",
        drafterApproximateBytes: 3_700_000_000)

    /// The small one: fits comfortably where a 27B does not, and answers sooner.
    static let qwen35small = ModelSpec(
        id: "qwen35-9b",
        title: "Qwen3.5-9B",
        subtitle: "light",
        repo: "mlx-community/Qwen3.5-9B-4bit",
        approximateBytes: 5_500_000_000,
        drafterRepo: "z-lab/Qwen3.5-9B-DFlash",
        drafterApproximateBytes: 1_600_000_000)

    /// Mixture-of-experts targets. Only about 3B of the 35B parameters are read per token,
    /// and decode here is bound by exactly that traffic, so these run several times faster
    /// than a dense 27B on the same machine while holding a comparable footprint on disk.
    static let qwen36moe = ModelSpec(
        id: "qwen36-35b-a3b",
        title: "Qwen3.6-35B-A3B",
        subtitle: "MoE, the fastest",
        repo: "mlx-community/Qwen3.6-35B-A3B-4bit",
        approximateBytes: 20_000_000_000,
        drafterRepo: "z-lab/Qwen3.6-35B-A3B-DFlash",
        drafterApproximateBytes: 3_700_000_000)

    static let qwen35moe = ModelSpec(
        id: "qwen35-35b-a3b",
        title: "Qwen3.5-35B-A3B",
        subtitle: "MoE",
        repo: "mlx-community/Qwen3.5-35B-A3B-4bit",
        approximateBytes: 20_000_000_000,
        drafterRepo: "z-lab/Qwen3.5-35B-A3B-DFlash",
        drafterApproximateBytes: 3_700_000_000)

    static let all: [ModelSpec] = [
        uncensored, stock, qwen36moe, qwen36, qwen35moe, qwen35, qwen35small,
    ]

    static func model(id: String) -> ModelSpec? {
        all.first { $0.id == id }
    }
}

/// Finds a repo on disk before deciding anything needs downloading.
enum ModelResolver {
    /// A directory counts as an installed model when it has a config and at least one
    /// weight shard — a half-finished download must never pass this test.
    static func isInstalled(_ directory: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: directory.appending(path: "config.json").path) else { return false }
        guard let entries = try? fm.contentsOfDirectory(atPath: directory.path) else { return false }
        return entries.contains { $0.hasSuffix(".safetensors") }
    }

    /// Candidate locations, most specific first. `recordedLocation` is where a previous
    /// download by this app landed (the Hugging Face client picks its own cache root, so we
    /// remember the URL it returned rather than guessing it).
    static func candidates(for spec: ModelSpec) -> [URL] {
        var result: [URL] = []
        if let recorded = recordedLocation(for: spec.repo) {
            result.append(recorded)
        }
        result.append(Paths.modelsRoot.appending(path: spec.directoryName, directoryHint: .isDirectory))
        result.append(
            Paths.home.appending(
                path: ".cache/huggingface/models/\(spec.repo)", directoryHint: .isDirectory))
        return result
    }

    static func installedLocation(for spec: ModelSpec) -> URL? {
        candidates(for: spec).first(where: isInstalled)
    }

    static func isPresent(_ spec: ModelSpec) -> Bool {
        installedLocation(for: spec) != nil
    }

    // MARK: - Remembering download destinations

    private static let recordKey = "downloadedModelLocations"

    static func recordLocation(_ url: URL, for repo: String) {
        var map = UserDefaults.standard.dictionary(forKey: recordKey) as? [String: String] ?? [:]
        map[repo] = url.path
        UserDefaults.standard.set(map, forKey: recordKey)
    }

    static func recordedLocation(for repo: String) -> URL? {
        guard let map = UserDefaults.standard.dictionary(forKey: recordKey) as? [String: String],
              let path = map[repo]
        else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }
}
