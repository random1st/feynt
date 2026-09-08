import Foundation

/// One of the two models the app offers. Deliberately a closed set: this is a launcher for
/// one local LLM, not a model manager, so there is no discovery and no free-form repo entry.
struct ModelSpec: Identifiable, Hashable {
    let id: String
    let title: String
    let subtitle: String
    let repo: String
    let approximateBytes: Int64

    var directoryName: String {
        repo.split(separator: "/").last.map(String.init) ?? repo
    }
}

enum ModelCatalog {
    static let stock = ModelSpec(
        id: "stock",
        title: "Qwen3.8-27B",
        subtitle: "стоковая",
        repo: "mlx-community/Qwen3.8-27B-4bit",
        approximateBytes: 16 * 1_000_000_000)

    static let uncensored = ModelSpec(
        id: "uncensored",
        title: "Qwen3.8-27B Uncensored",
        subtitle: "без цензуры",
        repo: "orcarouter/Qwen3.8-27B-Uncensored-MLX",
        approximateBytes: 16 * 1_000_000_000)

    /// Both targets share one drafter; without it generation still works, just without
    /// speculative decoding.
    static let drafter = ModelSpec(
        id: "drafter",
        title: "DFlash2 drafter",
        subtitle: "ускоритель",
        repo: "incoai/Qwen3.8-27B-DFlash2",
        approximateBytes: 3_700_000_000)

    static let all: [ModelSpec] = [stock, uncensored]

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
