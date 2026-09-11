import Foundation

/// A model the app offers, together with the drafter that makes it fast.
///
/// Two entries, one generation, one drafter. The rule that got here was measured - a model
/// stays only if speculation actually speeds it up - and everything that passed it was run
/// against its own plain decode, warm, twice:
///
///     Qwen3.6-35B-A3B Uncens.  86.2 -> 95   tok/s   1.1x   3.74 accepted per round
///     Qwen3.8-27B Uncensored   14.6 -> 40   tok/s   2.7x   4.10
///     Qwen3.8-27B              18.6 -> 35   tok/s   1.9x   3.77
///     ---------------------------------------------------- dropped
///     Qwen3.6-27B              17.2 -> 41   tok/s   2.4x   4.86
///     Qwen3.6-35B-A3B          73   -> 121  tok/s   1.7x   7.08
///     Qwen3.5-27B              17.9 -> 26   tok/s   1.4x   4.12
///     Qwen3.5-35B-A3B          74   -> 81   tok/s   1.1x   3.95
///     Qwen3.5-9B               55.9 -> 57   tok/s   1.0x   3.10
///
/// The 3.5 generation went on the measurement: DFlash 1 drafters, no candidate selector,
/// and acceptance that says so. The 3.6 generation went on a product call - one supported
/// generation instead of three - which costs the MoE, the fastest thing here at 121 tok/s.
/// `incoai/Qwen3.8-27B-DFlash2` is the only DFlash 2 drafter that exists for a Qwen, and
/// both remaining models share it.

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
            subtitle: "speculative drafter",
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
        approximateBytes: 16_100_000_000,
        drafterRepo: dflash2_27B,
        drafterApproximateBytes: 3_800_000_000)

    static let stock = ModelSpec(
        id: "stock",
        title: "Qwen3.8-27B",
        subtitle: "stock",
        repo: "mlx-community/Qwen3.8-27B-4bit",
        approximateBytes: 16_000_000_000,
        drafterRepo: dflash2_27B,
        drafterApproximateBytes: 3_700_000_000)

    /// Roman's daily model, and the exception to the rule above: speculation buys only
    /// 1.1x here (86.2 -> 95 tok/s, 3.74 accepted per round), because the drafter was
    /// trained against the original weights and this is an abliterated variant of them.
    /// It is listed anyway because it is the fastest model in this catalog by a wide
    /// margin: a MoE reads ~3B of its 35B parameters per token, so even unaccelerated it
    /// runs at twice what the dense 27B reaches with speculation.
    static let uncensoredMoE = ModelSpec(
        id: "uncensored-moe",
        title: "Qwen3.6-35B-A3B Uncensored",
        subtitle: "MoE, uncensored, the fastest",
        repo: "froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit",
        approximateBytes: 19_600_000_000,
        drafterRepo: "z-lab/Qwen3.6-35B-A3B-DFlash",
        drafterApproximateBytes: 771_800_000)

    static let all: [ModelSpec] = [uncensoredMoE, uncensored, stock]

    static func model(id: String) -> ModelSpec? {
        all.first { $0.id == id }
    }
}

/// Finds a repo on disk before deciding anything needs downloading.
enum ModelResolver {
    /// A directory counts as an installed model when it has a config and at least one
    /// weight shard with bytes in it.
    ///
    /// The size test is the whole point. A download that died left `model.safetensors` at
    /// zero length, this returned true, and the app decided the model was installed - so it
    /// never offered to fetch it again. Roman's machine sat at 14 KB on disk, which is the
    /// JSON files and an empty shard, with no way to retry.
    static func isInstalled(_ directory: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: directory.appending(path: "config.json").path) else { return false }
        guard let entries = try? fm.contentsOfDirectory(atPath: directory.path) else { return false }
        return entries.contains { name in
            guard name.hasSuffix(".safetensors") else { return false }
            let size = (try? fm.attributesOfItem(atPath: directory.appending(path: name).path)[.size]) as? Int64
            return (size ?? 0) > 0
        }
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

    /// Forgets where a repository was downloaded to. Needed when its files are deleted:
    /// a remembered path outlives them and would answer for a directory that is gone.
    static func forgetLocation(for repo: String) {
        var map = UserDefaults.standard.dictionary(forKey: recordKey) as? [String: String] ?? [:]
        map[repo] = nil
        UserDefaults.standard.set(map, forKey: recordKey)
    }

    static func recordedLocation(for repo: String) -> URL? {
        guard let map = UserDefaults.standard.dictionary(forKey: recordKey) as? [String: String],
              let path = map[repo]
        else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }
}
