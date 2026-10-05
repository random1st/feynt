import Foundation

/// A model the app offers, together with the drafter that makes it fast.
///
/// The rule that got here was measured - a model stays only if speculation actually speeds
/// it up - and everything that passed it was run against its own plain decode, warm.
///
/// Acceptance, and so the rate, depends on what is being written, so every candidate was
/// measured on three workloads: a short templated coding prompt, an explanation over 2.9k
/// tokens of this repository's own source, and code written against that same context.
///
///     Qwen3.6-35B-A3B Uncens.  225-230      75-76          51-52    tok/s
///       accepted per round       9.27        2.18           3.32
///     Qwen3.8-27B Uncensored    14.6 ->  40 tok/s   2.7x   4.10 accepted per round
///     Qwen3.8-27B               18.6 ->  35 tok/s   1.9x   3.77
///     ------------------------------------------------------------------ dropped
///     Ornith-1.5-35B-A3B       257-261      70-73          34-40    tok/s
///       accepted per round      11.55        1.56           2.15
///     Qwen3-Coder-Next          59-61  (no DFlash 2 drafter exists, and 42 GB on disk)
///     LFM2.5-8B-A1B            204-208 (fast, but loops on an empty tool result)
///     Qwen3.6-27B               17.2   ->  41     tok/s   2.4x    4.86
///     Qwen3.5-27B               17.9   ->  26     tok/s   1.4x    4.12
///     Qwen3.5-9B                55.9   ->  57     tok/s   1.0x    3.10
///
/// The MoE rows were measured on 2026-09-21; the 27B rows predate the kernel and
/// chain-by-default work and stand as they were taken.
///
/// Ornith was the coding entry and is dropped on its own numbers. It leads only where a
/// drafter can guess what comes next; on the workload an agent actually runs - a long
/// context and a short piece of code - the 35B-A3B is ahead, 51-52 tok/s against 34-40,
/// and ahead on prose too. What it had was tool discipline, and that stopped being a
/// reason once the server learned to parse the `qwen3_5` dialect the 35B-A3B speaks: it
/// calls with the argument taken from prose, declines when nothing fits, and reports on an
/// empty result. An entry nobody should download is 19.5 GB of temptation in the model
/// window, so it goes.
///
/// The 3.5 generation went on the measurement: DFlash 1 drafters, no candidate selector,
/// and acceptance that says so. A drafter has to be trained against the weights it drafts
/// for, which is why the two 27Bs share `incoai/Qwen3.8-27B-DFlash2`.

struct ModelSpec: Identifiable, Hashable {
    let id: String
    let title: String
    let subtitle: String
    let repo: String
    let approximateBytes: Int64
    /// Repo of the DFlash drafter trained against this target, when speculation pays for
    /// itself here. `nil` is a measured decision, not a gap: a model whose step is too
    /// cheap to amortise a drafter runs faster without one, and every entry in the catalog
    /// today has earned its drafter on the numbers.
    let drafterRepo: String?
    let drafterApproximateBytes: Int64

    var directoryName: String {
        repo.split(separator: "/").last.map(String.init) ?? repo
    }

    /// The drafter as its own downloadable spec, or `nil` for a model that runs plain.
    var drafter: ModelSpec? {
        guard let drafterRepo else { return nil }
        return ModelSpec(
            id: "\(id).drafter",
            title: "Drafter for \(title)",
            subtitle: "speculative drafter",
            repo: drafterRepo,
            approximateBytes: drafterApproximateBytes,
            drafterRepo: drafterRepo,
            drafterApproximateBytes: drafterApproximateBytes)
    }

    /// Everything this model needs on disk: itself, and the drafter if it has one.
    var artifacts: [ModelSpec] { [self] + (drafter.map { [$0] } ?? []) }

    /// Bytes a fresh download costs.
    var totalApproximateBytes: Int64 {
        approximateBytes + (drafterRepo == nil ? 0 : drafterApproximateBytes)
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

    /// Roman's daily model: a MoE that reads ~3B of its 35B parameters per token, so even
    /// its plain decode (106-109 tok/s) beats what the dense 27B reaches with speculation.
    ///
    /// Speculation used to buy only 1.1x here (86.2 -> 95 tok/s, 3.74 accepted per round),
    /// because the drafter was trained against the original weights and this is an
    /// abliterated variant of them. The kernel and chain-by-default work since then moved
    /// it to 225-230 tok/s at 9.27 accepted per round - 2.1x.
    ///
    /// The pairing below is tuned for an agent, and that is a trade rather than a free win.
    /// Measured on 2026-10-04, interleaved runs, spread under 1% within each configuration:
    ///
    ///                                   agent    short code   long ctx   prose
    ///     z-lab DFlash 1, block 16      84.8     184.9        65.8       110.8  tok/s
    ///     incoai DFlash 2 at 4 bits     97.8     201.7        73.5       100.2
    ///
    /// "agent" is 5.3k tokens of this repository's own source with a request to write code
    /// against it - the shape an agent actually sends. DFlash 2 wins it by 15% and accepts
    /// more per round there (3.22 against 3.07), and loses prose by 10%. Prose is what an
    /// agent does not write.
    ///
    /// The drafter runs as published. Quantising it was measured and dropped: four bits
    /// beat bf16 on one agent prompt by 6% and lost to it on another by 5%, and a round of
    /// the stand kept bf16 over four bits and then eight bits over bf16, each at P=1.00.
    /// That is the signature of noise, not of a knob. What the experiment did settle is
    /// that the round is not bound by the drafter's bytes: four bits leave a quarter of
    /// them, and dropping the other three quarters produced no consistent gain in either
    /// direction. What that rules out is the drafter's weight traffic. It does not
    /// establish what the cost *is* - that needs a per-round decomposition, which nothing
    /// here measures yet.
    ///
    /// The width is the knob that matters here and it is not in this file. At cap 3 the
    /// agent workload runs 117.7-118.9 tok/s against 86.6-88.4 at the default, +36%, while
    /// short code falls from 221-229 to 154-157. The drafter accepts about three tokens
    /// over a long context and drafts seven, so four verified rows a round are thrown
    /// away; on a short templated prompt it accepts six and the same narrowing throws the
    /// speedup away instead. Neither the cap nor the generator's gate reads the context
    /// length, which is what would let both workloads have it.
    static let uncensoredMoE = ModelSpec(
        id: "uncensored-moe",
        title: "Qwen3.6-35B-A3B Uncensored",
        subtitle: "MoE, uncensored, the fastest",
        repo: "froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit",
        approximateBytes: 19_600_000_000,
        drafterRepo: "incoai/Qwen3.6-35B-A3B-DFlash2",
        drafterApproximateBytes: 1_053_000_000)

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
