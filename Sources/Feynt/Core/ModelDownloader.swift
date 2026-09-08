import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon

/// Downloads a catalog repo using mlx-swift-lm's own Hugging Face client.
///
/// Its snapshot download already reports `Progress` and understands the repo layout, so the
/// app does not reimplement file listing or resume. Downloads land under the app's models
/// root; the resolved directory is remembered so the resolver finds it on the next launch.
@MainActor
final class ModelDownloader: ObservableObject {
    @Published private(set) var isDownloading = false
    @Published private(set) var fraction: Double = 0
    @Published private(set) var detail: String = ""
    @Published private(set) var currentRepo: String?
    @Published private(set) var errorMessage: String?

    private var task: Task<Void, Never>?

    /// Weights, configs and chat templates; everything else in a repo is dead weight here.
    private static let wanted = [".safetensors", ".json", ".jinja", ".txt", ".model"]

    func download(_ specs: [ModelSpec], completion: @escaping (Bool) -> Void) {
        guard !isDownloading else { return }
        let missing = specs.filter { !ModelResolver.isPresent($0) }
        guard !missing.isEmpty else {
            completion(true)
            return
        }

        isDownloading = true
        errorMessage = nil
        fraction = 0

        task = Task { [weak self] in
            guard let self else { return }
            var success = true
            for spec in missing {
                if Task.isCancelled { success = false; break }
                self.currentRepo = spec.repo
                self.detail = spec.title
                do {
                    let watcher = Self.observeBytes(
                        expected: spec.approximateBytes,
                        update: { value in
                            Task { @MainActor [weak self] in self?.fraction = value }
                        })
                    defer { watcher.cancel() }
                    let url = try await Self.fetch(spec: spec) { value, text in
                        Task { @MainActor [weak self] in
                            if value >= 0 { self?.fraction = value }
                            if !text.isEmpty { self?.detail = text }
                        }
                    }
                    ModelResolver.recordLocation(url, for: spec.repo)
                    AppLog.write("downloaded \(spec.repo) -> \(url.path)")
                } catch is CancellationError {
                    success = false
                    break
                } catch {
                    self.errorMessage = error.localizedDescription
                    AppLog.write("download failed \(spec.repo): \(error.localizedDescription)")
                    success = false
                    break
                }
            }
            self.isDownloading = false
            self.currentRepo = nil
            completion(success)
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
        isDownloading = false
        detail = "cancelled"
    }

    /// Drives the progress bar from bytes actually written under the models root.
    ///
    /// The library downloader's own progress is too coarse to watch a multi-gigabyte
    /// download by, and a bar that does not move is indistinguishable from a hang. Disk
    /// growth is the one signal that is always truthful here.
    private static func observeBytes(expected: Int64, update: @escaping (Double) -> Void) -> Task<Void, Never> {
        let base = Paths.directorySize(Paths.modelsRoot)
        return Task.detached {
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(700))
                if Task.isCancelled { return }
                let grown = Paths.directorySize(Paths.modelsRoot) - base
                guard expected > 0 else { continue }
                update(min(0.99, max(0, Double(grown) / Double(expected))))
            }
        }
    }

    /// The files to fetch, named one by one from the repository's root listing.
    ///
    /// A glob will not do. The client matches with `fnmatch` and no `FNM_PATHNAME`, so `*`
    /// crosses directory separators and `*.safetensors` takes every file in every
    /// subdirectory. Repositories that publish several quantisations of the same weights -
    /// the uncensored 27B ships 2-, 4-, 6- and 8-bit, with the 4-bit one at the root - then
    /// turn a 16 GB download into a 95 GB one, which is what a user sees before anything
    /// else has gone wrong.
    ///
    /// So the root is listed and its files are named exactly. Nothing recursive is ever
    /// asked for, and a repository that rearranges itself fails loudly here rather than
    /// silently pulling five copies of a model.
    private static func rootFiles(_ client: HubClient, repo: String) async throws -> [String] {
        let tree = try await client.modelTree(Repo.ID(stringLiteral: repo))
        return tree
            .filter { $0.type == .file && !$0.path.contains("/") }
            .map(\.path)
            .filter { path in wanted.contains { path.hasSuffix($0) } }
    }

    private static func fetch(
        spec: ModelSpec,
        onProgress: @Sendable @escaping (Double, String) -> Void
    ) async throws -> URL {
        Paths.ensureDirectory(Paths.modelsRoot)
        let client = HubClient(cache: HubCache(cacheDirectory: Paths.modelsRoot))
        let files = try await rootFiles(client, repo: spec.repo)
        guard files.contains(where: { $0.hasSuffix(".safetensors") }) else {
            throw DownloadError.noWeightsAtRoot(spec.repo)
        }
        let downloader = #hubDownloader(client)
        return try await downloader.download(
            id: spec.repo, revision: nil, matching: files, useLatest: false
        ) { progress in
            // Only the detail text is used from here. Measured against a real repo, both
            // `fractionCompleted` and the unit counts advance in rare jumps - fine for an
            // 80 MB model, but on a 16 GB one the bar would sit still for tens of minutes
            // and the app would read as hung. The caller drives the bar from bytes on disk
            // instead; see `observeBytes`.
            onProgress(-1, progress.localizedAdditionalDescription ?? "")
        }
    }
}

enum DownloadError: LocalizedError {
    case noWeightsAtRoot(String)

    var errorDescription: String? {
        switch self {
        case .noWeightsAtRoot(let repo):
            "\(repo) has no weights at its root — the catalog entry points at the wrong path"
        }
    }
}
