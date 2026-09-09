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
    nonisolated private static func rootFiles(
        _ client: HubClient, repo: String
    ) async throws -> (files: [RemoteFile], bytes: Int64) {
        let tree = try await client.modelTree(Repo.ID(stringLiteral: repo))
        let entries = tree.filter { entry in
            entry.type == .file && !entry.path.contains("/")
                && wanted.contains { entry.path.hasSuffix($0) }
        }
        let files = entries.map { RemoteFile(path: $0.path, bytes: Int64($0.size ?? 0)) }
        return (files, files.reduce(0) { $0 + $1.bytes })
    }

    private static func fetch(
        spec: ModelSpec,
        onProgress: @Sendable @escaping (Double, String) -> Void
    ) async throws -> URL {
        Paths.ensureDirectory(Paths.modelsRoot)
        let client = HubClient(cache: HubCache(cacheDirectory: Paths.modelsRoot))
        let (files, totalBytes) = try await rootFiles(client, repo: spec.repo)
        guard files.contains(where: { $0.path.hasSuffix(".safetensors") }) else {
            throw DownloadError.noWeightsAtRoot(spec.repo)
        }

        // The flat layout the resolver already looks in, rather than the client's
        // content-addressed cache: one directory named after the repository, which is what
        // every other model on disk here looks like.
        let destination = Paths.modelsRoot.appending(
            path: spec.directoryName, directoryHint: .isDirectory)
        let started = Date()
        try await ParallelDownloader().download(
            repo: spec.repo, files: files, into: destination
        ) { done in
            let fraction = totalBytes > 0 ? Double(done) / Double(totalBytes) : 0
            let elapsed = max(Date().timeIntervalSince(started), 1)
            let rate = Double(done) / elapsed
            let remaining = rate > 0 ? Double(totalBytes - done) / rate : 0
            onProgress(
                fraction,
                "\(Paths.formatBytes(done)) of \(Paths.formatBytes(totalBytes)) · "
                    + "\(Paths.formatBytes(Int64(rate)))/s · \(remainingText(remaining)) left")
        }
        return destination
    }

    nonisolated private static func remainingText(_ seconds: TimeInterval) -> String {
        guard seconds.isFinite, seconds > 0 else { return "—" }
        let minutes = Int(seconds.rounded()) / 60
        if minutes >= 60 { return "\(minutes / 60) h \(minutes % 60) min" }
        if minutes >= 1 { return "\(minutes) min" }
        return "\(Int(seconds.rounded())) s"
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
