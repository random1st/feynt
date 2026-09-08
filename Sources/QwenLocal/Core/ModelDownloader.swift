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
    private let patterns = ["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model"]

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
                    let url = try await Self.fetch(spec: spec, patterns: self.patterns) { value, text in
                        Task { @MainActor [weak self] in
                            self?.fraction = value
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
        detail = "отменено"
    }

    private static func fetch(
        spec: ModelSpec, patterns: [String],
        onProgress: @Sendable @escaping (Double, String) -> Void
    ) async throws -> URL {
        Paths.ensureDirectory(Paths.modelsRoot)
        let client = HubClient(cache: HubCache(cacheDirectory: Paths.modelsRoot))
        let downloader = #hubDownloader(client)
        return try await downloader.download(
            id: spec.repo, revision: nil, matching: patterns, useLatest: false
        ) { progress in
            onProgress(progress.fractionCompleted, progress.localizedAdditionalDescription ?? "")
        }
    }
}
