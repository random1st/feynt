import Foundation

/// One file to fetch, with the size the repository listing reported for it.
struct RemoteFile: Sendable, Hashable {
    let path: String
    let bytes: Int64
}

/// Downloads a repository's files with ranged requests, several in flight at once.
///
/// The Hub client fetches a file over a single connection. Measured on this machine against
/// the same 736 MB repository: 183 seconds through that client, 15 through a chunked
/// parallel one — 4 MB/s against 50. On a 20 GB model that is an hour and a half instead of
/// seven minutes, and an hour of a bar creeping a percent a minute is indistinguishable
/// from a hang, which is exactly how it was read.
///
/// Chunks are 32 MB and at most eight are in flight, so peak memory stays a quarter of a
/// gigabyte no matter how large the model is. Each chunk is written straight to its offset
/// in the destination file, so a partial download leaves a file of the right length with
/// holes rather than a truncated one — which is why resume compares sizes only for files
/// this downloader finished.
actor ParallelDownloader {
    enum Failure: LocalizedError {
        case badStatus(String, Int)
        case shortRead(String, expected: Int, got: Int)

        var errorDescription: String? {
            switch self {
            case .badStatus(let path, let code):
                "\(path): the server answered \(code)"
            case .shortRead(let path, let expected, let got):
                "\(path): expected \(expected) bytes, got \(got)"
            }
        }
    }

    /// Куски по 8 МБ, а не по 32: счётчик докладывает по завершённому куску, и на
    /// медленном канале тридцать два мегабайта — это минуты неподвижного «0 MB», которые
    /// невозможно отличить от зависшей загрузки.
    private static let chunkBytes = 8 << 20
    private static let inFlight = 8

    private let session: URLSession
    /// Where files are fetched from. Injectable so the two response shapes this downloader
    /// has to survive - a server that slices, and one that ignores `Range` - can both be
    /// exercised against a real socket instead of reasoned about.
    private let base: URL
    /// Sent only to `base`, and only when the environment or the Hub's own token file has
    /// one. Public repositories need none; a private or gated one needs it, and every other
    /// Hub client reads it from the same two places.
    private let token: String?

    /// The endpoint every Hugging Face client honours. A mirror, a corporate proxy or an
    /// enterprise deployment is configured through it, and hard-coding huggingface.co means
    /// the app is the one thing on such a machine that cannot reach the Hub.
    static var configuredEndpoint: URL {
        if let raw = ProcessInfo.processInfo.environment["HF_ENDPOINT"],
            let url = URL(string: raw.trimmingCharacters(in: .whitespaces)), url.host != nil
        {
            return url
        }
        return URL(string: "https://huggingface.co")!
    }

    static var storedToken: String? {
        let environment = ProcessInfo.processInfo.environment
        for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
            if let value = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
                !value.isEmpty
            {
                return value
            }
        }
        let file = Paths.home.appending(path: ".cache/huggingface/token")
        guard let contents = try? String(contentsOf: file, encoding: .utf8) else { return nil }
        let trimmed = contents.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    init(base: URL? = nil, token: String? = nil) {
        self.base = base ?? Self.configuredEndpoint
        self.token = token ?? Self.storedToken
        let configuration = URLSessionConfiguration.ephemeral
        // The default cap is six per host, which would silently bound the parallelism this
        // whole type exists for.
        configuration.httpMaximumConnectionsPerHost = Self.inFlight
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForRequest = 60
        session = URLSession(configuration: configuration)
    }

    /// Fetches `files` into `directory`, calling `onBytes` with the running total.
    ///
    /// Files already present at their full size are counted and skipped, so an interrupted
    /// download resumes at file granularity.
    func download(
        repo: String, revision: String = "main", files: [RemoteFile], into directory: URL,
        onBytes: @Sendable @escaping (Int64) -> Void
    ) async throws {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)

        var done: Int64 = 0
        for file in files {
            try Task.checkCancellation()
            let destination = directory.appending(path: file.path)
            if file.bytes > 0,
                let size = try? manager.attributesOfItem(atPath: destination.path)[.size] as? Int64,
                size == file.bytes
            {
                done += file.bytes
                onBytes(done)
                continue
            }
            let before = done
            try await fetch(repo: repo, revision: revision, file: file, to: destination) { written in
                onBytes(before + written)
            }
            done += file.bytes
            onBytes(done)
        }
    }

    private func fetch(
        repo: String, revision: String, file: RemoteFile, to destination: URL,
        onBytes: @Sendable @escaping (Int64) -> Void
    ) async throws {
        let manager = FileManager.default
        try manager.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        manager.createFile(atPath: destination.path, contents: nil)

        let url = url(repo: repo, revision: revision, path: file.path)

        // A listing that did not carry sizes must not turn into a download of nothing.
        // Splitting a file into chunks needs its length, and without one the ranges come
        // out empty - a request for `bytes=0--1`, a rejected response, and zero bytes on
        // disk. So an unknown size is fetched whole, in one request, which is slower and
        // correct; the fast path stays for everything the listing did describe.
        guard file.bytes > 0 else {
            try await fetchWhole(url: url, path: file.path, to: destination, onBytes: onBytes)
            return
        }

        let ranges = Self.ranges(of: Int(file.bytes))
        let progress = ByteCounter(onBytes)

        // The first chunk is also the probe. Hugging Face answers a file request with a 307
        // to its CDN, so every ranged request only works if `Range` survives that redirect -
        // it does here, and a proxy or a VPN on the way can drop it. When that happens the
        // server sends 200 with the whole file instead of 206 with a slice, and a downloader
        // that insists on 206 fails on the second chunk and writes nothing at all. So: ask
        // for the first slice, look at what came back, and only fan out if slicing worked.
        let (head, headCode) = try await request(url: url, range: ranges[0], path: file.path)
        guard headCode == 206 else {
            guard headCode == 200 else { throw Failure.badStatus(file.path, headCode) }
            // Ranges were ignored; this is the entire file.
            try head.write(to: destination)
            onBytes(Int64(head.count))
            return
        }
        try Self.write(head, to: destination, at: ranges[0].lowerBound)
        await progress.add(Int64(head.count))

        try await withThrowingTaskGroup(of: Void.self) { group in
            var started = 0
            for range in ranges.dropFirst() {
                if started >= Self.inFlight {
                    try await group.next()
                    started -= 1
                }
                group.addTask {
                    let (data, code) = try await self.request(
                        url: url, range: range, path: file.path)
                    guard code == 206 else { throw Failure.badStatus(file.path, code) }
                    guard data.count == range.count else {
                        throw Failure.shortRead(file.path, expected: range.count, got: data.count)
                    }
                    try Self.write(data, to: destination, at: range.lowerBound)
                    await progress.add(Int64(data.count))
                }
                started += 1
            }
            try await group.waitForAll()
        }
    }

    /// One ranged request, retried on a dropped connection: an hour into a download that
    /// should cost the chunk, not the model.
    private func request(url: URL, range: Range<Int>, path: String) async throws -> (Data, Int) {
        var request = URLRequest(url: url)
        request.setValue(
            "bytes=\(range.lowerBound)-\(range.upperBound - 1)", forHTTPHeaderField: "Range")
        authorize(&request)
        var attempt = 0
        while true {
            do {
                let (data, response) = try await session.data(for: request)
                let code = (response as? HTTPURLResponse)?.statusCode ?? 0
                // Every model this app offers is public, so an unauthenticated fetch is the
                // path that must always work. A token found on the machine is an attempt,
                // not a requirement: a stale one left by an old `hf login` would otherwise
                // turn a file that downloads fine for a stranger into a 401.
                if (code == 401 || code == 403),
                    request.value(forHTTPHeaderField: "Authorization") != nil
                {
                    request.setValue(nil, forHTTPHeaderField: "Authorization")
                    continue
                }
                return (data, code)
            } catch {
                attempt += 1
                if attempt >= 3 { throw error }
                try await Task.sleep(for: .seconds(attempt))
            }
        }
    }

    private static func write(_ data: Data, to destination: URL, at offset: Int) throws {
        let handle = try FileHandle(forWritingTo: destination)
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(offset))
        try handle.write(contentsOf: data)
    }

    /// One unranged GET, for a file whose length the listing did not report.
    private func fetchWhole(
        url: URL, path: String, to destination: URL,
        onBytes: @Sendable @escaping (Int64) -> Void
    ) async throws {
        var request = URLRequest(url: url)
        authorize(&request)
        let (data, response) = try await session.data(for: request)
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else { throw Failure.badStatus(path, code) }
        try data.write(to: destination)
        onBytes(Int64(data.count))
    }

    /// The token rides only to the configured endpoint. A redirect to the CDN is a
    /// different host, and URLSession drops the header there, which is what should happen.
    private func authorize(_ request: inout URLRequest) {
        guard let token, request.url?.host == base.host else { return }
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }

    /// - Parameter revision: the commit the listing came from, so a repository updated
    ///   mid-download cannot deliver shards from two different versions of the weights.
    private func url(repo: String, revision: String, path: String) -> URL {
        let encoded =
            path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? path
        return base.appending(path: "\(repo)/resolve/\(revision)/\(encoded)")
    }

    private static func ranges(of total: Int) -> [Range<Int>] {
        guard total > 0 else { return [0 ..< 0] }
        return stride(from: 0, to: total, by: chunkBytes).map { start in
            start ..< min(start + chunkBytes, total)
        }
    }
}

/// Chunks land out of order, so the caller is told how much arrived, not where from.
private actor ByteCounter {
    private var total: Int64 = 0
    private let report: @Sendable (Int64) -> Void

    init(_ report: @Sendable @escaping (Int64) -> Void) {
        self.report = report
    }

    func add(_ bytes: Int64) {
        total += bytes
        report(total)
    }
}
