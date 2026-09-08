import Foundation

/// Filesystem locations the app owns.
enum Paths {
    static var home: URL { FileManager.default.homeDirectoryForCurrentUser }

    /// Where models this app downloads are installed.
    static var modelsRoot: URL {
        home.appending(path: "Library/Application Support/Feynt/models", directoryHint: .isDirectory)
    }

    static var logDirectory: URL {
        home.appending(path: "Library/Logs/Feynt", directoryHint: .isDirectory)
    }

    static var logFile: URL {
        logDirectory.appending(path: "feynt.log")
    }

    static func ensureDirectory(_ url: URL) {
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    /// Free space on the volume holding the models root, in bytes.
    static func freeDiskBytes() -> Int64? {
        ensureDirectory(modelsRoot)
        let values = try? modelsRoot.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    /// Physical RAM, used by the wizard to warn before a 16 GB model is loaded.
    static func physicalMemoryBytes() -> Int64 {
        Int64(ProcessInfo.processInfo.physicalMemory)
    }

    static func directorySize(_ url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles])
        else { return 0 }
        var total: Int64 = 0
        for case let item as URL in enumerator {
            let size = (try? item.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            total += Int64(size)
        }
        return total
    }

    static func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useGB, .useMB]
        // The formatter localises its units from the system, which put Russian "ГБ" next to
        // an otherwise English interface. The UI is English, so the units are too.
        formatter.formattingContext = .standalone
        let text = formatter.string(fromByteCount: bytes)
        return text.replacingOccurrences(of: "ГБ", with: "GB")
            .replacingOccurrences(of: "МБ", with: "MB")
            .replacingOccurrences(of: "КБ", with: "KB")
            .replacingOccurrences(of: "байт", with: "bytes")
    }
}

/// Minimal append-only log so the "Open log" menu item has something to show.
enum AppLog {
    private static let queue = DispatchQueue(label: "feynt.log")

    static func write(_ message: String) {
        queue.async {
            Paths.ensureDirectory(Paths.logDirectory)
            let stamp = ISO8601DateFormatter().string(from: Date())
            let line = "[\(stamp)] \(message)\n"
            guard let data = line.data(using: .utf8) else { return }
            let path = Paths.logFile
            if let handle = try? FileHandle(forWritingTo: path) {
                defer { try? handle.close() }
                _ = try? handle.seekToEnd()
                try? handle.write(contentsOf: data)
            } else {
                try? data.write(to: path)
            }
        }
    }
}
