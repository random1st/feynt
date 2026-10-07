import CoreImage
import Foundation

/// Files an agent names for the model to read, loaded by Feynt itself.
///
/// The point is the agent's context: handing over a path keeps a 30k-token log out of the
/// caller's window entirely, and the model gets the contents up front instead of having to
/// decide to go and read them - which a 2B model, asked about a file, often did not.
enum FileInputs {
    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// One file is cut here; the model's own context limit bounds the total.
    static let maxBytesPerFile = 2_000_000
    static let maxFiles = 20

    /// Text files become one block for the prompt, each fenced with its path; images go to
    /// the vision path. Paths are checked like the file tools check theirs: inside the
    /// workspace when one is given, absolute otherwise, and never a credential location.
    static func load(_ paths: [String], workspace: Workspace?) throws -> (text: String, images: [Data]) {
        guard paths.count <= maxFiles else {
            throw Failure(message: "\(paths.count) files; at most \(maxFiles) per request")
        }
        var blocks: [String] = []
        var images: [Data] = []
        for path in paths {
            let url = try resolve(path, workspace: workspace)
            let shown = workspace?.display(url) ?? url.path
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
                !isDirectory.boolValue
            else { throw Failure(message: "\(path) is not a file") }
            let data = try Data(contentsOf: url, options: .mappedIfSafe)
            if isImage(data) {
                images.append(try ImageInput.validated(data))
                continue
            }
            guard !data.prefix(8192).contains(0), var text = String(data: data, encoding: .utf8) else {
                throw Failure(message: "\(shown) is neither UTF-8 text nor an image")
            }
            if data.count > maxBytesPerFile {
                text = String(text.utf8.prefix(maxBytesPerFile)) ?? String(text.prefix(maxBytesPerFile / 2))
                text += "\n[cut at \(maxBytesPerFile / 1_000_000) MB of \(data.count / 1_000_000) MB]"
            }
            blocks.append("<file path=\"\(shown)\">\n\(text)\n</file>")
        }
        return (blocks.joined(separator: "\n\n"), images)
    }

    private static func resolve(_ path: String, workspace: Workspace?) throws -> URL {
        if let workspace { return try workspace.resolve(path) }
        let expanded = (path as NSString).expandingTildeInPath
        guard expanded.hasPrefix("/") else {
            throw Failure(message: "\(path) is relative; give an absolute path, or a workspace to resolve it in")
        }
        let url = URL(fileURLWithPath: expanded).standardizedFileURL.resolvingSymlinksInPath()
        if let secret = Workspace.secret(url) {
            throw Failure(message: "\(path) is a credential location (\(secret)); Feynt does not read those")
        }
        return url
    }

    /// By the bytes, not the extension: an agent passes whatever path it has.
    static func isImage(_ data: Data) -> Bool {
        let head = [UInt8](data.prefix(12))
        guard head.count >= 4 else { return false }
        if head.starts(with: [0x89, 0x50, 0x4E, 0x47]) { return true }               // PNG
        if head.starts(with: [0xFF, 0xD8, 0xFF]) { return true }                      // JPEG
        if head.starts(with: [0x47, 0x49, 0x46, 0x38]) { return true }               // GIF
        if head.count >= 12, head[0 ..< 4] == [0x52, 0x49, 0x46, 0x46],
            head[8 ..< 12] == [0x57, 0x45, 0x42, 0x50] { return true }                // WebP
        if head.count >= 12, head[4 ..< 8] == [0x66, 0x74, 0x79, 0x70] {
            return CIImage(data: data) != nil                                          // HEIC/AVIF
        }
        return false
    }
}
