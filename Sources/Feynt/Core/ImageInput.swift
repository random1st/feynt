import CoreImage
import Foundation

/// Images as the protocols carry them, turned into the bytes an engine turn holds.
enum ImageInput {
    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// A decoded image larger than this is refused. The vision tower resizes everything to
    /// a few hundred thousand pixels anyway; past this a request is a mistake or a stress.
    static let maxBytes = 20_000_000

    /// `data:image/…;base64,…` to its bytes, checked to be an image Core Image can read, so
    /// a bad attachment fails here with a reason instead of vanishing inside the engine.
    static func decode(_ url: String) throws -> Data {
        guard url.hasPrefix("data:") else {
            throw Failure(message: "only inline images are accepted (a data: URL with base64); "
                + "Feynt does not fetch image URLs")
        }
        guard let comma = url.firstIndex(of: ","), url[..<comma].hasSuffix(";base64") else {
            throw Failure(message: "the image data URL is not base64-encoded")
        }
        return try decode(base64: String(url[url.index(after: comma)...]))
    }

    /// Plain base64, as MCP and A2A carry image bytes.
    static func decode(base64: String) throws -> Data {
        guard let data = Data(base64Encoded: base64, options: .ignoreUnknownCharacters) else {
            throw Failure(message: "the image is not valid base64")
        }
        return try validated(data)
    }

    /// `data` if it is an image Core Image can read and within ``maxBytes``.
    static func validated(_ data: Data) throws -> Data {
        guard data.count <= maxBytes else {
            throw Failure(message: "the image is \(data.count / 1_000_000) MB; the limit is "
                + "\(maxBytes / 1_000_000) MB")
        }
        guard CIImage(data: data) != nil else {
            throw Failure(message: "the attachment is not an image Feynt can read")
        }
        return data
    }
}
