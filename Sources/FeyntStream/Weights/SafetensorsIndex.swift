import Foundation

/// Where one tensor of a safetensors checkpoint lives on disk.
public struct TensorRef: Sendable, Equatable {
    public let file: URL
    /// Absolute byte offset of the tensor's data in `file`.
    public let offset: UInt64
    public let byteCount: Int
    public let dtype: String
    public let shape: [Int]
}

/// The tensors of a sharded safetensors checkpoint, read from the shard headers only.
///
/// Nothing is loaded here: the low-memory runtime reads each tensor - or one expert's slice
/// of a stacked tensor - from its byte range when it needs it.
public struct SafetensorsIndex: Sendable {
    public let tensors: [String: TensorRef]

    public enum Failure: Error, CustomStringConvertible {
        case unreadable(URL)
        case missing(String)

        public var description: String {
            switch self {
            case .unreadable(let url): return "\(url.lastPathComponent) is not a readable safetensors file"
            case .missing(let name): return "the checkpoint has no tensor \(name)"
            }
        }
    }

    public init(directory: URL) throws {
        let shards = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }
        var tensors: [String: TensorRef] = [:]
        for shard in shards {
            for (name, ref) in try Self.read(shard) { tensors[name] = ref }
        }
        self.tensors = tensors
    }

    public init(tensors: [String: TensorRef]) {
        self.tensors = tensors
    }

    public subscript(_ name: String) -> TensorRef? { tensors[name] }

    public func require(_ name: String) throws -> TensorRef {
        guard let ref = tensors[name] else { throw Failure.missing(name) }
        return ref
    }

    /// The header of one shard: 8-byte little-endian length, then JSON. Offsets in the JSON
    /// are relative to the end of the header.
    static func read(_ file: URL) throws -> [String: TensorRef] {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        guard let lengthBytes = try handle.read(upToCount: 8), lengthBytes.count == 8 else {
            throw Failure.unreadable(file)
        }
        let headerLength = lengthBytes.withUnsafeBytes { Int($0.loadUnaligned(as: UInt64.self)) }
        guard headerLength > 0, headerLength < 100_000_000,
            let headerData = try handle.read(upToCount: headerLength),
            let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else { throw Failure.unreadable(file) }
        let base = UInt64(8 + headerLength)
        var out: [String: TensorRef] = [:]
        for (name, value) in header where name != "__metadata__" {
            guard let entry = value as? [String: Any],
                let dtype = entry["dtype"] as? String,
                let shape = entry["shape"] as? [Int],
                let offsets = entry["data_offsets"] as? [Int], offsets.count == 2
            else { continue }
            out[name] = TensorRef(
                file: file, offset: base + UInt64(offsets[0]),
                byteCount: offsets[1] - offsets[0], dtype: dtype, shape: shape)
        }
        return out
    }
}
