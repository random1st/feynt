import Darwin
import Foundation
import Metal

/// The tensors that stay in memory for the whole run: everything but the routed experts.
///
/// For Qwen3.6-35B-A3B that is 1.39 GB - attention and DeltaNet projections, norms, the
/// router, the shared experts, the embedding and the head - against 18.1 GB of experts. They
/// are read once into one page-aligned allocation wrapped as a Metal buffer, each tensor at
/// a 256-byte aligned offset, so kernels bind `buffer` plus `offset` and nothing is copied
/// again.
public final class ResidentWeights: @unchecked Sendable {
    public struct Entry: Sendable {
        public let offset: Int
        public let ref: TensorRef
    }

    public let buffer: MTLBuffer
    public let entries: [String: Entry]
    public var byteCount: Int { buffer.length }

    public enum Failure: Error, CustomStringConvertible {
        case allocation(Int32)
        case wrap(Int)
        case read(String, Int32)
        case missing(String)

        public var description: String {
            switch self {
            case .allocation(let e): return "could not allocate resident weights (errno \(e))"
            case .wrap(let bytes): return "could not wrap \(bytes) bytes of weights in a Metal buffer"
            case .read(let name, let e): return "reading \(name) failed (errno \(e))"
            case .missing(let name): return "resident weights have no tensor \(name)"
            }
        }
    }

    /// - Parameter include: which tensor names to keep resident.
    public init(index: SafetensorsIndex, include: (String) -> Bool, device: MTLDevice,
                files: FileTable) throws {
        let names = index.tensors.keys.filter(include).sorted()
        var layoutOffsets: [String: Int] = [:]
        var cursor = 0
        for name in names {
            layoutOffsets[name] = cursor
            cursor += (index.tensors[name]!.byteCount + 255) / 256 * 256
        }
        let length = max(16384, (cursor + 16383) / 16384 * 16384)
        var raw: UnsafeMutableRawPointer?
        let status = posix_memalign(&raw, 16384, length)
        guard status == 0, let pointer = raw else { throw Failure.allocation(status) }
        nonisolated(unsafe) let owned = pointer
        guard let wrapped = device.makeBuffer(
            bytesNoCopy: pointer, length: length, options: .storageModeShared,
            deallocator: { _, _ in free(owned) })
        else {
            free(pointer)
            throw Failure.wrap(length)
        }
        var entries: [String: Entry] = [:]
        let errors = NSLock()
        nonisolated(unsafe) var firstError: Error?
        let refs = names.map { ($0, index.tensors[$0]!, layoutOffsets[$0]!) }
        let descriptors = try refs.map { try files.descriptor(for: $0.1.file) }
        DispatchQueue.concurrentPerform(iterations: refs.count) { i in
            let (name, ref, offset) = refs[i]
            let destination = pointer.advanced(by: offset)
            var filled = 0
            while filled < ref.byteCount {
                let n = pread(descriptors[i], destination.advanced(by: filled), ref.byteCount - filled,
                              off_t(ref.offset) + off_t(filled))
                if n <= 0 {
                    errors.lock()
                    if firstError == nil { firstError = Failure.read(name, n < 0 ? errno : 0) }
                    errors.unlock()
                    return
                }
                filled += n
            }
        }
        if let firstError { throw firstError }
        for (name, ref, offset) in refs { entries[name] = Entry(offset: offset, ref: ref) }
        buffer = wrapped
        self.entries = entries
    }

    public func entry(_ name: String) throws -> Entry {
        guard let entry = entries[name] else { throw Failure.missing(name) }
        return entry
    }

    /// The bytes of one tensor, for tests and CPU-side checks.
    public func bytes(_ name: String) throws -> UnsafeRawBufferPointer {
        let entry = try entry(name)
        return UnsafeRawBufferPointer(start: buffer.contents().advanced(by: entry.offset),
                                      count: entry.ref.byteCount)
    }
}
