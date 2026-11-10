import Darwin
import Foundation
import Metal

/// Where the nine parts of one routed expert sit inside a cache slot.
///
/// A Qwen3.5-MoE checkpoint stores each projection's experts stacked, `[expert, out, in/8]`,
/// so one expert's slice of each of the nine tensors (gate, up, down × weight, scales,
/// biases) is a contiguous byte range. The slot holds them back to back in this order, at
/// offsets that are the same for every expert, and the kernels read them from there.
public struct ExpertSlotLayout: Sendable, Equatable {
    public static let parts = [
        "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
        "up_proj.weight", "up_proj.scales", "up_proj.biases",
        "down_proj.weight", "down_proj.scales", "down_proj.biases",
    ]

    /// Byte size of one expert's slice of each part, in `parts` order.
    public let partBytes: [Int]
    /// Offset of each part inside a slot.
    public let partOffsets: [Int]
    /// Slot size, rounded up to a 16 KB page so slots never share a page.
    public let slotStride: Int

    public init(partBytes: [Int]) {
        precondition(partBytes.count == Self.parts.count)
        self.partBytes = partBytes
        var offsets: [Int] = []
        var cursor = 0
        for bytes in partBytes {
            offsets.append(cursor)
            cursor += (bytes + 15) / 16 * 16
        }
        partOffsets = offsets
        slotStride = (cursor + 16383) / 16384 * 16384
    }
}

/// Which slot holds which requested expert, and which of them still have to be read.
public struct ExpertCachePlan: Sendable, Equatable {
    public let experts: [Int]
    public let slots: [Int]
    /// Indices into `experts` that were not resident and must be read before use.
    public let misses: [Int]
    public var hits: Int { experts.count - misses.count }
}

/// One MoE layer's routed experts, cached in a fixed number of slots and read from the
/// checkpoint on a miss.
///
/// The slots live in one page-aligned allocation wrapped as a single Metal buffer, so the
/// GPU reads an expert at `slot * slotStride` with no copy. A miss is nine `pread`s from the
/// stacked tensors straight into the slot; misses of one step are read in parallel, which
/// on this machine's SSD reaches about 6 GB/s against 3 in one thread.
///
/// Eviction is least-recently-used. TurboFieldfare evicts least-frequently-used; replaying
/// this model's routing on code, a log summary and prose, LRU hit 58-70% at 28 slots and
/// 83-92% at 85, and LFU never did better - up to 21 points worse on the log. The planning
/// shape - assign hits first, protect slots queued GPU work still reads, then fill the
/// misses from the evictable rest - follows TurboFieldfare's `PreadExpertStreamer`
/// (Apache-2.0, see Vendor/TurboFieldfare/LICENSE).
public final class ExpertSlotCache: @unchecked Sendable {
    public let layout: ExpertSlotLayout
    public let slotCount: Int
    public let buffer: MTLBuffer

    private let base: UnsafeMutableRawPointer
    /// For each part: the file and the absolute offset of expert 0's slice.
    private let sources: [(fd: Int32, offset: UInt64)]
    /// Held so the descriptors in `sources` stay open as long as this cache can read.
    private let files: FileTable
    private var slotExpert: [Int]
    private var slotLastUse: [Int]
    /// Inverse of `slotExpert` for resident experts, -1 otherwise: hits in O(1).
    private var expertSlot: [Int]
    private var clock = 0
    private let lock = NSLock()

    public private(set) var totalHits = 0
    public private(set) var totalMisses = 0

    public enum Failure: Error, CustomStringConvertible {
        case allocation(Int32)
        case wrap
        case read(errno: Int32, part: String)
        case shortRead(part: String)
        case layout(String)

        public var description: String {
            switch self {
            case .allocation(let e): return "could not allocate expert slots (errno \(e))"
            case .wrap: return "could not wrap the expert slots in a Metal buffer"
            case .read(let e, let part): return "reading \(part) failed (errno \(e))"
            case .shortRead(let part): return "the checkpoint ended inside \(part)"
            case .layout(let detail): return detail
            }
        }
    }

    /// - Parameters:
    ///   - parts: the layer's nine stacked tensors, in `ExpertSlotLayout.parts` order.
    ///   - files: open read descriptors by file, shared across layers.
    public init(parts: [TensorRef], experts: Int, slotCount: Int, device: MTLDevice,
                files: FileTable) throws {
        guard parts.count == ExpertSlotLayout.parts.count, slotCount > 0, experts > 0 else {
            throw Failure.layout("an expert layer needs nine stacked tensors and at least one slot")
        }
        var partBytes: [Int] = []
        for (ref, name) in zip(parts, ExpertSlotLayout.parts) {
            guard ref.shape.first == experts, ref.byteCount % experts == 0 else {
                throw Failure.layout("\(name) is not stacked over \(experts) experts: \(ref.shape)")
            }
            partBytes.append(ref.byteCount / experts)
        }
        layout = ExpertSlotLayout(partBytes: partBytes)
        self.slotCount = slotCount
        self.files = files
        sources = try parts.map { (try files.descriptor(for: $0.file), $0.offset) }

        let length = layout.slotStride * slotCount
        var raw: UnsafeMutableRawPointer?
        let status = posix_memalign(&raw, 16384, length)
        guard status == 0, let pointer = raw else { throw Failure.allocation(status) }
        memset(pointer, 0, length)
        nonisolated(unsafe) let owned = pointer
        guard let wrapped = device.makeBuffer(
            bytesNoCopy: pointer, length: length, options: .storageModeShared,
            deallocator: { _, _ in free(owned) })
        else {
            free(pointer)
            throw Failure.wrap
        }
        base = pointer
        buffer = wrapped
        slotExpert = Array(repeating: -1, count: slotCount)
        slotLastUse = Array(repeating: 0, count: slotCount)
        expertSlot = Array(repeating: -1, count: experts)
    }

    /// Byte offset of `slot` in `buffer`.
    public func offset(ofSlot slot: Int) -> Int { slot * layout.slotStride }

    /// Assigns a slot to every requested expert without reading anything. Hits keep their
    /// slot; misses take the least recently used slots that are neither hit nor `avoiding`.
    /// Returns nil when there are not enough evictable slots.
    public func plan(_ experts: [Int], avoiding: Set<Int> = []) -> ExpertCachePlan? {
        // On the decode critical path: the GPU idles from the router until this returns,
        // so no sorting, closures or per-call collections beyond the result.
        lock.lock()
        defer { lock.unlock() }
        var slots = Array(repeating: -1, count: experts.count)
        var reserved = Array(repeating: false, count: slotCount)
        for (i, expert) in experts.enumerated() {
            let slot = expertSlot[expert]
            if slot >= 0 && !reserved[slot] {
                slots[i] = slot
                reserved[slot] = true
            }
        }
        for slot in avoiding where slot >= 0 && slot < slotCount { reserved[slot] = true }
        var misses: [Int] = []
        for i in experts.indices where slots[i] == -1 { misses.append(i) }
        clock += 1
        for slot in slots where slot >= 0 { slotLastUse[slot] = clock }
        for i in misses {
            var victim = -1
            for slot in 0 ..< slotCount where !reserved[slot] {
                if victim < 0 || slotLastUse[slot] < slotLastUse[victim] { victim = slot }
            }
            guard victim >= 0 else { return nil }
            reserved[victim] = true
            if slotExpert[victim] >= 0 { expertSlot[slotExpert[victim]] = -1 }
            slots[i] = victim
            slotExpert[victim] = -1  // in flight until read
            slotLastUse[victim] = clock
        }
        totalHits += experts.count - misses.count
        totalMisses += misses.count
        return ExpertCachePlan(experts: experts, slots: slots, misses: misses)
    }

    /// Reads the plan's misses into their slots and marks them resident. Each of the nine
    /// parts of each miss is its own pread, all in parallel: a layer usually misses one or
    /// two experts, and nine reads in flight copy faster than one and keep the SSD queue deep.
    public func fill(_ plan: ExpertCachePlan) throws {
        let parts = ExpertSlotLayout.parts.count
        let errors = NSLock()
        nonisolated(unsafe) var firstError: Error?
        DispatchQueue.concurrentPerform(iterations: plan.misses.count * parts) { n in
            let i = plan.misses[n / parts]
            do {
                try self.read(expert: plan.experts[i], part: n % parts, into: plan.slots[i])
            } catch {
                errors.lock()
                if firstError == nil { firstError = error }
                errors.unlock()
            }
        }
        if let firstError { throw firstError }
        lock.lock()
        for i in plan.misses {
            slotExpert[plan.slots[i]] = plan.experts[i]
            expertSlot[plan.experts[i]] = plan.slots[i]
        }
        lock.unlock()
    }

    /// Plans and fills in one call; the common decode path.
    public func load(_ experts: [Int], avoiding: Set<Int> = []) throws -> ExpertCachePlan {
        guard let plan = plan(experts, avoiding: avoiding) else {
            throw Failure.layout("\(experts.count) experts do not fit in \(slotCount) slots")
        }
        try fill(plan)
        return plan
    }

    private func read(expert: Int, part: Int, into slot: Int) throws {
        let bytes = layout.partBytes[part]
        let source = sources[part]
        let fileOffset = source.offset + UInt64(expert * bytes)
        let destination = base.advanced(by: offset(ofSlot: slot) + layout.partOffsets[part])
        var filled = 0
        while filled < bytes {
            let n = pread(source.fd, destination.advanced(by: filled), bytes - filled,
                          off_t(fileOffset) + off_t(filled))
            if n < 0 { throw Failure.read(errno: errno, part: ExpertSlotLayout.parts[part]) }
            if n == 0 { throw Failure.shortRead(part: ExpertSlotLayout.parts[part]) }
            filled += n
        }
    }
}

/// Read descriptors shared by every layer's cache, one per checkpoint file.
public final class FileTable: @unchecked Sendable {
    private var descriptors: [URL: Int32] = [:]
    private let lock = NSLock()
    /// Read past the OS file cache (`F_NOCACHE`): every miss goes to the SSD, as on a Mac
    /// with no memory to spare for caching. Off by default, because cached pages are not
    /// the process's memory and the system takes them back when it needs them.
    public let bypassCache: Bool

    public init(bypassCache: Bool = false) {
        self.bypassCache = bypassCache
    }

    public func descriptor(for file: URL) throws -> Int32 {
        lock.lock()
        defer { lock.unlock() }
        if let fd = descriptors[file] { return fd }
        let fd = open(file.path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else { throw ExpertSlotCache.Failure.read(errno: errno, part: file.lastPathComponent) }
        if bypassCache { _ = fcntl(fd, F_NOCACHE, 1) }
        descriptors[file] = fd
        return fd
    }

    deinit {
        for fd in descriptors.values { close(fd) }
    }
}
