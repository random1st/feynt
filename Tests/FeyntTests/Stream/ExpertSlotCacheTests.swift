import Foundation
import Metal
import Testing
@testable import FeyntStream

/// A tiny two-layer "checkpoint": nine stacked tensors per layer, `experts` deep, each
/// expert's slice filled with a byte pattern that names layer, part and expert.
private struct Fixture {
    let directory: URL
    let experts = 6
    let partBytes = [64, 16, 16, 64, 16, 16, 48, 8, 8]

    init() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("stream-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var header: [String: Any] = [:]
        var blob = Data()
        for layer in 0 ..< 2 {
            for (p, part) in ExpertSlotLayout.parts.enumerated() {
                let start = blob.count
                for e in 0 ..< experts {
                    blob.append(Data(repeating: Self.mark(layer, p, e), count: partBytes[p]))
                }
                header["layers.\(layer).mlp.switch_mlp.\(part)"] = [
                    "dtype": "U8", "shape": [experts, partBytes[p]], "data_offsets": [start, blob.count],
                ]
            }
        }
        header["layers.0.input_layernorm.weight"] = ["dtype": "U8", "shape": [32], "data_offsets": [blob.count, blob.count + 32]]
        blob.append(Data((0 ..< 32).map { UInt8($0) }))
        let json = try JSONSerialization.data(withJSONObject: header)
        var file = Data()
        var length = UInt64(json.count)
        file.append(Data(bytes: &length, count: 8))
        file.append(json)
        file.append(blob)
        try file.write(to: directory.appendingPathComponent("model.safetensors"))
    }

    static func mark(_ layer: Int, _ part: Int, _ expert: Int) -> UInt8 {
        UInt8(layer * 100 + part * 10 + expert)
    }

    func parts(_ index: SafetensorsIndex, layer: Int) -> [TensorRef] {
        ExpertSlotLayout.parts.map { index["layers.\(layer).mlp.switch_mlp.\($0)"]! }
    }

    func remove() { try? FileManager.default.removeItem(at: directory) }
}

@Suite struct ExpertSlotCacheTests {
    let device = MTLCreateSystemDefaultDevice()!

    @Test func indexReadsOffsetsAndShapes() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let index = try SafetensorsIndex(directory: fixture.directory)
        #expect(index.tensors.count == 19)
        let ref = try index.require("layers.1.mlp.switch_mlp.down_proj.weight")
        #expect(ref.shape == [6, 48] && ref.byteCount == 6 * 48)
    }

    @Test func aSlotHoldsExactlyTheExpertsSlicesOfAllNineParts() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let index = try SafetensorsIndex(directory: fixture.directory)
        let cache = try ExpertSlotCache(parts: fixture.parts(index, layer: 1), experts: 6, slotCount: 3,
                                        device: device, files: FileTable())
        let plan = try cache.load([4, 1])
        for (n, expert) in plan.experts.enumerated() {
            let slot = cache.buffer.contents().advanced(by: cache.offset(ofSlot: plan.slots[n]))
            for p in 0 ..< 9 {
                let bytes = UnsafeRawBufferPointer(
                    start: slot.advanced(by: cache.layout.partOffsets[p]), count: fixture.partBytes[p])
                #expect(bytes.allSatisfy { $0 == Fixture.mark(1, p, expert) })
            }
        }
    }

    @Test func hitsKeepTheirSlotAndTheLeastRecentlyUsedIsEvicted() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let index = try SafetensorsIndex(directory: fixture.directory)
        let cache = try ExpertSlotCache(parts: fixture.parts(index, layer: 0), experts: 6, slotCount: 3,
                                        device: device, files: FileTable())
        let first = try cache.load([0, 1, 2])
        #expect(first.misses.count == 3)
        _ = try cache.load([1])              // 0 is now the least recently used
        let third = try cache.load([3, 1])
        #expect(third.hits == 1)
        #expect(third.slots[1] == first.slots[1])
        #expect(third.slots[0] == first.slots[0])   // 3 took 0's slot
    }

    @Test func slotsQueuedWorkStillReadsAreNotReused() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let index = try SafetensorsIndex(directory: fixture.directory)
        let cache = try ExpertSlotCache(parts: fixture.parts(index, layer: 0), experts: 6, slotCount: 3,
                                        device: device, files: FileTable())
        let first = try cache.load([0, 1, 2])
        let busy = Set([first.slots[0], first.slots[1]])
        let next = try #require(cache.plan([5], avoiding: busy))
        #expect(next.slots[0] == first.slots[2])
        #expect(cache.plan([3, 4], avoiding: busy) == nil)   // only one slot is free
    }

    @Test func residentWeightsLoadByteForByte() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let index = try SafetensorsIndex(directory: fixture.directory)
        let resident = try ResidentWeights(index: index, include: { !$0.contains(".switch_mlp.") },
                                           device: device, files: FileTable())
        let bytes = try resident.bytes("layers.0.input_layernorm.weight")
        #expect(Array(bytes) == (0 ..< 32).map { UInt8($0) })
        #expect(resident.entries.count == 1)
    }
}
