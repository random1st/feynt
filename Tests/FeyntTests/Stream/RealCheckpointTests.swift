import Foundation
import Metal
import Testing
@testable import FeyntStream

/// Checks against the real Qwen3.6-35B-A3B checkpoint when it is on this machine; skipped
/// otherwise. What a synthetic file cannot show: that the names, shapes and stacking the
/// runtime assumes are the ones the published weights actually have.
@Suite struct RealCheckpointTests {
    static let directory = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
        "Library/Application Support/Feynt/models/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit")
    static var present: Bool { FileManager.default.fileExists(atPath: directory.appendingPathComponent("config.json").path) }

    @Test(.enabled(if: present)) func everyLayerHasNineStackedExpertTensorsOfTheExpectedSize() throws {
        let index = try SafetensorsIndex(directory: Self.directory)
        for layer in 0 ..< 40 {
            let parts = try ExpertSlotLayout.parts.map {
                try index.require("language_model.model.layers.\(layer).mlp.switch_mlp.\($0)")
            }
            #expect(parts.allSatisfy { $0.shape.first == 256 })
            let layout = ExpertSlotLayout(partBytes: parts.map { $0.byteCount / 256 })
            #expect(layout.partBytes.reduce(0, +) == 1_769_472)
        }
    }

    @Test(.enabled(if: present)) func residentTensorsAreTheOneAndAHalfGigabytesOutsideTheExperts() throws {
        let index = try SafetensorsIndex(directory: Self.directory)
        let resident = index.tensors.filter { Qwen35Names.isResident($0.key) }
        let bytes = resident.values.reduce(0) { $0 + $1.byteCount }
        #expect(bytes > 1_300_000_000 && bytes < 1_450_000_000)
    }

    @Test(.enabled(if: present)) func eightExpertsLoadFromTheCheckpointQuickly() throws {
        let index = try SafetensorsIndex(directory: Self.directory)
        let parts = try ExpertSlotLayout.parts.map {
            try index.require("language_model.model.layers.17.mlp.switch_mlp.\($0)")
        }
        let cache = try ExpertSlotCache(parts: parts, experts: 256, slotCount: 16,
                                        device: MTLCreateSystemDefaultDevice()!, files: FileTable())
        let started = Date()
        let plan = try cache.load([3, 41, 77, 128, 129, 200, 201, 255])
        let elapsed = Date().timeIntervalSince(started)
        #expect(plan.misses.count == 8)
        // Byte check of one part against a direct read of the same range.
        let ref = parts[6]
        let handle = try FileHandle(forReadingFrom: ref.file)
        try handle.seek(toOffset: ref.offset + UInt64(200 * cache.layout.partBytes[6]))
        let direct = try handle.read(upToCount: cache.layout.partBytes[6])!
        let slot = cache.buffer.contents().advanced(by: cache.offset(ofSlot: plan.slots[5]) + cache.layout.partOffsets[6])
        #expect(Data(bytes: slot, count: direct.count) == direct)
        print("8 cold experts in \(String(format: "%.1f", elapsed * 1000)) ms")
    }
}
