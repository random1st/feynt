import Foundation
import Metal

/// Qwen3.6-35B-A3B decode on the GPU with routed experts read from SSD.
///
/// One step: a command buffer runs from the previous layer's experts through this layer's
/// mixer and router; the CPU waits for it, reads the eight chosen experts, loads the
/// misses into this layer's slot cache, and encodes the next buffer. Forty waits a token,
/// the TurboFieldfare shape, because which weights the GPU needs next is only known once
/// the router has run.
public final class Qwen35Runtime: @unchecked Sendable {
    public struct Config: Sendable {
        public var slotsPerLayer: Int
        public var maxContext: Int
        public init(slotsPerLayer: Int, maxContext: Int = 8192) {
            self.slotsPerLayer = slotsPerLayer
            self.maxContext = maxContext
        }
    }

    // Model shape (Qwen3.6-35B-A3B; checked against config.json at load).
    let hidden = 2048, layers = 40, experts = 256, topK = 8, moeInter = 512
    let hk = 16, hv = 32, dk = 128, dv = 128, convDim = 8192
    let heads = 16, kvHeads = 2, headDim = 256, rotary = 64
    let ropeBase: Float = 10_000_000, eps: Float = 1e-6, vocab = 248_320

    public let device: MTLDevice
    let queue: MTLCommandQueue
    let resident: ResidentWeights
    let caches: [ExpertSlotCache]
    let config: Config
    private var pipelines: [String: MTLComputePipelineState] = [:]

    // Activations and state.
    private let x, h, mix, moe, shared, sharedGate: MTLBuffer
    private let qkv, z, a, b, u, qn, kn, g, beta, y, o: MTLBuffer
    private let qg, q, gate, kBuf, vBuf, attn: MTLBuffer
    private let routerLogits, routeIdx, routeWts, eg, eu, eh, ey, sg, su, sh: MTLBuffer
    private let logits, argmaxOut: MTLBuffer
    private var convState: [Int: MTLBuffer] = [:]
    private var recState: [Int: MTLBuffer] = [:]
    private var keyCache: [Int: MTLBuffer] = [:]
    private var valueCache: [Int: MTLBuffer] = [:]
    public private(set) var position = 0

    /// Where a step's time goes: GPU execution, waiting for it, reading experts, encoding.
    public struct Profile: Sendable {
        public var steps = 0
        public var gpu = 0.0, wait = 0.0, load = 0.0, total = 0.0
    }
    public private(set) var profile = Profile()

    public enum Failure: Error, CustomStringConvertible {
        case noDevice, kernel(String), shape(String), contextFull(Int)
        public var description: String {
            switch self {
            case .noDevice: return "no Metal device"
            case .kernel(let s): return "kernel: \(s)"
            case .shape(let s): return "unexpected model shape: \(s)"
            case .contextFull(let n): return "the context is full at \(n) tokens"
            }
        }
    }

    public init(directory: URL, config: Config) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw Failure.noDevice }
        self.device = device
        self.config = config
        queue = device.makeCommandQueue()!
        let files = FileTable()
        let index = try SafetensorsIndex(directory: directory)
        resident = try ResidentWeights(index: index, include: Qwen35Names.isResident, device: device, files: files)
        var caches: [ExpertSlotCache] = []
        for layer in 0 ..< 40 {
            let parts = try ExpertSlotLayout.parts.map { try index.require(Qwen35Names.expert(layer, $0)) }
            caches.append(try ExpertSlotCache(parts: parts, experts: 256, slotCount: config.slotsPerLayer,
                                              device: device, files: files))
        }
        self.caches = caches

        func f32(_ n: Int) -> MTLBuffer { device.makeBuffer(length: max(4, n * 4), options: .storageModeShared)! }
        x = f32(hidden); h = f32(hidden); mix = f32(hidden); moe = f32(hidden); shared = f32(hidden); sharedGate = f32(1)
        qkv = f32(convDim); z = f32(hv * dv); a = f32(hv); b = f32(hv); u = f32(convDim)
        qn = f32(hk * dk); kn = f32(hk * dk); g = f32(hv); beta = f32(hv); y = f32(hv * dv); o = f32(hv * dv)
        qg = f32(heads * headDim * 2); q = f32(heads * headDim); gate = f32(heads * headDim)
        kBuf = f32(kvHeads * headDim); vBuf = f32(kvHeads * headDim); attn = f32(heads * headDim)
        routerLogits = f32(experts); routeIdx = f32(topK); routeWts = f32(topK)
        eg = f32(topK * moeInter); eu = f32(topK * moeInter); eh = f32(topK * moeInter); ey = f32(topK * hidden)
        sg = f32(moeInter); su = f32(moeInter); sh = f32(moeInter)
        logits = f32(vocab); argmaxOut = f32(1)
        for layer in 0 ..< 40 {
            if Self.isAttention(layer) {
                keyCache[layer] = f32(kvHeads * config.maxContext * headDim)
                valueCache[layer] = f32(kvHeads * config.maxContext * headDim)
            } else {
                convState[layer] = f32(3 * convDim)
                recState[layer] = f32(hv * dv * dk)
            }
        }

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: Qwen35Kernels.source, options: nil)
        } catch {
            throw Failure.kernel(error.localizedDescription)
        }
        for name in ["qgemv", "embed_q4", "rmsnorm", "add_inplace", "conv_step", "delta_gates", "delta_step",
                     "gated_norm", "split_qgate", "rope_partial", "kv_append", "attend", "route_topk",
                     "swiglu", "gated_add", "zero", "argmax", "weighted_sum"] {
            guard let fn = library.makeFunction(name: name) else { throw Failure.kernel("missing \(name)") }
            pipelines[name] = try device.makeComputePipelineState(function: fn)
        }
    }

    public static func isAttention(_ layer: Int) -> Bool { (layer + 1) % 4 == 0 }

    /// Bytes this runtime holds for weights and expert slots (activations and caches aside).
    public var weightBytes: Int { resident.byteCount + caches.reduce(0) { $0 + $1.buffer.length } }

    public var hitRate: Double {
        let hits = caches.reduce(0) { $0 + $1.totalHits }, misses = caches.reduce(0) { $0 + $1.totalMisses }
        return hits + misses == 0 ? 0 : Double(hits) / Double(hits + misses)
    }

    public func resetProfile() { profile = Profile() }

    public func reset() {
        position = 0
        for buffer in convState.values { memset(buffer.contents(), 0, buffer.length) }
        for buffer in recState.values { memset(buffer.contents(), 0, buffer.length) }
    }

    /// Runs one token through the model at the current position and advances it. Returns the
    /// greedy next token when `logits` is requested, otherwise nil.
    @discardableResult
    public func step(token: Int, wantLogits: Bool) throws -> Int? {
        guard position < config.maxContext else { throw Failure.contextFull(config.maxContext) }
        let s0 = Date()
        defer { profile.total += Date().timeIntervalSince(s0); profile.steps += 1 }
        var cb = queue.makeCommandBuffer()!
        var enc = cb.makeComputeCommandEncoder()!
        let L = Qwen35Names.layer
        embed(enc, token: token)
        for layer in 0 ..< layers {
            let p = L(layer)
            try norm(enc, x, p + "input_layernorm.weight", h, dim: hidden)
            if Self.isAttention(layer) {
                try attention(enc, layer: layer, prefix: p + "self_attn.")
            } else {
                try deltaNet(enc, layer: layer, prefix: p + "linear_attn.")
            }
            add(enc, x, mix, hidden)
            try norm(enc, x, p + "post_attention_layernorm.weight", h, dim: hidden)
            try gemv(enc, p + "mlp.gate", h, routerLogits)
            dispatch(enc, "route_topk", [routerLogits, routeIdx, routeWts], u32: [UInt32(experts), UInt32(topK)],
                     threads: experts, groups: 1, perGroup: experts)
            enc.endEncoding()
            cb.commit()
            let w0 = Date()
            cb.waitUntilCompleted()
            profile.wait += Date().timeIntervalSince(w0)
            profile.gpu += cb.gpuEndTime - cb.gpuStartTime

            let idx = routeIdx.contents().bindMemory(to: Int32.self, capacity: topK)
            let wts = routeWts.contents().bindMemory(to: Float.self, capacity: topK)
            let chosen = (0 ..< topK).map { Int(idx[$0]) }
            let weights = (0 ..< topK).map { wts[$0] }
            let l0 = Date()
            let plan = try caches[layer].load(chosen)
            profile.load += Date().timeIntervalSince(l0)

            cb = queue.makeCommandBuffer()!
            enc = cb.makeComputeCommandEncoder()!
            try sharedExpert(enc, prefix: p + "mlp.")
            try routedExperts(enc, cache: caches[layer], slots: plan.slots, weights: weights)
            dispatch(enc, "gated_add", [moe, shared, sharedGate], u32: [UInt32(hidden)], threads: hidden)
            add(enc, x, moe, hidden)
        }
        var next: Int?
        if wantLogits {
            try norm(enc, x, Qwen35Names.prefix + "norm.weight", h, dim: hidden)
            try gemv(enc, "language_model.lm_head", h, logits)
            dispatch(enc, "argmax", [logits, argmaxOut], u32: [UInt32(vocab)], threads: 1024, groups: 1, perGroup: 1024)
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        profile.gpu += cb.gpuEndTime - cb.gpuStartTime
        if wantLogits { next = Int(argmaxOut.contents().load(as: Int32.self)) }
        position += 1
        return next
    }

    /// The last step's logits, for comparison against a reference.
    public func lastLogits() -> UnsafeBufferPointer<Float> {
        UnsafeBufferPointer(start: logits.contents().bindMemory(to: Float.self, capacity: vocab), count: vocab)
    }

    // MARK: - Benchmark

    /// GPU milliseconds per call of each block, timed in isolation (repeat calls in one
    /// command buffer). For finding where a step's GPU time goes.
    public func benchmark(repeats: Int = 40) throws -> [(String, Double)] {
        let plan = try caches[0].load(Array(0 ..< topK))
        let weights = [Float](repeating: 1 / Float(topK), count: topK)
        func time(_ body: (MTLComputeCommandEncoder) throws -> Void) throws -> Double {
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            for _ in 0 ..< repeats { try body(enc) }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) / Double(repeats) * 1000
        }
        var out: [(String, Double)] = []
        out.append(("empty command buffer", try time { _ in }))
        out.append(("lm_head", try time { try gemv($0, "language_model.lm_head", h, logits) }))
        out.append(("deltanet block", try time { try deltaNet($0, layer: 0, prefix: Qwen35Names.layer(0) + "linear_attn.") }))
        out.append(("attention block", try time { try attention($0, layer: 3, prefix: Qwen35Names.layer(3) + "self_attn.") }))
        out.append(("router + top-8", try time {
            try gemv($0, Qwen35Names.layer(0) + "mlp.gate", h, routerLogits)
            dispatch($0, "route_topk", [routerLogits, routeIdx, routeWts], u32: [UInt32(experts), UInt32(topK)],
                     threads: experts, groups: 1, perGroup: experts)
        }))
        out.append(("shared expert", try time { try sharedExpert($0, prefix: Qwen35Names.layer(0) + "mlp.") }))
        out.append(("8 routed experts", try time { try routedExperts($0, cache: caches[0], slots: plan.slots, weights: weights) }))
        out.append(("rmsnorm 2048", try time { try norm($0, x, Qwen35Names.layer(0) + "input_layernorm.weight", h, dim: hidden) }))
        return out
    }

    // MARK: - Blocks

    private func attention(_ enc: MTLComputeCommandEncoder, layer: Int, prefix: String) throws {
        try gemv(enc, prefix + "q_proj", h, qg)
        dispatch(enc, "split_qgate", [qg, q, gate], u32: [UInt32(headDim), UInt32(heads)], threads: heads * headDim)
        try gemv(enc, prefix + "k_proj", h, kBuf)
        try gemv(enc, prefix + "v_proj", h, vBuf)
        try norm(enc, q, prefix + "q_norm.weight", q, dim: headDim, heads: heads)
        try norm(enc, kBuf, prefix + "k_norm.weight", kBuf, dim: headDim, heads: kvHeads)
        rope(enc, q, heads: heads)
        rope(enc, kBuf, heads: kvHeads)
        dispatch(enc, "kv_append", [kBuf, vBuf, keyCache[layer]!, valueCache[layer]!],
                 u32: [UInt32(headDim), UInt32(config.maxContext), UInt32(position), UInt32(kvHeads)],
                 threads: kvHeads * headDim)
        let pso = pipelines["attend"]!
        enc.setComputePipelineState(pso)
        for (i, buffer) in [q, keyCache[layer]!, valueCache[layer]!, gate, attn].enumerated() {
            enc.setBuffer(buffer, offset: 0, index: i)
        }
        var d = UInt32(headDim), cap = UInt32(config.maxContext), t = UInt32(position + 1)
        var group = UInt32(heads / kvHeads), scale = 1 / Float(headDim).squareRoot()
        enc.setBytes(&d, length: 4, index: 5); enc.setBytes(&cap, length: 4, index: 6)
        enc.setBytes(&t, length: 4, index: 7); enc.setBytes(&group, length: 4, index: 8)
        enc.setBytes(&scale, length: 4, index: 9)
        enc.dispatchThreadgroups(MTLSize(width: heads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1))
        try gemv(enc, prefix + "o_proj", attn, mix)
    }

    private func deltaNet(_ enc: MTLComputeCommandEncoder, layer: Int, prefix: String) throws {
        try gemv(enc, prefix + "in_proj_qkv", h, qkv)
        try gemv(enc, prefix + "in_proj_z", h, z)
        try gemv(enc, prefix + "in_proj_b", h, b)
        try gemv(enc, prefix + "in_proj_a", h, a)
        let convW = try resident.entry(prefix + "conv1d.weight")
        enc.setComputePipelineState(pipelines["conv_step"]!)
        enc.setBuffer(qkv, offset: 0, index: 0)
        enc.setBuffer(convState[layer]!, offset: 0, index: 1)
        enc.setBuffer(resident.buffer, offset: convW.offset, index: 2)
        enc.setBuffer(u, offset: 0, index: 3)
        var c = UInt32(convDim)
        enc.setBytes(&c, length: 4, index: 4)
        threads(enc, convDim)
        // q and k, per head, weightless rmsnorm with the reference's scales.
        try normRaw(enc, u, 0, qn, dim: dk, heads: hk, weight: nil, scale: 1 / Float(dk))
        try normRaw(enc, u, hk * dk * 4, kn, dim: dk, heads: hk, weight: nil, scale: 1 / Float(dk).squareRoot())
        let aLog = try resident.entry(prefix + "A_log"), dt = try resident.entry(prefix + "dt_bias")
        enc.setComputePipelineState(pipelines["delta_gates"]!)
        enc.setBuffer(a, offset: 0, index: 0); enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(resident.buffer, offset: aLog.offset, index: 2)
        enc.setBuffer(resident.buffer, offset: dt.offset, index: 3)
        enc.setBuffer(g, offset: 0, index: 4); enc.setBuffer(beta, offset: 0, index: 5)
        var hvv = UInt32(hv)
        enc.setBytes(&hvv, length: 4, index: 6)
        threads(enc, hv)
        enc.setComputePipelineState(pipelines["delta_step"]!)
        enc.setBuffer(qn, offset: 0, index: 0); enc.setBuffer(kn, offset: 0, index: 1)
        enc.setBuffer(u, offset: 2 * hk * dk * 4, index: 2)
        enc.setBuffer(g, offset: 0, index: 3); enc.setBuffer(beta, offset: 0, index: 4)
        enc.setBuffer(recState[layer]!, offset: 0, index: 5); enc.setBuffer(y, offset: 0, index: 6)
        var dkk = UInt32(dk), dvv = UInt32(dv), ratio = UInt32(hv / hk)
        enc.setBytes(&dkk, length: 4, index: 7); enc.setBytes(&dvv, length: 4, index: 8); enc.setBytes(&ratio, length: 4, index: 9)
        enc.dispatchThreadgroups(MTLSize(width: dv / 4, height: hv, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        let nw = try resident.entry(prefix + "norm.weight")
        enc.setComputePipelineState(pipelines["gated_norm"]!)
        enc.setBuffer(y, offset: 0, index: 0); enc.setBuffer(z, offset: 0, index: 1)
        enc.setBuffer(resident.buffer, offset: nw.offset, index: 2); enc.setBuffer(o, offset: 0, index: 3)
        var dvd = UInt32(dv), e = eps
        enc.setBytes(&dvd, length: 4, index: 4); enc.setBytes(&e, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: hv, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        try gemv(enc, prefix + "out_proj", o, mix)
    }

    private func sharedExpert(_ enc: MTLComputeCommandEncoder, prefix: String) throws {
        try gemv(enc, prefix + "shared_expert.gate_proj", h, sg)
        try gemv(enc, prefix + "shared_expert.up_proj", h, su)
        dispatch(enc, "swiglu", [sg, su, sh], u32: [UInt32(moeInter)], threads: moeInter)
        try gemv(enc, prefix + "shared_expert.down_proj", sh, shared)
        try gemv(enc, prefix + "shared_expert_gate", h, sharedGate)
    }

    /// All eight routed experts in three batched GEMVs and one weighted sum.
    private func routedExperts(_ enc: MTLComputeCommandEncoder, cache: ExpertSlotCache, slots: [Int],
                               weights: [Float]) throws {
        let off = cache.layout.partOffsets
        func offsets(_ part: Int) -> [UInt32] { slots.map { UInt32(cache.offset(ofSlot: $0) + off[part]) } }
        let ones = [Float](repeating: 1, count: slots.count)
        batchedGemv(enc, base: cache.buffer, w: offsets(0), s: offsets(1), b: offsets(2), x: h, xStride: 0,
                    y: eg, yStride: moeInter, inDim: hidden, outDim: moeInter, bits: 4, alpha: ones)
        batchedGemv(enc, base: cache.buffer, w: offsets(3), s: offsets(4), b: offsets(5), x: h, xStride: 0,
                    y: eu, yStride: moeInter, inDim: hidden, outDim: moeInter, bits: 4, alpha: ones)
        dispatch(enc, "swiglu", [eg, eu, eh], u32: [UInt32(slots.count * moeInter)], threads: slots.count * moeInter)
        batchedGemv(enc, base: cache.buffer, w: offsets(6), s: offsets(7), b: offsets(8), x: eh, xStride: moeInter,
                    y: ey, yStride: hidden, inDim: moeInter, outDim: hidden, bits: 4, alpha: ones)
        enc.setComputePipelineState(pipelines["weighted_sum"]!)
        enc.setBuffer(ey, offset: 0, index: 0)
        var w = weights
        enc.setBytes(&w, length: 4 * w.count, index: 1)
        enc.setBuffer(moe, offset: 0, index: 2)
        var n = UInt32(hidden), c = UInt32(slots.count)
        enc.setBytes(&n, length: 4, index: 3); enc.setBytes(&c, length: 4, index: 4)
        threads(enc, hidden)
    }

    // MARK: - Primitives

    private func embed(_ enc: MTLComputeCommandEncoder, token: Int) {
        let n = "language_model.model.embed_tokens."
        let w = try! resident.entry(n + "weight"), s = try! resident.entry(n + "scales"), bb = try! resident.entry(n + "biases")
        enc.setComputePipelineState(pipelines["embed_q4"]!)
        enc.setBuffer(resident.buffer, offset: w.offset, index: 0)
        enc.setBuffer(resident.buffer, offset: s.offset, index: 1)
        enc.setBuffer(resident.buffer, offset: bb.offset, index: 2)
        enc.setBuffer(x, offset: 0, index: 3)
        var t = UInt32(token), d = UInt32(hidden)
        enc.setBytes(&t, length: 4, index: 4); enc.setBytes(&d, length: 4, index: 5)
        threads(enc, hidden)
    }

    /// GEMV over a resident quantized linear `name` (`.weight/.scales/.biases`); shapes and
    /// bit width come from the tensors.
    private func gemv(_ enc: MTLComputeCommandEncoder, _ name: String, _ input: MTLBuffer, _ output: MTLBuffer) throws {
        let w = try resident.entry(name + ".weight"), s = try resident.entry(name + ".scales"), bb = try resident.entry(name + ".biases")
        let outDim = w.ref.shape[0], inDim = s.ref.shape[1] * 64
        let bits = w.ref.shape[1] * 32 / inDim
        quantGemv(enc, w: (resident.buffer, w.offset), s: (resident.buffer, s.offset), b: (resident.buffer, bb.offset),
                  x: input, y: output, inDim: inDim, outDim: outDim, bits: bits)
    }

    private func quantGemv(_ enc: MTLComputeCommandEncoder, w: (MTLBuffer, Int), s: (MTLBuffer, Int), b: (MTLBuffer, Int),
                           x input: MTLBuffer, y output: MTLBuffer, inDim: Int, outDim: Int, bits: Int) {
        batchedGemv(enc, base: w.0, w: [UInt32(w.1)], s: [UInt32(s.1)], b: [UInt32(b.1)], x: input, xStride: 0,
                    y: output, yStride: 0, inDim: inDim, outDim: outDim, bits: bits, alpha: [1])
    }

    private func batchedGemv(_ enc: MTLComputeCommandEncoder, base: MTLBuffer, w: [UInt32], s: [UInt32], b: [UInt32],
                             x input: MTLBuffer, xStride: Int, y output: MTLBuffer, yStride: Int,
                             inDim: Int, outDim: Int, bits: Int, alpha: [Float], accumulate: Bool = false) {
        enc.setComputePipelineState(pipelines["qgemv"]!)
        enc.setBuffer(base, offset: 0, index: 0)
        var wo = w, so = s, bo = b, al = alpha
        enc.setBytes(&wo, length: 4 * wo.count, index: 1)
        enc.setBytes(&so, length: 4 * so.count, index: 2)
        enc.setBytes(&bo, length: 4 * bo.count, index: 3)
        enc.setBuffer(input, offset: 0, index: 4)
        enc.setBuffer(output, offset: 0, index: 5)
        var i = UInt32(inDim), o = UInt32(outDim), bt = UInt32(bits), xs = UInt32(xStride), ys = UInt32(yStride)
        var ac = UInt32(accumulate ? 1 : 0)
        enc.setBytes(&i, length: 4, index: 6); enc.setBytes(&o, length: 4, index: 7); enc.setBytes(&bt, length: 4, index: 8)
        enc.setBytes(&xs, length: 4, index: 9); enc.setBytes(&ys, length: 4, index: 10)
        enc.setBytes(&al, length: 4 * al.count, index: 11); enc.setBytes(&ac, length: 4, index: 12)
        enc.dispatchThreadgroups(MTLSize(width: (outDim + 31) / 32, height: w.count, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func norm(_ enc: MTLComputeCommandEncoder, _ input: MTLBuffer, _ weight: String, _ output: MTLBuffer,
                      dim: Int, heads: Int = 1) throws {
        try normRaw(enc, input, 0, output, dim: dim, heads: heads, weight: try resident.entry(weight), scale: 1)
    }

    private func normRaw(_ enc: MTLComputeCommandEncoder, _ input: MTLBuffer, _ inputOffset: Int, _ output: MTLBuffer,
                         dim: Int, heads: Int, weight: ResidentWeights.Entry?, scale: Float) throws {
        enc.setComputePipelineState(pipelines["rmsnorm"]!)
        enc.setBuffer(input, offset: inputOffset, index: 0)
        enc.setBuffer(resident.buffer, offset: weight?.offset ?? 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var d = UInt32(dim), e = eps, sc = scale, hw = UInt32(weight == nil ? 0 : 1)
        enc.setBytes(&d, length: 4, index: 3); enc.setBytes(&e, length: 4, index: 4)
        enc.setBytes(&sc, length: 4, index: 5); enc.setBytes(&hw, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: heads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: min(256, max(32, dim)), height: 1, depth: 1))
    }

    private func rope(_ enc: MTLComputeCommandEncoder, _ buffer: MTLBuffer, heads count: Int) {
        enc.setComputePipelineState(pipelines["rope_partial"]!)
        enc.setBuffer(buffer, offset: 0, index: 0)
        var d = UInt32(headDim), r = UInt32(rotary), p = UInt32(position), base = ropeBase, hh = UInt32(count)
        enc.setBytes(&d, length: 4, index: 1); enc.setBytes(&r, length: 4, index: 2); enc.setBytes(&p, length: 4, index: 3)
        enc.setBytes(&base, length: 4, index: 4); enc.setBytes(&hh, length: 4, index: 5)
        threads(enc, count * rotary / 2)
    }

    private func add(_ enc: MTLComputeCommandEncoder, _ target: MTLBuffer, _ source: MTLBuffer, _ n: Int) {
        dispatch(enc, "add_inplace", [target, source], u32: [UInt32(n)], threads: n)
    }

    private func dispatch(_ enc: MTLComputeCommandEncoder, _ name: String, _ buffers: [MTLBuffer], u32: [UInt32],
                          threads n: Int, groups: Int? = nil, perGroup: Int? = nil) {
        enc.setComputePipelineState(pipelines[name]!)
        for (i, buffer) in buffers.enumerated() { enc.setBuffer(buffer, offset: 0, index: i) }
        for (j, value) in u32.enumerated() {
            var v = value
            enc.setBytes(&v, length: 4, index: buffers.count + j)
        }
        if let groups, let perGroup {
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: perGroup, height: 1, depth: 1))
        } else {
            threads(enc, n)
        }
    }

    private func threads(_ enc: MTLComputeCommandEncoder, _ n: Int) {
        let width = 256
        enc.dispatchThreadgroups(MTLSize(width: (n + width - 1) / width, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}
