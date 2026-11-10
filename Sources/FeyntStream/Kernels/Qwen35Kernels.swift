/// Metal kernels for one decode step of Qwen3.5-MoE, compiled from source at runtime.
///
/// Correctness first: activations are fp32, weights are read in the MLX checkpoint's own
/// format (affine 4- or 8-bit, group 64, bf16 scales and biases, `[out, in*bits/32]` uint32
/// rows, low nibble first) with nothing repacked. The quantized GEMV follows the same
/// per-group form TurboFieldfare's `moe.metal` uses - `s * sum(q*x) + b * sum(x)` - with one
/// simdgroup per output row.
enum Qwen35Kernels {
    static let source = #"""
#include <metal_stdlib>
using namespace metal;

// y[row] = alpha * dot(dequant(W[row]), x)  (+ y[row] when accumulate)
// One simdgroup per row, ROWS rows per threadgroup.
constant uint ROWS = 8;
kernel void qgemv(device const uint*   W     [[buffer(0)]],
                  device const bfloat* S     [[buffer(1)]],
                  device const bfloat* B     [[buffer(2)]],
                  device const float*  x     [[buffer(3)]],
                  device float*        y     [[buffer(4)]],
                  constant uint&       inDim [[buffer(5)]],
                  constant uint&       outDim[[buffer(6)]],
                  constant uint&       bits  [[buffer(7)]],
                  constant float&      alpha [[buffer(8)]],
                  constant uint&       accumulate [[buffer(9)]],
                  uint tg   [[threadgroup_position_in_grid]],
                  uint sg   [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * ROWS + sg;
    if (row >= outDim) return;
    uint groups = inDim / 64;
    uint wordsPerGroup = 64 * bits / 32;
    uint perWord = 32 / bits;
    uint mask = (1u << bits) - 1u;
    device const uint* wrow = W + row * (inDim * bits / 32);
    device const bfloat* srow = S + row * groups;
    device const bfloat* brow = B + row * groups;
    float acc = 0.0f;
    for (uint g = lane; g < groups; g += 32) {
        float qx = 0.0f, sx = 0.0f;
        device const uint* wg = wrow + g * wordsPerGroup;
        device const float* xg = x + g * 64;
        for (uint j = 0; j < wordsPerGroup; ++j) {
            uint w = wg[j];
            for (uint k = 0; k < perWord; ++k) {
                float xv = xg[j * perWord + k];
                qx += float((w >> (bits * k)) & mask) * xv;
                sx += xv;
            }
        }
        acc += float(srow[g]) * qx + float(brow[g]) * sx;
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        float v = alpha * acc;
        y[row] = accumulate ? y[row] + v : v;
    }
}

// One row of a 4-bit embedding table, dequantized.
kernel void embed_q4(device const uint*   W [[buffer(0)]],
                     device const bfloat* S [[buffer(1)]],
                     device const bfloat* B [[buffer(2)]],
                     device float*        x [[buffer(3)]],
                     constant uint&       token [[buffer(4)]],
                     constant uint&       dim   [[buffer(5)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= dim) return;
    uint g = i / 64;
    uint word = W[token * (dim / 8) + i / 8];
    float q = float((word >> (4 * (i % 8))) & 0xFu);
    x[i] = float(S[token * (dim / 64) + g]) * q + float(B[token * (dim / 64) + g]);
}

// y = x * rsqrt(mean(x^2) + eps) * (w ? w : 1) * scale, per head of `dim` elements.
// One threadgroup per head.
kernel void rmsnorm(device const float*  x     [[buffer(0)]],
                    device const bfloat* w     [[buffer(1)]],
                    device float*        y     [[buffer(2)]],
                    constant uint&       dim   [[buffer(3)]],
                    constant float&      eps   [[buffer(4)]],
                    constant float&      scale [[buffer(5)]],
                    constant uint&       hasWeight [[buffer(6)]],
                    uint head [[threadgroup_position_in_grid]],
                    uint tid  [[thread_position_in_threadgroup]],
                    uint tcount [[threads_per_threadgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[32];
    device const float* xh = x + head * dim;
    device float* yh = y + head * dim;
    float ss = 0.0f;
    for (uint i = tid; i < dim; i += tcount) { float v = xh[i]; ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint n = (tcount + 31) / 32;
        float t = lane < n ? partial[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) partial[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(partial[0] / float(dim) + eps) * scale;
    for (uint i = tid; i < dim; i += tcount) {
        float v = xh[i] * inv;
        yh[i] = hasWeight ? v * float(w[i]) : v;
    }
}

kernel void add_inplace(device float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                        constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) a[i] += b[i];
}

// Gated DeltaNet short conv, one decode step, then silu. state: [3][C] fp32, w: [C][4] bf16.
kernel void conv_step(device const float*  qkv   [[buffer(0)]],
                      device float*        state [[buffer(1)]],
                      device const bfloat* w     [[buffer(2)]],
                      device float*        u     [[buffer(3)]],
                      constant uint&       C     [[buffer(4)]],
                      uint c [[thread_position_in_grid]]) {
    if (c >= C) return;
    float s0 = state[c], s1 = state[C + c], s2 = state[2 * C + c], v = qkv[c];
    float pre = float(w[c * 4 + 0]) * s0 + float(w[c * 4 + 1]) * s1
              + float(w[c * 4 + 2]) * s2 + float(w[c * 4 + 3]) * v;
    state[c] = s1; state[C + c] = s2; state[2 * C + c] = v;
    u[c] = pre / (1.0f + exp(-pre));
}

// beta = sigmoid(b), g = exp(-exp(A_log) * softplus(a + dt_bias)), per value head.
kernel void delta_gates(device const float*  a     [[buffer(0)]],
                        device const float*  b     [[buffer(1)]],
                        device const bfloat* aLog  [[buffer(2)]],
                        device const bfloat* dtBias[[buffer(3)]],
                        device float*        g     [[buffer(4)]],
                        device float*        beta  [[buffer(5)]],
                        constant uint&       H     [[buffer(6)]],
                        uint h [[thread_position_in_grid]]) {
    if (h >= H) return;
    float x = a[h] + float(dtBias[h]);
    float sp = x > 20.0f ? x : log(1.0f + exp(x));
    g[h] = exp(-exp(float(aLog[h])) * sp);
    beta[h] = 1.0f / (1.0f + exp(-b[h]));
}

// Gated delta rule, one token. state S: [Hv][Dv][Dk] fp32; q,k: [Hk][Dk]; v: [Hv][Dv].
// One simdgroup per (hv, dv) row; each lane holds Dk/32 state values. Ported from the
// fork's gated_delta_step (MLXLMCommon/GatedDelta.swift) with T = 1.
kernel void delta_step(device const float* q    [[buffer(0)]],
                       device const float* k    [[buffer(1)]],
                       device const float* v    [[buffer(2)]],
                       device const float* g    [[buffer(3)]],
                       device const float* beta [[buffer(4)]],
                       device float*       S    [[buffer(5)]],
                       device float*       y    [[buffer(6)]],
                       constant uint&      Dk   [[buffer(7)]],
                       constant uint&      Dv   [[buffer(8)]],
                       constant uint&      ratio[[buffer(9)]],
                       uint2 gid  [[threadgroup_position_in_grid]],
                       uint  sg   [[simdgroup_index_in_threadgroup]],
                       uint  lane [[thread_index_in_simdgroup]]) {
    uint hv = gid.y;
    uint dv = gid.x * 4 + sg;
    if (dv >= Dv) return;
    uint hk = hv / ratio;
    device float* row = S + (hv * Dv + dv) * Dk;
    device const float* kh = k + hk * Dk;
    device const float* qh = q + hk * Dk;
    float decay = g[hv];
    float kv = 0.0f;
    float st[8];
    uint per = Dk / 32;
    for (uint i = 0; i < per; ++i) {
        uint d = lane * per + i;
        st[i] = row[d] * decay;
        kv += st[i] * kh[d];
    }
    kv = simd_sum(kv);
    float delta = (v[hv * Dv + dv] - kv) * beta[hv];
    float out = 0.0f;
    for (uint i = 0; i < per; ++i) {
        uint d = lane * per + i;
        st[i] += delta * kh[d];
        row[d] = st[i];
        out += st[i] * qh[d];
    }
    out = simd_sum(out);
    if (lane == 0) y[hv * Dv + dv] = out;
}

// o = rmsnorm(y_head; w) * silu(z), per value head. One threadgroup per head.
kernel void gated_norm(device const float*  y   [[buffer(0)]],
                       device const float*  z   [[buffer(1)]],
                       device const bfloat* w   [[buffer(2)]],
                       device float*        o   [[buffer(3)]],
                       constant uint&       dim [[buffer(4)]],
                       constant float&      eps [[buffer(5)]],
                       uint head [[threadgroup_position_in_grid]],
                       uint tid  [[thread_position_in_threadgroup]],
                       uint sg   [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[32];
    device const float* yh = y + head * dim;
    float ss = 0.0f;
    for (uint i = tid; i < dim; i += 128) { float v = yh[i]; ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = partial[0] + partial[1] + partial[2] + partial[3];
    float inv = rsqrt(total / float(dim) + eps);
    for (uint i = tid; i < dim; i += 128) {
        float zz = z[head * dim + i];
        float silu = zz / (1.0f + exp(-zz));
        o[head * dim + i] = yh[i] * inv * float(w[i]) * silu;
    }
}

// Split q_proj output [H][2*D] into q [H][D] and gate [H][D].
kernel void split_qgate(device const float* qg [[buffer(0)]], device float* q [[buffer(1)]],
                        device float* gate [[buffer(2)]], constant uint& D [[buffer(3)]],
                        constant uint& H [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= H * D) return;
    uint h = i / D, d = i % D;
    q[i] = qg[h * 2 * D + d];
    gate[i] = qg[h * 2 * D + D + d];
}

// NeoX rotary on the first `rot` dims of each head: pairs (i, i + rot/2).
kernel void rope_partial(device float* x [[buffer(0)]], constant uint& D [[buffer(1)]],
                         constant uint& rot [[buffer(2)]], constant uint& pos [[buffer(3)]],
                         constant float& base [[buffer(4)]], constant uint& H [[buffer(5)]],
                         uint i [[thread_position_in_grid]]) {
    uint hr = rot / 2;
    if (i >= H * hr) return;
    uint h = i / hr, j = i % hr;
    float theta = float(pos) * pow(base, -2.0f * float(j) / float(rot));
    float c = cos(theta), s = sin(theta);
    device float* xh = x + h * D;
    float a = xh[j], b = xh[j + hr];
    xh[j] = a * c - b * s;
    xh[j + hr] = a * s + b * c;
}

// Append this token's k and v ([Hkv][D] each) at position `pos` of the cache [Hkv][cap][D].
kernel void kv_append(device const float* k [[buffer(0)]], device const float* v [[buffer(1)]],
                      device float* K [[buffer(2)]], device float* V [[buffer(3)]],
                      constant uint& D [[buffer(4)]], constant uint& cap [[buffer(5)]],
                      constant uint& pos [[buffer(6)]], constant uint& Hkv [[buffer(7)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= Hkv * D) return;
    uint h = i / D, d = i % D;
    K[(h * cap + pos) * D + d] = k[i];
    V[(h * cap + pos) * D + d] = v[i];
}

// Decode attention for one query head per threadgroup over T cached positions, then
// multiplied by sigmoid(gate). D = 256, 256 threads: thread d owns output dim d.
kernel void attend(device const float* q    [[buffer(0)]],
                   device const float* K    [[buffer(1)]],
                   device const float* V    [[buffer(2)]],
                   device const float* gate [[buffer(3)]],
                   device float*       out  [[buffer(4)]],
                   constant uint& D   [[buffer(5)]],
                   constant uint& cap [[buffer(6)]],
                   constant uint& T   [[buffer(7)]],
                   constant uint& groupSize [[buffer(8)]],
                   constant float& scale [[buffer(9)]],
                   uint h   [[threadgroup_position_in_grid]],
                   uint tid [[thread_position_in_threadgroup]],
                   uint sg  [[simdgroup_index_in_threadgroup]],
                   uint lane[[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    threadgroup float shared_score;
    uint kvh = h / groupSize;
    float qd = q[h * D + tid];
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (uint t = 0; t < T; ++t) {
        float p = qd * K[(kvh * cap + t) * D + tid];
        p = simd_sum(p);
        if (lane == 0) partial[sg] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float s = 0.0f;
            for (uint i = 0; i < 8; ++i) s += partial[i];
            shared_score = s * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s = shared_score;
        float mNew = max(m, s);
        float corr = exp(m - mNew);
        float e = exp(s - mNew);
        l = l * corr + e;
        acc = acc * corr + e * V[(kvh * cap + t) * D + tid];
        m = mNew;
    }
    float o = acc / l;
    float gg = gate[h * D + tid];
    out[h * D + tid] = o / (1.0f + exp(-gg));
}

// Router: softmax over E logits, top-K, renormalized weights. One threadgroup of E threads.
kernel void route_topk(device const float* logits [[buffer(0)]],
                       device int*         idx    [[buffer(1)]],
                       device float*       wts    [[buffer(2)]],
                       constant uint&      E      [[buffer(3)]],
                       constant uint&      K      [[buffer(4)]],
                       uint e [[thread_position_in_threadgroup]]) {
    threadgroup float p[256];
    float x = e < E ? logits[e] : -INFINITY;
    p[e] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) {
        float mx = -INFINITY;
        for (uint i = 0; i < E; ++i) mx = max(mx, p[i]);
        float sum = 0.0f;
        for (uint i = 0; i < E; ++i) { p[i] = exp(p[i] - mx); sum += p[i]; }
        for (uint i = 0; i < E; ++i) p[i] /= sum;
        // Top-K by value; ties go to the lower index. Selected experts are then listed in
        // ascending index order, as the reference does.
        bool taken[256];
        for (uint i = 0; i < E; ++i) taken[i] = false;
        float total = 0.0f;
        for (uint k = 0; k < K; ++k) {
            int best = -1; float bv = -1.0f;
            for (uint i = 0; i < E; ++i) if (!taken[i] && p[i] > bv) { bv = p[i]; best = int(i); }
            taken[best] = true;
            total += bv;
        }
        uint n = 0;
        for (uint i = 0; i < E && n < K; ++i) if (taken[i]) { idx[n] = int(i); wts[n] = p[i] / total; ++n; }
    }
}

// h = silu(g) * u
kernel void swiglu(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                   device float* h [[buffer(2)]], constant uint& n [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float gg = g[i];
    h[i] = gg / (1.0f + exp(-gg)) * u[i];
}

// out += sigmoid(gate[0]) * s
kernel void gated_add(device float* out [[buffer(0)]], device const float* s [[buffer(1)]],
                      device const float* gate [[buffer(2)]], constant uint& n [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    out[i] += s[i] / (1.0f + exp(-gate[0]));
}

kernel void zero(device float* a [[buffer(0)]], constant uint& n [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
    if (i < n) a[i] = 0.0f;
}

// Argmax over n logits; one threadgroup of 1024 threads.
kernel void argmax(device const float* x [[buffer(0)]], device int* out [[buffer(1)]],
                   constant uint& n [[buffer(2)]],
                   uint tid [[thread_position_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup int bi[32];
    float best = -INFINITY; int bestI = 0;
    for (uint i = tid; i < n; i += 1024) { float v = x[i]; if (v > best) { best = v; bestI = int(i); } }
    for (ushort off = 16; off > 0; off /= 2) {
        float ov = simd_shuffle_down(best, off); int oi = simd_shuffle_down(bestI, off);
        if (ov > best || (ov == best && oi < bestI)) { best = ov; bestI = oi; }
    }
    if (lane == 0) { bv[sg] = best; bi[sg] = bestI; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float b = bv[0]; int i0 = bi[0];
        for (uint s = 1; s < 32; ++s) if (bv[s] > b || (bv[s] == b && bi[s] < i0)) { b = bv[s]; i0 = bi[s]; }
        out[0] = i0;
    }
}
"""#
}
