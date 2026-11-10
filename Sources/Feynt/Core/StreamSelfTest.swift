import Darwin
import Foundation
import FeyntStream
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// `Feynt --generate <id> <prompt> --stream <slots per layer> [--max-tokens N]`: one greedy
/// answer on the low-memory runtime, printing the text, the rates, the expert cache's hit
/// rate and the process footprint. The MLX engine is never loaded, so the footprint is the
/// runtime's own.
@MainActor func streamSelfTest(spec: ModelSpec, prompt: String, slots: Int, maxTokens: Int) async {
    guard let directory = ModelResolver.installedLocation(for: spec) else {
        print("\(spec.repo) is not installed"); exit(2)
    }
    do {
        let started = Date()
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: directory)
        let runtime = try Qwen35Runtime(directory: directory, config: .init(slotsPerLayer: slots))
        print(String(format: "loaded in %.1fs, %.2f GB weights + slots, footprint %.2f GB",
                     Date().timeIntervalSince(started), Double(runtime.weightBytes) / 1e9, footprintGB()))
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": prompt]], tools: nil,
            additionalContext: ["enable_thinking": false])
        let t0 = Date()
        var next = 0
        for (i, token) in promptTokens.enumerated() {
            next = try runtime.step(token: token, wantLogits: i == promptTokens.count - 1) ?? next
        }
        let prefill = Date().timeIntervalSince(t0)
        var generated: [Int] = []
        let eos = Set([tokenizer.eosTokenId].compactMap { $0 } + [248046, 248044])
        let t1 = Date()
        while generated.count < maxTokens, !eos.contains(next) {
            generated.append(next)
            next = try runtime.step(token: next, wantLogits: true) ?? 0
        }
        let decode = Date().timeIntervalSince(t1)
        print("--- answer ---")
        print(tokenizer.decode(tokenIds: generated))
        print(String(format: "--- %d tokens · %.1f tok/s · prompt %d at %.1f tok/s · hit rate %.0f%% · footprint %.2f GB ---",
                     generated.count, Double(generated.count) / max(decode, 0.001), promptTokens.count,
                     Double(promptTokens.count) / max(prefill, 0.001), runtime.hitRate * 100, footprintGB()))
        print("TOKENS " + generated.map(String.init).joined(separator: ","))
        exit(0)
    } catch {
        print("stream failed: \(error)")
        exit(1)
    }
}

/// The process's physical footprint, the number Activity Monitor shows as Memory.
func footprintGB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? Double(info.phys_footprint) / 1e9 : -1
}
