import Foundation

/// Splits a Qwen text stream into reasoning and answer parts.
///
/// The MLX generation loop yields one decoded string per step with no channel marker, so
/// the `<think>…</think>` block the template asks for arrives inline. Buffering just enough
/// characters to recognise a straddling tag keeps a tag from being emitted as visible text.
struct ThinkingSplitter {
    private static let open = "<think>"
    private static let close = "</think>"

    private var inThinking = false
    private var pending = ""

    /// Longest prefix of either tag that could still be completed by the next chunk.
    private var maxTagLength: Int { max(Self.open.count, Self.close.count) }

    mutating func consume(_ chunk: String) -> [EngineEvent] {
        pending += chunk
        var events: [EngineEvent] = []

        while true {
            let tag = inThinking ? Self.close : Self.open
            if let range = pending.range(of: tag) {
                let head = String(pending[pending.startIndex ..< range.lowerBound])
                if !head.isEmpty {
                    events.append(inThinking ? .reasoning(head) : .text(head))
                }
                pending = String(pending[range.upperBound...])
                inThinking.toggle()
                continue
            }
            break
        }

        // Hold back a possible partial tag at the tail; emit everything before it.
        let keep = partialTagSuffixLength()
        if pending.count > keep {
            let cut = pending.index(pending.endIndex, offsetBy: -keep)
            let ready = String(pending[pending.startIndex ..< cut])
            pending = String(pending[cut...])
            if !ready.isEmpty {
                events.append(inThinking ? .reasoning(ready) : .text(ready))
            }
        }
        return events
    }

    /// Flush whatever is still buffered when the stream ends.
    mutating func finish() -> [EngineEvent] {
        guard !pending.isEmpty else { return [] }
        let tail = pending
        pending = ""
        return [inThinking ? .reasoning(tail) : .text(tail)]
    }

    private func partialTagSuffixLength() -> Int {
        let tag = inThinking ? Self.close : Self.open
        let limit = min(pending.count, maxTagLength - 1)
        var length = limit
        while length > 0 {
            let suffix = pending.suffix(length)
            if tag.hasPrefix(suffix) { return length }
            length -= 1
        }
        return 0
    }
}
