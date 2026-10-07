import Foundation

/// MCP progress notifications for a long `tools/call`, written onto the request's own SSE
/// response before the result.
///
/// Clients give up on a tool call that says nothing for about a minute, and a local model
/// reading a long prompt can be silent that long before its first token. So the request
/// reports as tokens arrive, at most once a second, and a heartbeat covers the silent
/// stretch - the prompt being read, a model being loaded.
final class MCPProgress: @unchecked Sendable {
    private let token: Any
    private let responder: HTTPResponder
    private let lock = NSLock()
    private var tokens = 0
    private var started = Date()
    private var firstToken: Date?
    private var lastSent = Date.distantPast
    private var heartbeat: Task<Void, Never>?
    /// The spec has `progress` strictly increase from one notification to the next, and a
    /// heartbeat during prefill has no new tokens to report - so the value counts
    /// notifications, and the message carries the numbers.
    private var sequence = 0

    static let heartbeatSeconds: UInt64 = 10

    init(token: Any, responder: HTTPResponder) {
        self.token = token
        self.responder = responder
    }

    func start() {
        responder.beginEventStream()
        started = Date()
        send(message: "started")
        heartbeat = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: Self.heartbeatSeconds * 1_000_000_000)
                guard let self, !Task.isCancelled else { return }
                self.send(message: self.status())
            }
        }
    }

    /// A chunk of the answer arrived.
    func generated(_ chunk: String) {
        let due: Bool = lock.withLock {
            if firstToken == nil { firstToken = Date() }
            tokens += max(1, chunk.count / 4)
            return Date().timeIntervalSince(lastSent) >= 1
        }
        if due { send(message: status()) }
    }

    func note(_ message: String) { send(message: message) }

    /// The result, as the last event of the stream; the response ends with it.
    func finish(with payload: [String: Any]) {
        heartbeat?.cancel()
        let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
        responder.writeEvent("event: message\ndata: \(String(decoding: data, as: UTF8.self))\n\n")
        responder.finish()
    }

    private func status() -> String {
        lock.withLock {
            let elapsed = Int(Date().timeIntervalSince(started))
            guard let firstToken else { return "reading the prompt, \(elapsed) s" }
            let decoding = max(Date().timeIntervalSince(firstToken), 0.001)
            return "~\(tokens) tokens, \(Int(Double(tokens) / decoding)) tok/s, \(elapsed) s"
        }
    }

    private func send(message: String) {
        let progress: Int = lock.withLock {
            lastSent = Date()
            sequence += 1
            return sequence
        }
        let notification: [String: Any] = [
            "jsonrpc": "2.0", "method": "notifications/progress",
            "params": ["progressToken": token, "progress": progress, "message": message],
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: notification) else { return }
        responder.writeEvent("event: message\ndata: \(String(decoding: data, as: UTF8.self))\n\n")
    }
}
