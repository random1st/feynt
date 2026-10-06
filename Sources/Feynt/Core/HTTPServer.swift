import Foundation
import Network

struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data
}

/// Writes one response per connection. Either a complete body or an SSE stream; both end by
/// closing the socket, so no chunked transfer encoding is needed.
final class HTTPResponder: @unchecked Sendable {
    private let connection: NWConnection
    private let lock = NSLock()
    private var closed = false
    private var peerHandler: (() -> Void)?
    private var peerGone = false

    init(connection: NWConnection) {
        self.connection = connection
    }

    /// Called once if the client closes the connection before the response is finished.
    ///
    /// Without it a client that gave up - a tool call that timed out in Claude Code, a turn
    /// interrupted in pi - left the generation running to its token budget with the GPU
    /// held and every other request queued behind it. A handler set after the client has
    /// already gone runs at once, so a route cannot miss a disconnect by setting it late.
    var onPeerClosed: (() -> Void)? {
        get { lock.withLock { peerHandler } }
        set {
            let runNow: Bool = lock.withLock {
                peerHandler = newValue
                return peerGone && !closed
            }
            if runNow, let newValue { newValue() }
        }
    }

    /// The server's read side reports end-of-stream here. Our own `finish()` marks the
    /// response closed before it cancels the connection, so that cancellation is not
    /// mistaken for the client leaving.
    func peerClosed() {
        let handler: (() -> Void)? = lock.withLock {
            guard !closed, !peerGone else { return nil }
            peerGone = true
            return peerHandler
        }
        handler?()
    }

    func send(status: Int, contentType: String, body: Data) {
        var head = "HTTP/1.1 \(status) \(Self.reason(status))\r\n"
        head += "Content-Type: \(contentType)\r\n"
        head += "Content-Length: \(body.count)\r\n"
        head += "Connection: close\r\n\r\n"
        var payload = Data(head.utf8)
        payload.append(body)
        write(payload, then: { self.finish() })
    }

    func sendJSON(status: Int, object: [String: Any]) {
        let body = (try? JSONSerialization.data(withJSONObject: object)) ?? Data("{}".utf8)
        send(status: status, contentType: "application/json", body: body)
    }

    func beginEventStream() {
        let head = "HTTP/1.1 200 OK\r\n"
            + "Content-Type: text/event-stream\r\n"
            + "Cache-Control: no-cache\r\n"
            + "Connection: close\r\n\r\n"
        write(Data(head.utf8), then: nil)
    }

    func writeEvent(_ text: String) {
        write(Data(text.utf8), then: nil)
    }

    /// Half-closes the send side and only then cancels. A bare `cancel()` drops writes still
    /// queued on the connection, which silently truncated SSE responses.
    func finish() {
        let wasClosed: Bool = lock.withLock {
            defer { closed = true }
            return closed
        }
        guard !wasClosed else { return }
        connection.send(
            content: nil,
            contentContext: .finalMessage,
            isComplete: true,
            completion: .contentProcessed { [connection] _ in connection.cancel() })
    }

    private func write(_ data: Data, then completion: (() -> Void)?) {
        guard !lock.withLock({ closed }) else { return }
        connection.send(
            content: data,
            completion: .contentProcessed { _ in completion?() })
    }

    private static func reason(_ status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 202: return "Accepted"
        case 400: return "Bad Request"
        case 403: return "Forbidden"
        case 404: return "Not Found"
        case 405: return "Method Not Allowed"
        case 503: return "Service Unavailable"
        default: return "Error"
        }
    }
}

/// Minimal localhost HTTP/1.1 server on Network.framework.
///
/// Hand-rolled rather than pulled from a package: the app ships as a drag-and-drop bundle and
/// an embedded loopback endpoint does not justify a third-party dependency.
final class HTTPServer {
    typealias Handler = (HTTPRequest, HTTPResponder) -> Void

    private let queue = DispatchQueue(label: "feynt.http", qos: .userInitiated)
    private var listener: NWListener?
    private let handler: Handler

    /// Requests larger than this are refused; a chat prompt never approaches it.
    private let maxBodyBytes = 8 * 1024 * 1024

    init(handler: @escaping Handler) {
        self.handler = handler
    }

    var isRunning: Bool { listener != nil }

    func start(port: UInt16) throws {
        stop()
        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            throw NSError(
                domain: "Feynt", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid port \(port)"])
        }
        let parameters = NWParameters.tcp
        // Loopback only: this endpoint must never be reachable from the network.
        parameters.requiredLocalEndpoint = .hostPort(host: .ipv4(.loopback), port: nwPort)
        parameters.allowLocalEndpointReuse = true

        let listener = try NWListener(using: parameters)
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }
        listener.start(queue: queue)
        self.listener = listener
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        receive(connection, buffer: Data())
    }

    private func receive(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) {
            [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }

            if buffer.count > self.maxBodyBytes {
                connection.cancel()
                return
            }
            if let request = Self.parse(buffer) {
                let responder = HTTPResponder(connection: connection)
                self.handler(request, responder)
                self.watchForClose(connection, responder)
                return
            }
            if isComplete || error != nil {
                connection.cancel()
                return
            }
            self.receive(connection, buffer: buffer)
        }
    }

    /// Keeps reading after the request so that the client closing its side is noticed.
    ///
    /// The responses close the connection themselves (`Connection: close`), so anything a
    /// client sends after its request is not a second request and is dropped. What matters
    /// is end-of-stream: it is the only way a client says it no longer wants the answer.
    private func watchForClose(_ connection: NWConnection, _ responder: HTTPResponder) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) {
            [weak self] _, _, isComplete, error in
            if isComplete || error != nil {
                responder.peerClosed()
                return
            }
            self?.watchForClose(connection, responder)
        }
    }

    /// Returns nil while the request is still incomplete.
    static func parse(_ buffer: Data) -> HTTPRequest? {
        let separator = Data("\r\n\r\n".utf8)
        guard let range = buffer.range(of: separator) else { return nil }
        let headerData = buffer[buffer.startIndex ..< range.lowerBound]
        guard let headerText = String(data: headerData, encoding: .utf8) else { return nil }

        var lines = headerText.components(separatedBy: "\r\n")
        guard !lines.isEmpty else { return nil }
        let requestLine = lines.removeFirst().split(separator: " ", omittingEmptySubsequences: true)
        guard requestLine.count >= 2 else { return nil }

        var headers: [String: String] = [:]
        for line in lines {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[line.startIndex ..< colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }

        let bodyStart = range.upperBound
        let available = buffer[bodyStart...]
        let expected = Int(headers["content-length"] ?? "0") ?? 0
        guard available.count >= expected else { return nil }

        return HTTPRequest(
            method: String(requestLine[0]).uppercased(),
            path: String(requestLine[1]),
            headers: headers,
            body: Data(available.prefix(expected)))
    }
}
