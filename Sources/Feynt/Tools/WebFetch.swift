import Foundation

/// Fetches a public page for the model, and refuses anything that is not public.
///
/// The model chooses the URL, and the model reads text that other people wrote - a page can
/// tell it to fetch `http://192.168.1.1/...` or `http://localhost:19234/...`. Feynt runs on
/// the user's machine, so without a check that is a request from inside their network on a
/// stranger's behalf. Every address a name resolves to is checked before the request and
/// again for every redirect, and only http(s) to public addresses goes through.
enum WebFetch {
    static let maxBytes = 2_000_000
    static let timeout: TimeInterval = 20

    struct Refusal: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    static func fetch(_ raw: String) async throws -> String {
        guard let url = URL(string: raw.trimmingCharacters(in: .whitespaces)),
            let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
            url.host != nil
        else { throw Refusal(message: "Only http and https URLs can be fetched: \(raw)") }
        try ensurePublic(url)

        var request = URLRequest(url: url, timeoutInterval: timeout)
        request.setValue("Mozilla/5.0 (Macintosh) Feynt", forHTTPHeaderField: "User-Agent")
        request.setValue("text/html,text/plain,application/json;q=0.9,*/*;q=0.5", forHTTPHeaderField: "Accept")

        let guardian = RedirectGuard()
        let session = URLSession(configuration: .ephemeral, delegate: guardian, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        let (bytes, response) = try await session.bytes(for: request)
        if let refused = guardian.refusal { throw refused }
        guard let http = response as? HTTPURLResponse else { throw Refusal(message: "No HTTP response") }
        guard (200 ..< 300).contains(http.statusCode) else {
            throw Refusal(message: "HTTP \(http.statusCode) from \(url.absoluteString)")
        }
        let type = (http.value(forHTTPHeaderField: "Content-Type") ?? "").lowercased()
        guard type.isEmpty || type.contains("text") || type.contains("json") || type.contains("xml") else {
            throw Refusal(message: "\(url.absoluteString) is \(type), not a text page")
        }

        var data = Data()
        for try await byte in bytes {
            data.append(byte)
            if data.count >= maxBytes { break }
        }
        let text = String(decoding: data, as: UTF8.self)
        let body = type.contains("html") || text.prefix(512).lowercased().contains("<html")
            ? htmlToText(text) : text
        return "URL: \(http.url?.absoluteString ?? url.absoluteString)\n\n" + body
    }

    /// Checks each redirect target the same way as the first URL - a public page redirecting
    /// to a private address is the classic way around a check made only once.
    private final class RedirectGuard: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
        private let lock = NSLock()
        private var stored: Refusal?
        var refusal: Refusal? { lock.withLock { stored } }

        func urlSession(
            _ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
            newRequest request: URLRequest
        ) async -> URLRequest? {
            guard let url = request.url, let scheme = url.scheme?.lowercased(),
                scheme == "http" || scheme == "https"
            else {
                lock.withLock { stored = Refusal(message: "Redirect to a non-http URL refused") }
                return nil
            }
            do {
                try WebFetch.ensurePublic(url)
                return request
            } catch let refusal as Refusal {
                lock.withLock { stored = refusal }
                return nil
            } catch {
                return nil
            }
        }
    }

    // MARK: - Address checks

    static func ensurePublic(_ url: URL) throws {
        guard let host = url.host?.lowercased() else { throw Refusal(message: "URL has no host") }
        if host == "localhost" || host.hasSuffix(".localhost") || host.hasSuffix(".local")
            || host.hasSuffix(".internal")
        {
            throw Refusal(message: "\(host) is a local address; only public sites can be fetched")
        }
        let addresses = resolve(host)
        guard !addresses.isEmpty else { throw Refusal(message: "\(host) does not resolve") }
        for address in addresses where !isPublic(address) {
            throw Refusal(message: "\(host) resolves to \(address), a private address; only public sites can be fetched")
        }
    }

    private static func resolve(_ host: String) -> [String] {
        var hints = addrinfo(
            ai_flags: AI_ADDRCONFIG, ai_family: AF_UNSPEC, ai_socktype: SOCK_STREAM, ai_protocol: 0,
            ai_addrlen: 0, ai_canonname: nil, ai_addr: nil, ai_next: nil)
        var result: UnsafeMutablePointer<addrinfo>?
        guard getaddrinfo(host, nil, &hints, &result) == 0, let first = result else { return [] }
        defer { freeaddrinfo(first) }
        var out: [String] = []
        var cursor: UnsafeMutablePointer<addrinfo>? = first
        while let info = cursor {
            var buffer = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(info.pointee.ai_addr, info.pointee.ai_addrlen, &buffer, socklen_t(buffer.count),
                           nil, 0, NI_NUMERICHOST) == 0
            {
                out.append(String(cString: buffer))
            }
            cursor = info.pointee.ai_next
        }
        return out
    }

    /// Public unicast only: no loopback, private, link-local, carrier-grade NAT, multicast or
    /// unspecified addresses, in either family, IPv4-mapped IPv6 included.
    static func isPublic(_ address: String) -> Bool {
        var v4 = in_addr()
        if inet_pton(AF_INET, address, &v4) == 1 {
            let b = withUnsafeBytes(of: v4.s_addr) { Array($0) }
            return isPublicV4(b)
        }
        var v6 = in6_addr()
        guard inet_pton(AF_INET6, address.split(separator: "%").first.map(String.init) ?? address, &v6) == 1
        else { return false }
        let b = withUnsafeBytes(of: v6) { Array($0) }
        if b[0 ..< 10].allSatisfy({ $0 == 0 }) && b[10] == 0xff && b[11] == 0xff {
            return isPublicV4(Array(b[12 ..< 16]))
        }
        if b.allSatisfy({ $0 == 0 }) { return false }                         // ::
        if b[0 ..< 15].allSatisfy({ $0 == 0 }) && b[15] == 1 { return false }  // ::1
        if b[0] & 0xfe == 0xfc { return false }                               // fc00::/7
        if b[0] == 0xfe && b[1] & 0xc0 == 0x80 { return false }               // fe80::/10
        if b[0] == 0xff { return false }                                      // multicast
        return true
    }

    private static func isPublicV4(_ b: [UInt8]) -> Bool {
        switch (b[0], b[1]) {
        case (0, _), (10, _), (127, _): return false
        case (169, 254): return false
        case (172, 16 ... 31): return false
        case (192, 168): return false
        case (100, 64 ... 127): return false
        case (224 ... 255, _): return false
        default: return true
        }
    }

    // MARK: - HTML

    /// Enough to give a model the words on a page: scripts, styles and tags out, entities
    /// decoded, whitespace folded. Not a renderer - layout and links are lost, which for
    /// "what does this page say" is the right trade.
    static func htmlToText(_ html: String) -> String {
        var text = html
        for pattern in ["(?is)<script\\b.*?</script>", "(?is)<style\\b.*?</style>",
                        "(?is)<noscript\\b.*?</noscript>", "(?s)<!--.*?-->"]
        {
            text = text.replacingOccurrences(of: pattern, with: " ", options: .regularExpression)
        }
        text = text.replacingOccurrences(
            of: "(?i)<(br|/p|/div|/li|/h[1-6]|/tr)\\b[^>]*>", with: "\n", options: .regularExpression)
        text = text.replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)
        for (entity, char) in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                               ("&quot;", "\""), ("&#39;", "'"), ("&apos;", "'")]
        {
            text = text.replacingOccurrences(of: entity, with: char)
        }
        text = text.replacingOccurrences(of: "[ \\t]+", with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: "\\n\\s*\\n+", with: "\n\n", options: .regularExpression)
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
