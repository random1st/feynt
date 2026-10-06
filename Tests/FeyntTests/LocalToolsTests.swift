import Foundation
import Testing
@testable import Feynt

private func call(_ name: String, _ arguments: String) -> EngineToolCall {
    EngineToolCall(id: UUID().uuidString, name: name, argumentsJSON: arguments)
}

@Suite struct LocalToolsTests {
    @Test func withoutAFolderOnlyWebFetchIsOffered() {
        #expect(LocalTools(workspace: nil).names == ["web_fetch"])
    }

    @Test func withAFolderAllFourAreOffered() throws {
        let box = try Sandbox()
        defer { box.remove() }
        #expect(LocalTools(workspace: box.workspace).names == ["read_file", "list_files", "grep", "web_fetch"])
    }

    @Test func readFileNumbersLines() async throws {
        let box = try Sandbox()
        defer { box.remove() }
        let result = await LocalTools(workspace: box.workspace).run(call("read_file", #"{"path":"src/main.swift"}"#))
        #expect(!result.isError)
        #expect(result.text.hasPrefix("1\tlet answer = 42\n2\tlet name"))
    }

    @Test func readFileHonoursOffsetAndLimit() async throws {
        let box = try Sandbox()
        defer { box.remove() }
        let result = await LocalTools(workspace: box.workspace)
            .run(call("read_file", #"{"path":"src/main.swift","offset":2,"limit":1}"#))
        #expect(result.text.hasPrefix("2\tlet name = \"feynt\""))
        #expect(result.text.contains("more lines; use offset"))
    }

    @Test func readFileOutsideIsAnErrorForTheModel() async throws {
        let box = try Sandbox()
        defer { box.remove() }
        let tools = LocalTools(workspace: box.workspace)
        let escaped = await tools.run(call("read_file", #"{"path":"escape/private.txt"}"#))
        #expect(escaped.isError)
        #expect(!escaped.text.contains("secret"))
        let env = await tools.run(call("read_file", #"{"path":".env"}"#))
        #expect(env.isError)
        #expect(!env.text.contains("TOKEN"))
    }

    @Test func listFilesMatchesGlobAndSkipsWhatItShould() async throws {
        let box = try Sandbox()
        defer { box.remove() }
        let tools = LocalTools(workspace: box.workspace)
        let swift = await tools.run(call("list_files", #"{"pattern":"**/*.swift"}"#))
        #expect(swift.text == "src/main.swift")
        let all = await tools.run(call("list_files", "{}"))
        #expect(all.text.contains("README.md"))
        #expect(!all.text.contains(".env"))
        #expect(!all.text.contains("node_modules"))
        #expect(!all.text.contains("private.txt"))
    }

    @Test func grepFindsMatchesInsideOnly() async throws {
        let box = try Sandbox()
        defer { box.remove() }
        let result = await LocalTools(workspace: box.workspace).run(call("grep", #"{"pattern":"answer"}"#))
        let lines = Set(result.text.split(separator: "\n").map(String.init))
        #expect(lines == ["README.md:2: answer is here", "src/main.swift:1: let answer = 42"])
    }

    @Test func grepWithGlobAndCase() async throws {
        let box = try Sandbox()
        defer { box.remove() }
        let result = await LocalTools(workspace: box.workspace)
            .run(call("grep", #"{"pattern":"ANSWER","glob":"*.md","ignore_case":true}"#))
        #expect(result.text == "README.md:2: answer is here")
    }

    @Test func badRegexIsAnErrorNotACrash() async throws {
        let box = try Sandbox()
        defer { box.remove() }
        let result = await LocalTools(workspace: box.workspace).run(call("grep", #"{"pattern":"("}"#))
        #expect(result.isError)
    }

    @Test func fileToolsWithoutAFolderSaySo() async {
        let result = await LocalTools(workspace: nil).run(call("read_file", #"{"path":"x"}"#))
        #expect(result.isError)
        #expect(result.text.contains("No working folder"))
    }

    @Test(arguments: [
        "http://127.0.0.1:19234/v1/models", "http://localhost/", "http://192.168.1.1/",
        "http://10.0.0.1/", "http://[::1]/", "http://169.254.169.254/latest/meta-data/",
        "http://printer.local/", "ftp://example.com/", "file:///etc/passwd",
    ])
    func webFetchRefusesLocalTargets(_ url: String) async {
        let result = await LocalTools(workspace: nil).run(call("web_fetch", #"{"url":"\#(url)"}"#))
        #expect(result.isError)
    }

    @Test func longResultsAreClipped() {
        let clipped = LocalTools.clip(String(repeating: "a", count: LocalTools.resultLimit + 10))
        #expect(clipped.hasSuffix("[truncated: 10 more characters]"))
    }
}

@Suite struct GlobTests {
    @Test(arguments: [
        ("**/*.swift", "a.swift", true), ("**/*.swift", "src/deep/a.swift", true),
        ("*.swift", "src/a.swift", false), ("src/*.md", "src/README.md", true),
        ("src/?.md", "src/a.md", true), ("src/?.md", "src/ab.md", false),
        ("*.(md)", "x.(md)", true), ("**/*", "anything/at/all", true),
    ])
    func matches(_ pattern: String, _ path: String, _ expected: Bool) throws {
        #expect(try Glob(pattern).matches(path) == expected)
    }
}

@Suite struct WebFetchTests {
    @Test(arguments: [
        "127.0.0.1", "10.1.2.3", "172.16.0.1", "172.31.255.255", "192.168.0.10", "169.254.169.254",
        "100.64.0.1", "0.0.0.0", "224.0.0.1", "255.255.255.255",
        "::", "::1", "fc00::1", "fd12:3456::1", "fe80::1", "fe80::1%en0", "ff02::1", "::ffff:127.0.0.1",
        "::ffff:192.168.1.1", "not an address",
    ])
    func privateAddressesAreNotPublic(_ address: String) {
        #expect(!WebFetch.isPublic(address))
    }

    @Test(arguments: ["93.184.215.14", "1.1.1.1", "172.32.0.1", "100.128.0.1", "2606:4700::1111", "::ffff:8.8.8.8"])
    func publicAddressesArePublic(_ address: String) {
        #expect(WebFetch.isPublic(address))
    }

    @Test func htmlLosesScriptsAndTags() {
        let text = WebFetch.htmlToText(
            "<html><script>var x=1</script><style>p{}</style><p>Hello &amp; <b>world</b></p></html>")
        #expect(!text.contains("var x"))
        #expect(!text.contains("<"))
        #expect(text.contains("Hello & world"))
    }
}

@Suite struct VersionTests {
    @Test @MainActor func comparesNumerically() {
        #expect(UpdateChecker.isVersion("0.10.0", newerThan: "0.9.9"))
        #expect(UpdateChecker.isVersion("0.8.2", newerThan: "0.8.1"))
        #expect(UpdateChecker.isVersion("1.0", newerThan: "0.99.99"))
        #expect(!UpdateChecker.isVersion("0.8.1", newerThan: "0.8.1"))
        #expect(!UpdateChecker.isVersion("0.8", newerThan: "0.8.0"))
        #expect(!UpdateChecker.isVersion("0.8.0", newerThan: "0.8.1"))
    }
}
