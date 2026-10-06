import Foundation
import Testing
@testable import Feynt

/// A throwaway folder with a file, a subfolder and a link pointing out of it.
struct Sandbox {
    let root: URL
    let outside: URL

    init() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("feynt-tests-\(UUID().uuidString)")
        root = base.appendingPathComponent("work")
        outside = base.appendingPathComponent("elsewhere")
        let fm = FileManager.default
        try fm.createDirectory(at: root.appendingPathComponent("src"), withIntermediateDirectories: true)
        try fm.createDirectory(at: outside, withIntermediateDirectories: true)
        try "let answer = 42\nlet name = \"feynt\"\n".write(
            to: root.appendingPathComponent("src/main.swift"), atomically: true, encoding: .utf8)
        try "# Notes\nanswer is here\n".write(
            to: root.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)
        try "TOKEN=abc\n".write(to: root.appendingPathComponent(".env"), atomically: true, encoding: .utf8)
        try "secret\n".write(to: outside.appendingPathComponent("private.txt"), atomically: true, encoding: .utf8)
        try fm.createSymbolicLink(
            at: root.appendingPathComponent("escape"), withDestinationURL: outside)
        try fm.createDirectory(at: root.appendingPathComponent("node_modules/pkg"), withIntermediateDirectories: true)
        try "answer in a dependency\n".write(
            to: root.appendingPathComponent("node_modules/pkg/index.js"), atomically: true, encoding: .utf8)
    }

    var workspace: Workspace { Workspace(root) }

    func remove() {
        try? FileManager.default.removeItem(at: root.deletingLastPathComponent())
    }
}

@Suite struct WorkspaceTests {
    @Test func relativePathResolvesInsideTheRoot() throws {
        let box = try Sandbox()
        defer { box.remove() }
        let url = try box.workspace.resolve("src/main.swift")
        #expect(box.workspace.contains(url))
        #expect(box.workspace.display(url) == "src/main.swift")
    }

    @Test func dotDotOutOfTheRootIsRefused() throws {
        let box = try Sandbox()
        defer { box.remove() }
        #expect(throws: Workspace.Refusal.self) { try box.workspace.resolve("../elsewhere/private.txt") }
    }

    @Test func absolutePathOutsideIsRefused() throws {
        let box = try Sandbox()
        defer { box.remove() }
        #expect(throws: Workspace.Refusal.self) { try box.workspace.resolve("/etc/hosts") }
    }

    @Test func symlinkOutOfTheRootIsRefused() throws {
        let box = try Sandbox()
        defer { box.remove() }
        #expect(throws: Workspace.Refusal.self) { try box.workspace.resolve("escape/private.txt") }
        // A file that does not exist yet behind the link is caught through its ancestor.
        #expect(throws: Workspace.Refusal.self) { try box.workspace.resolve("escape/new.txt") }
    }

    @Test func tmpAndPrivateTmpAreTheSamePlace() throws {
        let box = try Sandbox()
        defer { box.remove() }
        let resolved = try box.workspace.resolve(box.root.path + "/README.md")
        #expect(box.workspace.display(resolved) == "README.md")
    }

    @Test func credentialFilesAreRefusedInsideTheRoot() throws {
        let box = try Sandbox()
        defer { box.remove() }
        #expect(throws: Workspace.Refusal.self) { try box.workspace.resolve(".env") }
    }

    @Test(arguments: [".ssh/config", ".aws/credentials", ".diana/vault", ".config/gh/hosts.yml"])
    func homeCredentialDirectoriesAreSecret(_ relative: String) {
        let home = FileManager.default.homeDirectoryForCurrentUser.resolvingSymlinksInPath()
        #expect(Workspace.secret(home.appendingPathComponent(relative)) != nil)
    }

    @Test(arguments: [".env", ".env.local", "server.pem", "cert.p12", "tls.key", "id_ed25519", ".netrc", "credentials"])
    func credentialNamesAreSecret(_ name: String) {
        #expect(Workspace.secret(URL(fileURLWithPath: "/tmp/project/\(name)")) != nil)
    }

    @Test(arguments: ["main.swift", "id_card.png", "environment.md", "keynote.txt", "README.md"])
    func ordinaryNamesAreNotSecret(_ name: String) {
        #expect(Workspace.secret(URL(fileURLWithPath: "/tmp/project/\(name)")) == nil)
    }

    @Test func homeAsAWorkspaceStillHidesSsh() throws {
        let home = Workspace(FileManager.default.homeDirectoryForCurrentUser)
        #expect(throws: Workspace.Refusal.self) { try home.resolve(".ssh/id_rsa") }
    }
}
