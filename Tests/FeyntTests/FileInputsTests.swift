import Foundation
import Testing
@testable import Feynt

@Suite struct FileInputsTests {
    @Test func textFilesAreFencedWithTheirPaths() throws {
        let box = try Sandbox()
        defer { box.remove() }
        let loaded = try FileInputs.load(["README.md", "src/main.swift"], workspace: box.workspace)
        #expect(loaded.text.contains("<file path=\"README.md\">\n# Notes"))
        #expect(loaded.text.contains("<file path=\"src/main.swift\">\nlet answer = 42"))
        #expect(loaded.images.isEmpty)
    }

    @Test func absolutePathsWorkWithoutAWorkspace() throws {
        let box = try Sandbox()
        defer { box.remove() }
        let loaded = try FileInputs.load([box.root.path + "/README.md"], workspace: nil)
        #expect(loaded.text.contains("answer is here"))
    }

    @Test func relativePathWithoutAWorkspaceIsRefused() {
        #expect(throws: FileInputs.Failure.self) { try FileInputs.load(["README.md"], workspace: nil) }
    }

    @Test func secretsAndEscapesAreRefused() throws {
        let box = try Sandbox()
        defer { box.remove() }
        #expect(throws: (any Error).self) { try FileInputs.load([".env"], workspace: box.workspace) }
        #expect(throws: (any Error).self) { try FileInputs.load(["escape/private.txt"], workspace: box.workspace) }
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        #expect(throws: FileInputs.Failure.self) { try FileInputs.load([home + "/.ssh/config"], workspace: nil) }
    }

    @Test func imagesAreRecognisedByTheirBytes() throws {
        let box = try Sandbox()
        defer { box.remove() }
        let png = Data(base64Encoded:
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")!
        // Named .txt on purpose: the bytes decide, not the extension.
        try png.write(to: box.root.appendingPathComponent("picture.txt"))
        let loaded = try FileInputs.load(["picture.txt"], workspace: box.workspace)
        #expect(loaded.images.count == 1)
        #expect(loaded.text.isEmpty)
    }

    @Test func tooManyFilesAreRefused() {
        let paths = (0 ... FileInputs.maxFiles).map { "/tmp/f\($0)" }
        #expect(throws: FileInputs.Failure.self) { try FileInputs.load(paths, workspace: nil) }
    }
}

@Suite struct ResponseFormatArgumentTests {
    @Test @MainActor func aSchemaObjectBecomesAConstraint() throws {
        let format = try APIServer.responseFormat([
            "type": "object",
            "properties": ["name": ["type": "string"], "age": ["type": "integer"]],
            "required": ["name", "age"],
        ])
        #expect(format != nil)
    }

    @Test @MainActor func noSchemaMeansNoConstraint() throws {
        #expect(try APIServer.responseFormat(nil) == nil)
    }

    @Test @MainActor func anUnsupportedSchemaIsAnError() {
        #expect(throws: (any Error).self) {
            try APIServer.responseFormat(["type": "object", "patternProperties": ["^x": ["type": "string"]]])
        }
    }
}
