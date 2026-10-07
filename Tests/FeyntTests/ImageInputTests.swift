import Foundation
import Testing
@testable import Feynt

/// A 1x1 PNG.
private let pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

@Suite struct ImageInputTests {
    @Test func dataURLDecodes() throws {
        let data = try ImageInput.decode("data:image/png;base64,\(pixel)")
        #expect(data.starts(with: [0x89, 0x50, 0x4E, 0x47]))
    }

    @Test func remoteURLIsRefusedNotFetched() {
        #expect(throws: ImageInput.Failure.self) { try ImageInput.decode("https://example.com/a.png") }
        #expect(throws: ImageInput.Failure.self) { try ImageInput.decode("file:///etc/passwd") }
    }

    @Test func dataURLMustBeBase64() {
        #expect(throws: ImageInput.Failure.self) { try ImageInput.decode("data:image/png,rawbytes") }
    }

    @Test func bytesThatAreNotAnImageAreRefused() {
        let junk = Data("not a picture".utf8).base64EncodedString()
        #expect(throws: ImageInput.Failure.self) { try ImageInput.decode(base64: junk) }
    }

    @Test func oversizedIsRefused() {
        #expect(throws: ImageInput.Failure.self) {
            try ImageInput.validated(Data(count: ImageInput.maxBytes + 1))
        }
    }

    @Test func imagesReachTheChatTemplateOnUserTurnsOnly() throws {
        let png = try ImageInput.decode(base64: pixel)
        let messages = ToolBridge.messages(from: [
            EngineTurn(role: .system, content: "s"),
            EngineTurn(role: .user, content: "what is this?", images: [png]),
            EngineTurn(role: .assistant, content: "a pixel"),
        ])
        #expect(messages.map(\.images.count) == [0, 1, 0])
    }

    @Test func aChatMessageCarriesItsImagesIntoTheTurn() throws {
        let png = try ImageInput.decode(base64: pixel)
        let message = ChatMessage(role: .user, text: "look", images: [png])
        #expect(message.engineTurn.images == [png])
    }
}
