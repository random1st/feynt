import Testing
@testable import Feynt

@Suite struct ContextLimitTests {
    @Test func withinTheLimitPasses() throws {
        var options = GenerationOptions(maxTokens: 500)
        options.contextLimit = 56_000
        try options.checkContext(promptTokens: 55_500)
    }

    @Test func promptPlusReplyPastTheLimitIsRefused() {
        var options = GenerationOptions(maxTokens: 500)
        options.contextLimit = 56_000
        #expect(throws: EngineError.self) { try options.checkContext(promptTokens: 55_501) }
    }

    @Test func noLimitMeansNoCheck() throws {
        try GenerationOptions(maxTokens: 500).checkContext(promptTokens: 1_000_000)
    }

    @Test func onlyTheSmallModelsAreCapped() {
        #expect(ModelCatalog.small.contextLimit == 56_000)
        #expect(ModelCatalog.tiny.contextLimit == 120_000)
        #expect(ModelCatalog.all.filter { $0.contextLimit != nil }.map(\.id) == ["small", "tiny"])
    }
}
