import Foundation
import Testing
@testable import Feynt

/// Stands in for the engine: each round asks `script` what to say given the conversation
/// and the options, and records what it was given.
@MainActor
final class ScriptedModel {
    var rounds: [(turns: [EngineTurn], offered: Set<String>)] = []
    let script: (_ round: Int, _ offered: Bool) -> [EngineEvent]

    init(_ script: @escaping (_ round: Int, _ offered: Bool) -> [EngineEvent]) {
        self.script = script
    }

    func generate(_ turns: [EngineTurn], _ options: GenerationOptions) -> AsyncStream<EngineEvent> {
        let offered = Set((options.tools ?? []).compactMap {
            ($0["function"] as? [String: any Sendable])?["name"] as? String
        })
        rounds.append((turns, offered))
        let events = script(rounds.count - 1, !offered.isEmpty)
        return AsyncStream { continuation in
            for event in events { continuation.yield(event) }
            continuation.yield(.finished(GenerationStats(generatedTokens: 1)))
            continuation.finish()
        }
    }
}

private func toolCall(_ name: String, _ arguments: String, id: String = UUID().uuidString) -> EngineEvent {
    .toolCall(EngineToolCall(id: id, name: name, argumentsJSON: arguments))
}

@MainActor
private func run(_ model: ScriptedModel, tools: LocalTools?) async throws -> LocalCompletion {
    try await APIServer.runToolLoop(
        turns: [EngineTurn(role: .user, content: "What is the answer?")],
        spec: ModelCatalog.stock, maxTokens: 64, tools: tools,
        generate: { turns, options in model.generate(turns, options) },
        onText: nil, onToolUse: nil)
}

@Suite @MainActor struct ToolLoopTests {
    @Test func toolResultGoesBackAndTheModelAnswers() async throws {
        let box = try Sandbox()
        defer { box.remove() }
        let model = ScriptedModel { round, _ in
            round == 0
                ? [.text("Let me look. "), toolCall("read_file", #"{"path":"src/main.swift"}"#, id: "c1")]
                : [.text("It is 42.")]
        }
        let result = try await run(model, tools: LocalTools(workspace: box.workspace))

        #expect(result.text == "It is 42.")
        #expect(result.transcript == "Let me look. It is 42.")
        #expect(result.toolCalls.count == 1)
        #expect(result.stats.generatedTokens == 2)
        #expect(model.rounds.count == 2)

        let second = model.rounds[1].turns
        #expect(second.first?.role == .system)  // the hint naming the folder
        #expect(second.first?.content.contains(box.workspace.root.path) == true)
        let assistant = try #require(second.first { $0.role == .assistant })
        #expect(assistant.toolCalls.map(\.id) == ["c1"])
        let tool = try #require(second.last)
        #expect(tool.role == .tool && tool.toolCallID == "c1" && tool.toolName == "read_file")
        #expect(tool.content.contains("let answer = 42"))
    }

    @Test func aModelThatNeverStopsIsCutAtTheBudget() async throws {
        let box = try Sandbox()
        defer { box.remove() }
        // Asks for a tool whenever it is offered one; answers once it is not.
        let model = ScriptedModel { _, offered in
            offered ? [toolCall("list_files", "{}")] : [.text("Here is what I found.")]
        }
        let result = try await run(model, tools: LocalTools(workspace: box.workspace))

        #expect(result.toolCalls.count == APIServer.toolCallBudget)
        #expect(result.text == "Here is what I found.")
        #expect(model.rounds.count == APIServer.toolCallBudget + 1)
        #expect(model.rounds.last?.offered.isEmpty == true)
    }

    @Test func callsPastTheBudgetInOneRoundAreAnsweredNotRun() async throws {
        let box = try Sandbox()
        defer { box.remove() }
        let model = ScriptedModel { round, _ in
            round == 0 ? (0 ..< 8).map { _ in toolCall("list_files", "{}") } : [.text("done")]
        }
        let result = try await run(model, tools: LocalTools(workspace: box.workspace))

        #expect(result.toolCalls.count == 8)
        #expect(result.toolCalls.prefix(6).allSatisfy { !$0.isError })
        #expect(result.toolCalls.suffix(2).allSatisfy { $0.isError && $0.result.contains("budget") })
        // Every call still has a result to pair with; templates reject an unanswered one.
        #expect(model.rounds[1].turns.filter { $0.role == .tool }.count == 8)
    }

    @Test func anUnknownToolIsAnErrorTheModelSees() async throws {
        let model = ScriptedModel { round, _ in
            round == 0 ? [toolCall("run_shell", #"{"cmd":"rm -rf /"}"#)] : [.text("ok")]
        }
        let result = try await run(model, tools: LocalTools(workspace: nil))
        #expect(result.toolCalls.first?.isError == true)
        #expect(result.toolCalls.first?.result.contains("No such tool") == true)
    }

    @Test func fileToolsAreNotOfferedWithoutAFolder() async throws {
        let model = ScriptedModel { _, _ in [.text("hi")] }
        _ = try await run(model, tools: LocalTools(workspace: nil))
        #expect(model.rounds[0].offered == ["web_fetch"])
    }

    @Test func withoutToolsNothingIsOfferedOrRun() async throws {
        let model = ScriptedModel { _, _ in [.text("plain"), toolCall("read_file", "{}")] }
        let result = try await run(model, tools: nil)
        #expect(result.text == "plain")
        #expect(result.toolCalls.isEmpty)
        #expect(model.rounds.count == 1)
        #expect(model.rounds[0].offered.isEmpty)
        #expect(model.rounds[0].turns.first?.role == .user)  // no hint without tools
    }

    @Test func anExistingSystemPromptIsKept() async throws {
        let model = ScriptedModel { _, _ in [.text("hi")] }
        _ = try await APIServer.runToolLoop(
            turns: [EngineTurn(role: .system, content: "Be terse."), EngineTurn(role: .user, content: "hi")],
            spec: ModelCatalog.stock, maxTokens: 64, tools: LocalTools(workspace: nil),
            generate: { turns, options in model.generate(turns, options) },
            onText: nil, onToolUse: nil)
        #expect(model.rounds[0].turns.map(\.content) == ["Be terse.", "hi"])
    }
}
