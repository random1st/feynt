import Foundation
import SwiftUI

struct ChatMessage: Identifiable, Equatable {
    let id = UUID()
    var role: EngineTurn.Role
    var text: String = ""
    var reasoning: String = ""
    var isStreaming = false
    /// An assistant turn that asked for tools carries the calls, and each `.tool` message
    /// answers one of them; both go back to the model with the history.
    var toolCalls: [EngineToolCall] = []
    var toolCallID: String? = nil
    var toolName: String? = nil
    var toolArguments: String? = nil
    var isError = false

    var engineTurn: EngineTurn {
        EngineTurn(role: role, content: text, toolCalls: toolCalls, toolCallID: toolCallID, toolName: toolName)
    }
}

/// Conversation state plus the streaming loop that feeds it.
@MainActor
final class ChatStore: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var draft: String = ""
    @Published var thinkingEnabled: Bool
    @Published private(set) var isStreaming = false
    @Published var errorMessage: String?
    /// Whether the model may use its read-only tools. On by default: they only look things
    /// up, and a question about a file or a page is answered from the file or the page.
    @Published var toolsEnabled: Bool {
        didSet { UserDefaults.standard.set(toolsEnabled, forKey: "chatToolsEnabled") }
    }
    /// The folder the file tools read. Without one the model can still fetch public pages.
    @Published var workspace: Workspace? {
        didSet { UserDefaults.standard.set(workspace?.root.path, forKey: "chatWorkspace") }
    }

    private let engine: EngineController
    private let settings: AppSettings
    private var streamTask: Task<Void, Never>?

    init(engine: EngineController, settings: AppSettings) {
        self.engine = engine
        self.settings = settings
        thinkingEnabled = settings.thinkingByDefault
        toolsEnabled = (UserDefaults.standard.object(forKey: "chatToolsEnabled") as? Bool) ?? true
        // A folder that has since been moved or deleted is forgotten rather than offered.
        if let saved = UserDefaults.standard.string(forKey: "chatWorkspace"),
            FileManager.default.fileExists(atPath: saved)
        {
            workspace = Workspace(URL(fileURLWithPath: saved))
        } else {
            workspace = nil
        }
    }

    var canSend: Bool {
        !isStreaming && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func send() {
        let prompt = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !isStreaming else { return }
        draft = ""
        errorMessage = nil
        messages.append(ChatMessage(role: .user, text: prompt))
        let placeholder = ChatMessage(role: .assistant, isStreaming: true)
        messages.append(placeholder)
        isStreaming = true

        // Prior turns go back with every request; the engine keeps no session of its own.
        let history = messages.dropLast().map(\.engineTurn)
        let thinking = thinkingEnabled
        let tools = toolsEnabled ? LocalTools(workspace: workspace) : nil
        let spec = settings.selectedModel

        streamTask = Task { [weak self] in
            guard let self else { return }
            guard await self.engine.ensureLoaded() else {
                self.finishStream(error: "No model loaded")
                return
            }
            var current = placeholder.id
            var needsBubble = false
            let started = Date()
            var produced = 0
            do {
                let result = try await APIServer.runToolLoop(
                    turns: Array(history), spec: spec, maxTokens: self.settings.maxTokens,
                    tools: tools, thinking: thinking,
                    generate: { [engine = self.engine] turns, options in
                        try await engine.generate(turns: turns, options: options)
                    },
                    onText: { chunk in
                        // A tool round ends one assistant bubble; the text after it opens
                        // the next, so each tool row sits between what led to it and what
                        // followed.
                        if needsBubble {
                            let bubble = ChatMessage(role: .assistant, isStreaming: true)
                            self.messages.append(bubble)
                            current = bubble.id
                            needsBubble = false
                        }
                        self.append(text: chunk, to: current)
                        produced += chunk.count
                        let elapsed = Date().timeIntervalSince(started)
                        if elapsed > 1 {
                            self.engine.reportLiveRate(Double(produced) / 4.0 / elapsed)
                        }
                    },
                    onReasoning: { chunk in self.append(reasoning: chunk, to: current) },
                    onToolUse: { use in
                        if let index = self.messages.firstIndex(where: { $0.id == current }) {
                            self.messages[index].toolCalls.append(use.call)
                            self.messages[index].isStreaming = false
                        }
                        self.messages.append(ChatMessage(
                            role: .tool, text: use.result, toolCallID: use.call.id,
                            toolName: use.call.name, toolArguments: use.call.argumentsJSON,
                            isError: use.isError))
                        needsBubble = true
                    })
                self.engine.generationFinished(result.stats)
                self.finishStream(error: nil)
            } catch is CancellationError {
                self.finishStream(error: nil)
            } catch {
                self.finishStream(error: error.localizedDescription)
            }
        }
    }

    func stop() {
        streamTask?.cancel()
        streamTask = nil
        finishStream(error: nil)
    }

    func clear() {
        stop()
        messages.removeAll()
        errorMessage = nil
    }

    private func append(text: String, to id: UUID) {
        guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[index].text += text
    }

    private func append(reasoning: String, to id: UUID) {
        guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[index].reasoning += reasoning
    }

    private func finishStream(error: String?) {
        isStreaming = false
        if let index = messages.indices.last {
            messages[index].isStreaming = false
            // With thinking on, the model can spend the whole budget inside its reasoning and
            // return no answer at all. That is normal output, not a failure — say so instead
            // of leaving an empty bubble.
            if messages[index].role == .assistant, messages[index].text.isEmpty,
               !messages[index].reasoning.isEmpty
            {
                messages[index].text = "(the model spent its token budget reasoning)"
            }
        }
        errorMessage = error
        engine.generationFinished(nil)
    }
}
