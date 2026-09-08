import Foundation
import SwiftUI

struct ChatMessage: Identifiable, Equatable {
    let id = UUID()
    var role: EngineTurn.Role
    var text: String = ""
    var reasoning: String = ""
    var isStreaming = false
}

/// Conversation state plus the streaming loop that feeds it.
@MainActor
final class ChatStore: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var draft: String = ""
    @Published var thinkingEnabled: Bool
    @Published private(set) var isStreaming = false
    @Published var errorMessage: String?

    private let engine: EngineController
    private let settings: AppSettings
    private var streamTask: Task<Void, Never>?

    init(engine: EngineController, settings: AppSettings) {
        self.engine = engine
        self.settings = settings
        thinkingEnabled = settings.thinkingByDefault
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
        let history = messages.dropLast().map { EngineTurn(role: $0.role, content: $0.text) }
        let thinking = thinkingEnabled

        streamTask = Task { [weak self] in
            guard let self else { return }
            guard await self.engine.ensureLoaded() else {
                self.finishStream(error: "Модель не загружена")
                return
            }
            do {
                let options = self.engine.uiOptions(thinking: thinking)
                let stream = try await self.engine.generate(
                    turns: Array(history), options: options)
                let started = Date()
                var produced = 0
                for await event in stream {
                    if Task.isCancelled { break }
                    switch event {
                    case .text(let chunk):
                        self.append(text: chunk, to: placeholder.id)
                        produced += chunk.count
                    case .reasoning(let chunk):
                        self.append(reasoning: chunk, to: placeholder.id)
                    case .finished(let stats):
                        self.engine.generationFinished(stats)
                    }
                    let elapsed = Date().timeIntervalSince(started)
                    if elapsed > 1 {
                        // Rough live rate for the menu bar; the exact count arrives with .finished.
                        self.engine.reportLiveRate(Double(produced) / 4.0 / elapsed)
                    }
                }
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
                messages[index].text = "(модель израсходовала бюджет токенов на размышления)"
            }
        }
        errorMessage = error
        engine.generationFinished(nil)
    }
}
