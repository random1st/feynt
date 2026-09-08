import SwiftUI

struct ChatView: View {
    @EnvironmentObject private var state: AppState
    @ObservedObject var chat: ChatStore
    @ObservedObject var engine: EngineController

    var body: some View {
        VStack(spacing: 0) {
            transcript
            Divider()
            composer
        }
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(chat.messages) { message in
                        MessageBubble(message: message).id(message.id)
                    }
                    if let error = chat.errorMessage {
                        Label(error, systemImage: "xmark.octagon").foregroundStyle(.red)
                    }
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .onChange(of: chat.messages.count) {
                if let last = chat.messages.last { proxy.scrollTo(last.id, anchor: .bottom) }
            }
        }
    }

    private var composer: some View {
        VStack(spacing: 8) {
            HStack {
                Toggle("Размышления", isOn: $chat.thinkingEnabled)
                    .toggleStyle(.switch)
                    .disabled(chat.isStreaming)
                Spacer()
                Text(engine.statusText).font(.callout).foregroundStyle(.secondary)
                Button("Очистить", action: chat.clear).disabled(chat.messages.isEmpty)
            }
            HStack(alignment: .bottom, spacing: 8) {
                TextEditor(text: $chat.draft)
                    .font(.body)
                    .frame(minHeight: 56, maxHeight: 120)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.secondary.opacity(0.3)))
                if chat.isStreaming {
                    Button("Стоп", action: chat.stop).keyboardShortcut(".", modifiers: .command)
                } else {
                    Button("Отправить", action: chat.send)
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(!chat.canSend)
                }
            }
        }
        .padding(12)
    }
}

private struct MessageBubble: View {
    let message: ChatMessage
    @State private var reasoningExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(message.role == .user ? "Вы" : "Модель")
                .font(.caption).foregroundStyle(.secondary)

            if !message.reasoning.isEmpty {
                DisclosureGroup(isExpanded: $reasoningExpanded) {
                    Text(message.reasoning)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                } label: {
                    Label("Размышления", systemImage: "brain")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            ForEach(Array(MarkdownBlock.parse(message.text).enumerated()), id: \.offset) { _, block in
                switch block {
                case .code(let code):
                    Text(code)
                        .font(.system(.callout, design: .monospaced))
                        .textSelection(.enabled)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.secondary.opacity(0.12))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                case .text(let text):
                    Text(text).textSelection(.enabled)
                }
            }

            if message.isStreaming && message.text.isEmpty && message.reasoning.isEmpty {
                ProgressView().controlSize(.small)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Minimal markdown for v1: fenced code blocks render monospaced, everything else is plain.
enum MarkdownBlock {
    case text(String)
    case code(String)

    static func parse(_ input: String) -> [MarkdownBlock] {
        guard input.contains("```") else {
            return input.isEmpty ? [] : [.text(input)]
        }
        var blocks: [MarkdownBlock] = []
        let segments = input.components(separatedBy: "```")
        for (index, segment) in segments.enumerated() {
            if segment.isEmpty { continue }
            if index % 2 == 1 {
                // Drop a language hint on the fence line, if present.
                var body = segment
                if let newline = body.firstIndex(of: "\n") {
                    let head = body[body.startIndex ..< newline]
                    if !head.contains(" ") { body = String(body[body.index(after: newline)...]) }
                }
                blocks.append(.code(body.trimmingCharacters(in: .newlines)))
            } else {
                blocks.append(.text(segment))
            }
        }
        return blocks
    }
}
