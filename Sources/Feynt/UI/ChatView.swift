import SwiftUI

struct ChatView: View {
    @EnvironmentObject private var state: AppState
    @ObservedObject var chat: ChatStore
    @ObservedObject var engine: EngineController
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(spacing: 0) {
            modelBar
            Divider()
            transcript
            Divider()
            composer
        }
    }

    /// Which model is loaded, how to put another one in its place, and how to get it out of
    /// memory - at the top of the window where it is being used, not three clicks away in a
    /// settings pane. Ejecting matters here: the weights are sixteen to nineteen gigabytes,
    /// and waiting out the idle timer to get them back is not a thing anyone wants to do.
    private var modelBar: some View {
        HStack(spacing: 10) {
            Menu {
                // Только скачанные: в окне, где идёт разговор, выбор модели должен
                // переключать, а не начинать часовую загрузку. Всё остальное — в окне
                // моделей, где для этого есть место и прогресс.
                ForEach(ModelCatalog.all.filter(ModelResolver.isPresent)) { spec in
                    Button {
                        state.switchModel(to: spec)
                    } label: {
                        if spec.id == state.settings.selectedModelID {
                            Label(spec.title, systemImage: "checkmark")
                        } else if engine.loadedModels.contains(where: { $0.id == spec.id }) {
                            Text("\(spec.title) — in memory")
                        } else {
                            Text(spec.title)
                        }
                    }
                    .disabled(engine.state.isBusy || state.downloader.isDownloading)
                }

                Divider()

                // Re-downloading lives in the model window, next to what is on disk and how
                // much of it. In a chat the menu should switch between loaded models and
                // nothing else - deleting sixteen gigabytes does not belong one slip away
                // from picking a model mid-conversation.
                Button("Manage models…") {
                    NSApp.activate(ignoringOtherApps: true)
                    openWindow(id: MainWindowID.value)
                }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "cpu")
                    Text(state.settings.selectedModel.title).lineLimit(1)
                }
            }
            .menuStyle(.borderlessButton)
            .fixedSize()

            if engine.state == .unloaded {
                Button {
                    Task { await engine.load(state.settings.selectedModel) }
                } label: {
                    Label("Load", systemImage: "play")
                }
                .disabled(engine.state.isBusy || state.downloader.isDownloading)
                .help("Load the weights now instead of on the first message")
            } else {
                Button {
                    Task { await engine.unload() }
                } label: {
                    Label("Unload", systemImage: "eject")
                }
                .disabled(chat.isStreaming || engine.state.isBusy)
                .help("Free the weights from memory")

                // A model can be left in a state a message will not fix - a failed load, a
                // drafter swapped on disk. Unload-then-load is the cure, and asking for it
                // twice by hand is not an interface.
                Button {
                    Task { await engine.reload(state.settings.selectedModel) }
                } label: {
                    Label("Reload", systemImage: "arrow.clockwise")
                }
                .disabled(chat.isStreaming || engine.state.isBusy)
                .help("Unload and load the same model again")
            }

            if state.downloader.isDownloading {
                Text(state.downloader.detail).font(.callout).foregroundStyle(.secondary)
                    .lineLimit(1)
            } else if state.downloader.errorMessage != nil {
                Button {
                    state.switchModel(to: state.settings.selectedModel)
                } label: {
                    Label("Retry download", systemImage: "arrow.clockwise.circle")
                }
                .help(state.downloader.errorMessage ?? "")
            }

            Spacer()
            Text(engine.statusText).font(.callout).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
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
                Toggle("Reasoning", isOn: $chat.thinkingEnabled)
                    .toggleStyle(.switch)
                    .disabled(chat.isStreaming)
                Spacer()
                // The chat is the demo surface: whether speculation is on, and what it buys,
                // belongs here rather than only behind a menu-bar click.
                SpeedReadout(engine: engine)
                Button("Clear", action: chat.clear).disabled(chat.messages.isEmpty)
            }
            HStack(alignment: .bottom, spacing: 8) {
                // A TextEditor swallows Return as a newline, so a plain Enter never sent
                // anything and the only way out was a menu-less Cmd+Return nobody guesses.
                // Return now sends, Shift+Return still breaks the line.
                TextEditor(text: $chat.draft)
                    .font(.body)
                    .frame(minHeight: 56, maxHeight: 120)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.secondary.opacity(0.3)))
                    .onKeyPress(.return, phases: .down) { press in
                        guard !press.modifiers.contains(.shift) else { return .ignored }
                        guard chat.canSend else { return .handled }
                        chat.send()
                        return .handled
                    }
                if chat.isStreaming {
                    Button("Stop", action: chat.stop).keyboardShortcut(".", modifiers: .command)
                } else {
                    Button("Send", action: chat.send)
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(!chat.canSend)
                }
            }
        }
        .padding(12)
    }
}

/// Live decode rate and how many drafted tokens the target accepted per round — the two
/// numbers that say whether the speculation is working.
private struct SpeedReadout: View {
    @ObservedObject var engine: EngineController

    var body: some View {
        HStack(spacing: 10) {
            if engine.liveTokensPerSecond > 0 {
                Label(
                    String(format: "%.1f tok/s", engine.liveTokensPerSecond),
                    systemImage: "speedometer")
            }
            if engine.stats.acceptedPerStep > 0 {
                Label(
                    String(format: "%.2f accepted/round", engine.stats.acceptedPerStep),
                    systemImage: "arrow.triangle.branch")
            }
            // Why the second turn of a conversation starts answering so much sooner than
            // the first: most of its prompt was never re-read.
            if engine.stats.cachedPromptTokens > 0 {
                Label(
                    "\(engine.stats.cachedPromptTokens) cached",
                    systemImage: "bolt.horizontal.circle")
            }
            if engine.state != .unloaded {
                Text(engine.speculative ? "speculative" : "plain")
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(
                        Capsule().fill(
                            (engine.speculative ? Color.green : Color.secondary).opacity(0.15)))
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .labelStyle(.titleAndIcon)
    }
}

private struct MessageBubble: View {
    let message: ChatMessage
    @State private var reasoningExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(message.role == .user ? "You" : "Model")
                .font(.caption).foregroundStyle(.secondary)

            if !message.reasoning.isEmpty {
                DisclosureGroup(isExpanded: $reasoningExpanded) {
                    Text(message.reasoning)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                } label: {
                    Label("Reasoning", systemImage: "brain")
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
