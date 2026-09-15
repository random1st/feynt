import Foundation
import MLXLMCommon

/// Everything that turns OpenAI-shaped tool traffic into what a chat template expects, and
/// the model's answer back into tool calls. Shared by both engines so a request behaves the
/// same whether or not speculation is on.
enum ToolBridge {

    /// Which dialect this model speaks. Resolution order matters: the loader knows best when
    /// it knows at all, and it only knows for models named in its registry — Feynt loads from
    /// a directory, so for everything else the chat template is the evidence. Guessing `.json`
    /// for an LFM2 model would parse none of its `<|tool_call_start|>` calls and leak the
    /// protocol into the answer instead.
    static func format(for context: ModelContext, directory: URL?) -> ToolCallFormat {
        if let declared = context.configuration.toolCallFormat { return declared }
        if let template = chatTemplate(in: directory),
            let inferred = ToolCallFormat.inferred(fromChatTemplate: template)
        {
            return inferred
        }
        return .json
    }

    /// Conversation turns as the chat template wants them, with tool traffic attached.
    ///
    /// An assistant turn that asked for tools has to carry the calls, and a `tool` turn has
    /// to name the call it answers — templates correlate on one or the other, so both go in.
    static func messages(from turns: [EngineTurn]) -> [Chat.Message] {
        turns.map { turn in
            switch turn.role {
            case .system: return .system(turn.content)
            case .user: return .user(turn.content)
            case .assistant:
                guard !turn.toolCalls.isEmpty else { return .assistant(turn.content) }
                return Chat.Message(
                    role: .assistant, content: turn.content,
                    tool: .calls(turn.toolCalls.map(toolCall(from:))))
            case .tool:
                return Chat.Message(
                    role: .tool, content: turn.content,
                    tool: .result(id: turn.toolCallID ?? "", name: turn.toolName))
            }
        }
    }

    /// The template renders arguments from a dictionary, so the JSON text the client sent is
    /// decoded here rather than passed through. Unparseable arguments become an empty object:
    /// dropping the call would silently rewrite the conversation's history.
    private static func toolCall(from call: EngineToolCall) -> ToolCall {
        let data = Data(call.argumentsJSON.utf8)
        let arguments = (try? JSONDecoder().decode([String: JSONValue].self, from: data)) ?? [:]
        return ToolCall(function: .init(name: call.name, arguments: arguments), id: call.id)
    }

    /// A parsed call on its way out to a client: arguments back to JSON text, and an id the
    /// client can correlate, because not every format carries one.
    static func engineCall(from call: ToolCall) -> EngineToolCall {
        let data = (try? JSONEncoder().encode(call.function.arguments)) ?? Data("{}".utf8)
        return EngineToolCall(
            id: call.id ?? "call_\(UUID().uuidString.prefix(24))",
            name: call.function.name,
            argumentsJSON: String(data: data, encoding: .utf8) ?? "{}")
    }

    /// The template as the tokenizer files carry it: newer exports put it in its own
    /// `chat_template.jinja`, older ones keep it inside `tokenizer_config.json`.
    private static func chatTemplate(in directory: URL?) -> String? {
        guard let directory else { return nil }
        let jinja = directory.appending(path: "chat_template.jinja")
        if let text = try? String(contentsOf: jinja, encoding: .utf8), !text.isEmpty {
            return text
        }
        guard
            let data = try? Data(contentsOf: directory.appending(path: "tokenizer_config.json")),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        if let text = root["chat_template"] as? String { return text }
        // A few exports ship a list of named templates; the default one drives generation.
        guard let variants = root["chat_template"] as? [[String: Any]] else { return nil }
        return (variants.first { $0["name"] as? String == "default" } ?? variants.first)?["template"]
            as? String
    }
}
