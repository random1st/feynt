import SwiftUI

struct SettingsView: View {
    @ObservedObject var settings: AppSettings
    @ObservedObject var engine: EngineController
    @ObservedObject var api: APIServer

    var body: some View {
        Form {
            Section("Server") {
                TextField("Port", value: $settings.port, format: .number.grouping(.never))
                    .onSubmit { api.restartIfRunning() }
                LabeledContent("Endpoint", value: api.baseURL)
                HStack {
                    Text(api.isRunning ? "Running" : "Stopped")
                        .foregroundStyle(api.isRunning ? .green : .secondary)
                    Spacer()
                    Button("Copy URL") { Pasteboard.copy(api.baseURL) }
                }
                if let error = api.lastError {
                    Text(error).font(.caption).foregroundStyle(.red)
                }
                Text("The server starts with the model and listens on 127.0.0.1 only.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Generating") {
                Toggle("Reasoning by default", isOn: $settings.thinkingByDefault)
                Stepper(
                    "Reply length: \(settings.maxTokens) tokens",
                    value: $settings.maxTokens, in: 256 ... 32768, step: 256)
            }

            Section("Memory") {
                Picker("Unload the model when idle", selection: $settings.idleTimeout) {
                    ForEach(AppSettings.idleChoices, id: \.self) { value in
                        Text(AppSettings.idleLabel(value)).tag(value)
                    }
                }
                Text("Unloading returns about 16 GB: the weights are released and the MLX cache is cleared.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Diagnostics") {
                LabeledContent("Status", value: engine.statusText)
                LabeledContent("Speculative decoding", value: engine.speculative ? "on" : "off")
                LabeledContent("Log", value: Paths.logFile.path)
                Button("Open log") { openLog() }
            }
        }
        .formStyle(.grouped)
        .padding(4)
    }

    private func openLog() {
        Paths.ensureDirectory(Paths.logDirectory)
        NSWorkspace.shared.open(Paths.logFile)
    }
}
