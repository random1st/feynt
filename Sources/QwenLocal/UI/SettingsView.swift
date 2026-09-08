import SwiftUI

struct SettingsView: View {
    @ObservedObject var settings: AppSettings
    @ObservedObject var engine: EngineController
    @ObservedObject var api: APIServer

    var body: some View {
        Form {
            Section("Сервер") {
                TextField("Порт", value: $settings.port, format: .number.grouping(.never))
                    .onSubmit { api.restartIfRunning() }
                LabeledContent("Endpoint", value: api.baseURL)
                HStack {
                    Text(api.isRunning ? "Запущен" : "Остановлен")
                        .foregroundStyle(api.isRunning ? .green : .secondary)
                    Spacer()
                    Button("Копировать URL") { Pasteboard.copy(api.baseURL) }
                }
                if let error = api.lastError {
                    Text(error).font(.caption).foregroundStyle(.red)
                }
                Text("Сервер поднимается вместе с моделью и слушает только 127.0.0.1.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Генерация") {
                Toggle("Размышления по умолчанию", isOn: $settings.thinkingByDefault)
                Stepper(
                    "Максимум токенов: \(settings.maxTokens)",
                    value: $settings.maxTokens, in: 256 ... 8192, step: 256)
            }

            Section("Память") {
                Picker("Выгружать модель после простоя", selection: $settings.idleTimeout) {
                    ForEach(AppSettings.idleChoices, id: \.self) { value in
                        Text(AppSettings.idleLabel(value)).tag(value)
                    }
                }
                Text("Выгрузка освобождает ~16 ГБ: ссылки на веса сбрасываются, кэш MLX очищается.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Диагностика") {
                LabeledContent("Состояние", value: engine.statusText)
                LabeledContent("Спекулятивное декодирование", value: engine.speculative ? "вкл" : "выкл")
                LabeledContent("Лог", value: Paths.logFile.path)
                Button("Открыть лог") { openLog() }
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
