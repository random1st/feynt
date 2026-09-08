import SwiftUI

/// First-run wizard: capability check, model choice, download. The app does not fall back to
/// a menu-bar-only mode until this completes.
struct WizardView: View {
    @EnvironmentObject private var state: AppState
    @ObservedObject var downloader: ModelDownloader

    @State private var step = 0
    @State private var chosen: ModelSpec = ModelCatalog.uncensored
    @State private var downloadFinished = false

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            header
            Divider()
            Group {
                switch step {
                case 0: requirementsStep
                case 1: modelStep
                default: downloadStep
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            Divider()
            footer
        }
        .padding(24)
        .frame(minWidth: 620, minHeight: 460)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Настройка Feynt").font(.title2).bold()
            Text("Шаг \(step + 1) из 3").foregroundStyle(.secondary).font(.callout)
        }
    }

    // MARK: - Step 1: resources

    private var requirementsStep: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Проверка ресурсов").font(.headline)
            Text("Модель работает внутри приложения через MLX — внешние программы не нужны.")
                .foregroundStyle(.secondary)

            checkRow(
                ok: ramBytes >= 24_000_000_000,
                title: "Оперативная память",
                detail: "\(Paths.formatBytes(ramBytes)) — рекомендуется от 24 ГБ")
            checkRow(
                ok: (freeBytes ?? 0) >= requiredBytes,
                title: "Свободное место",
                detail: "\(freeBytes.map(Paths.formatBytes) ?? "неизвестно") при потребности ~\(Paths.formatBytes(requiredBytes))")

            if ramBytes < 24_000_000_000 {
                Label(
                    "Памяти мало: 27B-модель в 4-битном кванте занимает ~16 ГБ. Возможны свопы и замедление.",
                    systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
            }
        }
    }

    private func checkRow(ok: Bool, title: String, detail: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: ok ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                .foregroundStyle(ok ? .green : .orange)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                Text(detail).font(.callout).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Step 2: model choice

    private var modelStep: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Выберите модель").font(.headline)
            ForEach(ModelCatalog.all) { spec in
                ModelCard(spec: spec, selected: chosen.id == spec.id) { chosen = spec }
            }
            Text("Свободно на диске: \(freeBytes.map(Paths.formatBytes) ?? "неизвестно")")
                .font(.callout).foregroundStyle(.secondary)
        }
    }

    // MARK: - Step 3: download

    private var downloadStep: some View {
        VStack(alignment: .leading, spacing: 14) {
            if downloadFinished || state.missingArtifacts(for: chosen).isEmpty {
                Text("Готово").font(.headline)
                Text("Модель \(chosen.title) готова к работе. Приложение живёт в строке меню.")
                    .foregroundStyle(.secondary)
                if let path = ModelResolver.installedLocation(for: chosen)?.path {
                    HStack {
                        Text(path).font(.system(.callout, design: .monospaced)).lineLimit(1)
                        Button("Копировать") { copy(path) }
                    }
                }
            } else {
                Text("Загрузка").font(.headline)
                Text(downloader.detail.isEmpty ? "Подготовка…" : downloader.detail)
                    .foregroundStyle(.secondary).lineLimit(1)
                ProgressView(value: downloader.fraction)
                if let error = downloader.errorMessage {
                    Label(error, systemImage: "xmark.octagon").foregroundStyle(.red)
                }
                Button("Отменить") { downloader.cancel() }
                    .disabled(!downloader.isDownloading)
            }
        }
    }

    // MARK: - Footer

    private var footer: some View {
        HStack {
            if step > 0 {
                Button("Назад") { step -= 1 }.disabled(downloader.isDownloading)
            }
            Spacer()
            switch step {
            case 0:
                Button("Далее") { step = 1 }
            case 1:
                Button(nextTitle) { startStepThree() }
            default:
                Button("Завершить") { state.finishWizard(with: chosen) }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!(downloadFinished || state.missingArtifacts(for: chosen).isEmpty))
            }
        }
    }

    private var nextTitle: String {
        state.missingArtifacts(for: chosen).isEmpty ? "Далее" : "Скачать"
    }

    private func startStepThree() {
        step = 2
        let missing = state.missingArtifacts(for: chosen)
        guard !missing.isEmpty else {
            downloadFinished = true
            return
        }
        downloader.download(missing) { success in
            Task { @MainActor in downloadFinished = success }
        }
    }

    private var ramBytes: Int64 { Paths.physicalMemoryBytes() }
    private var freeBytes: Int64? { Paths.freeDiskBytes() }
    private var requiredBytes: Int64 {
        state.missingArtifacts(for: chosen).reduce(0) { $0 + $1.approximateBytes }
    }

    private func copy(_ text: String) {
        Pasteboard.copy(text)
    }
}

private struct ModelCard: View {
    let spec: ModelSpec
    let selected: Bool
    let onSelect: () -> Void

    var body: some View {
        Button(action: onSelect) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: selected ? "largecircle.fill.circle" : "circle")
                    .foregroundStyle(selected ? Color.accentColor : .secondary)
                VStack(alignment: .leading, spacing: 4) {
                    Text(spec.title).font(.headline)
                    Text(spec.subtitle).foregroundStyle(.secondary)
                    HStack(spacing: 8) {
                        Text("~\(Paths.formatBytes(spec.approximateBytes))")
                        if ModelResolver.isPresent(spec) {
                            Label("уже скачана", systemImage: "internaldrive")
                                .foregroundStyle(.green)
                        }
                    }
                    .font(.callout)
                }
                Spacer()
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(selected ? Color.accentColor : Color.secondary.opacity(0.3)))
        }
        .buttonStyle(.plain)
    }
}
