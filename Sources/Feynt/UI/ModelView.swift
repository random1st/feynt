import SwiftUI

struct ModelView: View {
    @EnvironmentObject private var state: AppState
    @ObservedObject var engine: EngineController
    @ObservedObject var settings: AppSettings
    @ObservedObject var downloader: ModelDownloader

    @State private var sizes: [String: Int64] = [:]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("Активная модель").font(.headline)
                ForEach(ModelCatalog.all) { spec in
                    row(for: spec)
                }

                Divider()
                Text("Драфтер").font(.headline)
                drafterRow

                if downloader.isDownloading {
                    ProgressView(value: downloader.fraction) {
                        Text(downloader.detail).lineLimit(1)
                    }
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .task { await measure() }
    }

    private func row(for spec: ModelSpec) -> some View {
        let active = settings.selectedModelID == spec.id
        let location = ModelResolver.installedLocation(for: spec)
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: active ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(active ? .green : .secondary)
                Text(spec.title).font(.body).bold()
                Text(spec.subtitle).foregroundStyle(.secondary)
                Spacer()
                Button(active ? "Активна" : "Переключить") { switchTo(spec) }
                    .disabled(active || engine.state.isBusy || downloader.isDownloading)
            }
            Text(location?.path ?? "не скачана")
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            if let size = sizes[spec.id], size > 0 {
                Text("На диске: \(Paths.formatBytes(size))").font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.25)))
    }

    /// The drafter belongs to whichever model is selected, so this row follows the
    /// selection rather than naming a single shared accelerator.
    private var drafterRow: some View {
        let spec = ModelCatalog.model(id: settings.selectedModelID) ?? ModelCatalog.uncensored
        return VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(spec.drafter.title)
                Spacer()
                Text(engine.speculative ? "используется" : "не активен")
                    .foregroundStyle(engine.speculative ? .green : .secondary)
            }
            Text(ModelResolver.installedLocation(for: spec.drafter)?.path ?? "не скачан")
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            if engine.stats.acceptedPerStep > 0 {
                Text(String(format: "Принято токенов за шаг: %.2f", engine.stats.acceptedPerStep))
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.25)))
    }

    /// Switching stops the current model first, downloads if needed, then loads the other one.
    private func switchTo(_ spec: ModelSpec) {
        let missing = state.missingArtifacts(for: spec)
        settings.selectedModelID = spec.id
        if missing.isEmpty {
            Task { await engine.switchTo(spec) }
        } else {
            downloader.download(missing) { success in
                guard success else { return }
                Task { await engine.switchTo(spec) }
            }
        }
    }

    private func measure() async {
        var result: [String: Int64] = [:]
        for spec in ModelCatalog.all {
            if let url = ModelResolver.installedLocation(for: spec) {
                result[spec.id] = Paths.directorySize(url)
            }
        }
        sizes = result
    }
}
