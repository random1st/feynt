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
                Text("Active model").font(.headline)
                ForEach(ModelCatalog.all) { spec in
                    row(for: spec)
                }

                Divider()
                Text("Drafter").font(.headline)
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
                Button(active ? "Active" : "Switch") { switchTo(spec) }
                    .disabled(active || engine.state.isBusy || downloader.isDownloading)
                if !active, engine.loadedModels.contains(where: { $0.id == spec.id }) {
                    Text("in memory").font(.caption).foregroundStyle(.green)
                }
                // The window that lists what is on disk is the place to say "this copy is
                // wrong, fetch it again" - it was reachable only from the menu bar and the
                // chat, which is not where anyone looks for it.
                if location != nil {
                    Button("Re-download") { state.redownload(spec) }
                        .disabled(engine.state.isBusy || downloader.isDownloading)
                }
            }
            Text(location?.path ?? "not downloaded")
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            if let size = sizes[spec.id], size > 0 {
                Text("On disk: \(Paths.formatBytes(size))").font(.caption)
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
                Text(engine.speculative ? "in use" : "inactive")
                    .foregroundStyle(engine.speculative ? .green : .secondary)
                // The drafter travels with its model, so re-fetching one re-fetches both.
                Button("Re-download") { state.redownload(spec) }
                    .disabled(engine.state.isBusy || downloader.isDownloading)
            }
            Text(ModelResolver.installedLocation(for: spec.drafter)?.path ?? "not downloaded")
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            if engine.stats.acceptedPerStep > 0 {
                Text(String(format: "Accepted tokens per round: %.2f", engine.stats.acceptedPerStep))
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.25)))
    }

    private func switchTo(_ spec: ModelSpec) {
        state.switchModel(to: spec)
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
