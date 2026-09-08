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
            Text("Set up Feynt").font(.title2).bold()
            Text("Step \(step + 1) of 3").foregroundStyle(.secondary).font(.callout)
        }
    }

    // MARK: - Step 1: resources

    private var requirementsStep: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("System check").font(.headline)
            Text("The model runs inside the app through MLX — no external programs needed.")
                .foregroundStyle(.secondary)

            checkRow(
                ok: ramBytes >= 24_000_000_000,
                title: "Memory",
                detail: "\(Paths.formatBytes(ramBytes)) — 24 GB or more recommended")
            checkRow(
                ok: (freeBytes ?? 0) >= requiredBytes,
                title: "Free space",
                detail: "\(freeBytes.map(Paths.formatBytes) ?? "unknown") free, ~\(Paths.formatBytes(requiredBytes)) needed")

            if ramBytes < 24_000_000_000 {
                Label(
                    "Low memory: a 27B model in 4-bit takes ~16 GB. Expect swapping and slowdowns.",
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
            Text("Choose a model").font(.headline)
            // Seven cards do not fit the window, and an unscrollable VStack does not clip
            // the overflow at the bottom - it pushes the first row up out of sight, so the
            // model the wizard has already selected is the one the user cannot see.
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(ModelCatalog.all) { spec in
                        ModelCard(spec: spec, selected: chosen.id == spec.id) { chosen = spec }
                    }
                }
                .padding(.trailing, 4)
            }
            Text("Free on disk: \(freeBytes.map(Paths.formatBytes) ?? "unknown")")
                .font(.callout).foregroundStyle(.secondary)
        }
    }

    // MARK: - Step 3: download

    private var downloadStep: some View {
        VStack(alignment: .leading, spacing: 14) {
            if downloadFinished || state.missingArtifacts(for: chosen).isEmpty {
                Text("Ready").font(.headline)
                Text("\(chosen.title) is ready. The app lives in the menu bar.")
                    .foregroundStyle(.secondary)
                if let path = ModelResolver.installedLocation(for: chosen)?.path {
                    HStack {
                        Text(path).font(.system(.callout, design: .monospaced)).lineLimit(1)
                        Button("Copy") { copy(path) }
                    }
                }
            } else {
                Text("Loading").font(.headline)
                Text(downloader.detail.isEmpty ? "Preparing…" : downloader.detail)
                    .foregroundStyle(.secondary).lineLimit(1)
                ProgressView(value: downloader.fraction)
                if let error = downloader.errorMessage {
                    Label(error, systemImage: "xmark.octagon").foregroundStyle(.red)
                }
                Button("Cancel") { downloader.cancel() }
                    .disabled(!downloader.isDownloading)
            }
        }
    }

    // MARK: - Footer

    private var footer: some View {
        HStack {
            if step > 0 {
                Button("Back") { step -= 1 }.disabled(downloader.isDownloading)
            }
            Spacer()
            switch step {
            case 0:
                Button("Next") { step = 1 }
            case 1:
                Button(nextTitle) { startStepThree() }
            default:
                Button("Finish") { state.finishWizard(with: chosen) }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!(downloadFinished || state.missingArtifacts(for: chosen).isEmpty))
            }
        }
    }

    private var nextTitle: String {
        state.missingArtifacts(for: chosen).isEmpty ? "Next" : "Download"
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
                            Label("already downloaded", systemImage: "internaldrive")
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
