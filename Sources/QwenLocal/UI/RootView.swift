import SwiftUI

/// Wizard until it completes, then the tabbed main window. Reopening the window later shows
/// the chat, never the wizard again.
struct RootView: View {
    @EnvironmentObject private var state: AppState
    @ObservedObject var settings: AppSettings

    var body: some View {
        if settings.wizardCompleted {
            MainWindowView(
                chat: state.chat,
                engine: state.engine,
                settings: settings,
                downloader: state.downloader,
                api: state.api)
        } else {
            WizardView(downloader: state.downloader)
        }
    }
}

struct MainWindowView: View {
    @ObservedObject var chat: ChatStore
    @ObservedObject var engine: EngineController
    @ObservedObject var settings: AppSettings
    @ObservedObject var downloader: ModelDownloader
    @ObservedObject var api: APIServer

    var body: some View {
        TabView {
            ChatView(chat: chat, engine: engine)
                .tabItem { Label("Чат", systemImage: "bubble.left.and.bubble.right") }
            ModelView(engine: engine, settings: settings, downloader: downloader)
                .tabItem { Label("Модель", systemImage: "cpu") }
            SettingsView(settings: settings, engine: engine, api: api)
                .tabItem { Label("Настройки", systemImage: "gearshape") }
        }
        .frame(minWidth: 720, minHeight: 520)
    }
}
