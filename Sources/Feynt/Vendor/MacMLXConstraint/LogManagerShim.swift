// Not part of mac-mlx: the vendored processor logs through mac-mlx's LogManager, and the two
// calls it makes are routed to Feynt's log here so the files themselves stay as published.

enum LogManager {
    static let shared = Shim()

    struct Shim {
        enum Level { case error }
        enum Category { case error }

        func logSync(_ message: String, level: Level, category: Category) {
            AppLog.write(message)
        }
    }
}
