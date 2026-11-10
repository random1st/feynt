/// Tensor names of a Qwen3.5-MoE MLX checkpoint, as the low-memory runtime reads them.
public enum Qwen35Names {
    public static let prefix = "language_model.model."

    public static func layer(_ i: Int) -> String { "\(prefix)layers.\(i)." }

    public static func expert(_ layer: Int, _ part: String) -> String {
        "\(Self.layer(layer))mlp.switch_mlp.\(part)"
    }

    /// Resident: the language model without its routed experts. The vision tower, the MTP
    /// head (absent in this checkpoint) and the experts are not loaded.
    public static func isResident(_ name: String) -> Bool {
        name.hasPrefix("language_model.") && !name.contains(".switch_mlp.") && !name.contains(".mtp.")
    }
}
