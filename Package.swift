// swift-tools-version:5.9
import PackageDescription

// mlx-swift-lm comes from a sibling checkout, not from GitHub: the Qwen3.5 MTP drafter
// (Qwen35MTP.swift + Qwen35TextMTPRegistration.swift) is not in any released tag, and that
// tree carries the local patch for the DFlash2 drafter port. mlx-swift itself is pulled
// transitively from GitHub by that manifest.
let package = Package(
    name: "QwenLocal",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../mlx-swift-lm"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "QwenLocal",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/QwenLocal")
    ]
)
