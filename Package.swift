// swift-tools-version:5.9
import PackageDescription

// mlx-swift-lm comes from a sibling checkout, not from GitHub: the Qwen3.5 MTP drafter
// (Qwen35MTP.swift + Qwen35TextMTPRegistration.swift) is not in any released tag, and that
// tree carries the local patch for the DFlash2 drafter port. mlx-swift itself is pulled
// transitively from GitHub by that manifest.
let package = Package(
    name: "Feynt",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../mlx-swift-lm"),
        // The DFlash 2 drafter that replaced the MTP path; a sibling checkout for the same
        // reason — it is built against the multilayer-tap patch in mlx-swift-lm above.
        .package(path: "../dflash-swift"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "Feynt",
            dependencies: [
                .product(name: "DFlashKit", package: "dflash-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/Feynt")
    ]
)
