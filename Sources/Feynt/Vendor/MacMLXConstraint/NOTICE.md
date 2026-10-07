# Vendored from mac-mlx

The Swift files in this folder, other than `LogManagerShim.swift`, come from
[magicnight/mac-mlx](https://github.com/magicnight/mac-mlx), commit
`0a5fa30f938c33deaa19b57fc6d2de570550fe4c`, `MacMLXCore/Sources/MacMLXCore/Constraint/`.
Copyright © 2026 macMLX, licensed under the Apache License 2.0 (`LICENSE` here).

Changes made for Feynt: an `import MLXLMCommon` line added to files that relied on
mac-mlx's module-wide imports, and `LogManagerShim.swift` added in place of mac-mlx's
logger. The logic is unchanged.

What it does here: grammar-constrained decoding against a JSON schema (`json_schema` on
MCP `generate`, `response_format` on the OpenAI endpoint), so an agent gets JSON it can
parse rather than JSON it has to hope for.
