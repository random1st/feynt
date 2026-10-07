Tests and fixtures in this folder come from [magicnight/mac-mlx](https://github.com/magicnight/mac-mlx),
commit `0a5fa30f938c33deaa19b57fc6d2de570550fe4c`, `MacMLXCore/Tests/MacMLXCoreTests/Constraint/` and `Fixtures/`,
Apache License 2.0 (see `Sources/Feynt/Vendor/MacMLXConstraint/LICENSE`). Changed: imports point
at the Feynt module. Left out: `JSONConstraintProcessorTests` and `WhitespaceSuppressionTests`,
which need the MLX runtime that a SwiftPM test run does not load, and `StructuredOutputModelTests`,
which needs a downloaded model.
