# FeyntStream

Feynt's low-memory runtime: Qwen3.6-35B-A3B with its routed experts read from SSD on demand,
so the process holds about 1.4 GB of resident weights plus a fixed expert cache (4 or 8 GB
in total) instead of 20 GB.

It is plain Swift and Metal, not MLX. With MLX, finding out which experts a layer wants
means stopping the lazy graph after its router; measured on this model that costs about
0.43 ms per layer, 17 ms per token over 40 layers, which alone caps decode near 31 tok/s
before a byte is read. A hand-built command-buffer pipeline, as in TurboFieldfare (see
Vendor/TurboFieldfare), pays far less per layer.

- `Weights/` - safetensors index, resident tensors, Qwen3.5 tensor names.
- `Streaming/` - the per-layer expert slot cache (LRU, parallel `pread`).
