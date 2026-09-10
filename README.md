# Feynt

Local language models on Apple Silicon at 2.7x the speed of plain decoding: a small drafter
model pulls a feint, proposing a block of eight tokens, and the large one confirms them in a
single pass. Hence the name.

The app lives in the menu bar. Inside it there is a chat, a first-run wizard and an
OpenAI-compatible server. The engine runs **inside the app** through MLX Swift: no Python,
no child process, nothing to install beforehand.

## Install

```sh
brew install --cask random1st/feynt/feynt
```

The full name taps the repository itself. If Homebrew asks you to trust it first, run
`brew trust random1st/feynt` and repeat.

The app is signed with a Developer ID and notarised by Apple, and both the app and the DMG
carry their own ticket, so the first launch needs no network round trip and no manual
quarantine dance. macOS 14 or newer.

## Where the speed comes from

Decoding is bound by reading the weights rather than by arithmetic: one token costs a full
sweep over 16 GB. Verifying a block of drafted tokens reads those same weights **once**, so
throughput follows how many drafts survive verification.

The MTP head built into the model drafts one token at a time and lands 2.3–2.6 accepted
tokens per round. [DFlash 2](https://github.com/random1st/dflash-swift) proposes a whole
block in one forward pass and lands 4.1–5.7 on the same model. The output is still the large
model's own: it confirms every token.

For that to pay off, verifying a block has to cost about what verifying a single token
costs. On stock MLX kernels it does not: a quantised matmul re-reads the weights per row,
and a forward on eight rows cost 3.07 forwards on one. A kernel built on `simdgroup_matrix`
reads and dequantises each weight group once for the whole block, and those same eight rows
cost **1.51**. That is what makes the full block worth drafting, and it is worth **1.45x**
end to end on the same hardware and prompt.

The second source of pauses is prefill rather than generation: every turn of a conversation
re-sends the whole history, and the model used to re-read all of it. The state of recent
prompts now stays in memory, and the second turn reuses **1024 of 1042** prompt tokens:
prefill is 28 times faster than cold, and the answer is identical to the character.

## Models

Two of them, one generation, one drafter. A model gets listed only if speculation measurably
speeds it up; every candidate was run against its own plain decode, warm, twice.

| Model | Plain decode | With speculation | Speedup | Accepted per round |
|---|---:|---:|---:|---:|
| Qwen3.8-27B Uncensored | 14.6 | 40 | **2.7x** | 4.10 |
| Qwen3.8-27B | 18.6 | 35 | 1.9x | 3.77 |

Tokens per second, M3 Max. Both models use `incoai/Qwen3.8-27B-DFlash2`, the only
second-generation drafter that exists for a Qwen, the one with the candidate selector.

The 3.5 generation failed the measurement: Qwen3.5-9B gained exactly nothing (57 against
56), and the 3.5 MoE managed 1.1x. The 3.6 generation was dropped deliberately, to support
one generation instead of three; it held the 35B-A3B MoE at 121 tok/s, the fastest thing
measured here.

Weights that are already on disk are found before the app offers to download anything.
Models live in `~/Library/Application Support/Feynt/models`.

## What it does

- First-run wizard: memory and disk check, then a model choice, then a download of whatever
  is missing with progress in bytes actually written.
- Menu bar: state, live tok/s, accepted tokens per round, the port and a copy-URL button,
  load and unload, re-download, chat, log, idle timeout.
- Chat with streaming and a separate collapsible reasoning area, a model switcher and an
  unload button in its top bar.
- Idle unload: the references to the weights are dropped, the MLX cache is cleared and the
  memory goes back to the system. The timeout runs from a minute to an hour, or off.
- Prefix cache: up to four conversations stay warm, capped at 12 GB. The memory is spent on
  purpose, so that the same prefill is never paid for twice.
- Model downloads over ranged requests, 8 MB chunks, eight in flight: 45 MB/s against the
  4 MB/s of a single-stream client. An interrupted download resumes, a rejected chunk is
  retried, and a file appears under its real name only once it is whole.

## Server

It comes up with the model, listens on loopback only and has no third-party HTTP
dependencies. Requests are served strictly one at a time, because the GPU is not shared.

| Method | Path | What it does |
|---|---|---|
| POST | `/v1/chat/completions` | a plain answer, or SSE when `stream: true` |
| GET | `/v1/models` | the model list |
| GET | `/health` | `ok` / `loading` / `no_model` / `error` |
| GET | `/metrics` | requests, tokens, speed, accepted tokens per round |

`messages`, `max_tokens`, `stream`, `temperature` and
`chat_template_kwargs.enable_thinking` are honoured; unknown fields are ignored. Message
content may be a string or a list of typed parts. Reasoning arrives separately, in
`reasoning_content`.

```sh
curl http://127.0.0.1:19234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"hello"}],"max_tokens":256}'
```

The listener stays up when the idle timeout unloads the weights, and a completion request
loads them again on demand.

Speculation runs under greedy decoding; a request with `temperature > 0` is served by the
plain path without it.

## Diagnosing a download

```sh
/Applications/Feynt.app/Contents/MacOS/Feynt --download uncensored
```

This runs the wizard's download inside the real bundle and prints the endpoint, the pinned
commit, the bytes as they arrive and the reason if it stops. Model ids are `uncensored` and
`stock`.

## Building from source

```sh
./package-app.sh release     # build/Feynt.app
./release.sh 0.4.4           # signed DMG, notarised, ticket attached
```

macOS 14+ and Xcode 16+. Dependencies are fetched from the network, including
[DFlashKit](https://github.com/random1st/dflash-swift) and a fork of `mlx-swift-lm` with the
two patches the drafter needs: hidden states tapped from several layers, and rollback of the
recurrent state after a partial accept.

Signing is ad-hoc by default. For a distributable build:

```sh
FEYNT_SIGN_IDENTITY="Developer ID Application: …" ./package-app.sh release
```

Run the tests through `xcodebuild` rather than `swift test`: from a SwiftPM test run
`mlx-swift` does not find its metallib.

## Architecture

`InferenceEngine` is the seam between the interface and whatever turns the weights.
`DFlashEngine` implements it through DFlashKit; `MLXEngine` remains the fallback for the
cases speculation does not cover yet. The weights are loaded once and handed between them,
because nobody needs a second copy of 16 GB.

## Credits

The method and the drafter are not mine. DFlash comes from
[z-lab](https://github.com/z-lab/dflash) ("DFlash: Block Diffusion for Flash Speculative
Decoding", Chen et al., arXiv:2602.06036, MIT); the DFlash 2 components are from
[sglang](https://github.com/sgl-project/sglang) (Apache 2.0); the checkpoints are published
by Inco AI (Apache 2.0); and this port was written against
[mlx-dspark](https://github.com/ARahim3/mlx-dspark) (MIT), whose `small_m_qmm` kernel it
also carries. See the [NOTICE](https://github.com/random1st/dflash-swift/blob/main/NOTICE)
in DFlashKit for the full attribution.
