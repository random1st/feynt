# Feynt

Local language models on Apple Silicon at 2.7x the speed of plain decoding: a small drafter
model pulls a feint, proposing a whole block of tokens at once, and the large one confirms
them in a single pass. Hence the name.

The app lives in the menu bar. Inside it there is a chat, a first-run wizard, an
OpenAI-compatible server, and MCP and A2A endpoints so other agents can use the local model. The engine runs **inside the app** through MLX Swift: no Python,
no child process, nothing to install beforehand.

## Install

```sh
brew install --cask random1st/feynt/feynt
```

If Homebrew asks you to trust the tap first, run `brew trust random1st/feynt` and repeat.

Homebrew reads `random1st/feynt` as the repository `random1st/homebrew-feynt`. That name is
a GitHub redirect to this repository, so the cask lives in one place — `Casks/feynt.rb`
here, updated with every release — and the short command still works.

Installed before 0.8.1, and Homebrew says `Cask 'feynt' is unreadable … syntax errors
found`, or never offers a new version? Your tap is a clone of the old, separate tap
repository. `brew update` tried to rebase that history onto this one, stopped on a conflict
and left the rebase half done, so every later update fails without saying so. Abandon the
rebase and reset the tap once:

```sh
cd "$(brew --repo random1st/feynt)"
git rebase --abort; git fetch origin && git reset --hard origin/main
brew upgrade --cask feynt
```

`brew update-reset` alone does not get out of this state while the rebase is still open.

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

A round also used to wait on itself. Between drafting and verifying, the CPU stopped until
the GPU had finished the draft — a sync that existed only to time the phase for the log, since
nothing on the CPU reads the draft. Queuing it instead lets the CPU build the verify while the
GPU still drafts: **+8%** on an agent-shaped prompt (92.0 → 99.6 tok/s) and **+15%** on short
code (194.2 → 224.1), with the same output token for token.

The second source of pauses is prefill rather than generation: every turn of a conversation
re-sends the whole history, and the model used to re-read all of it. The state of recent
prompts now stays in memory, and the second turn reuses **1024 of 1042** prompt tokens:
prefill is 28 times faster than cold, and the answer is identical to the character.

## Models

Five of them. The three large ones are listed because speculation measurably speeds them
up; every candidate was run against its own plain decode, warm. The two small ones are
listed for memory and for small jobs — see [the small ones](#the-small-ones).

| Model | Plain decode | With speculation | Speedup | Accepted per round |
|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B Uncensored | 106-109 | 225-230 | 2.1x | 9.27 |
| Qwen3.8-27B Uncensored | 14.6 | 40 | 2.7x | 4.10 |
| Qwen3.8-27B | 18.6 | 35 | 1.9x | 3.77 |

Tokens per second, M3 Max. Those are short-prompt numbers, and a drafter only earns them
when it can guess what comes next. Measured over 2.9k tokens of this repository's own
source, the same two MoEs look different:

| Model | Long context, prose | Long context, code |
|---|---:|---:|
| Qwen3.6-35B-A3B Uncensored | 75-76 (2.18) | **51-52 (3.32)** |

So **point a coding agent at `Qwen3.6-35B-A3B Uncensored`**. The two 27Bs share
`incoai/Qwen3.8-27B-DFlash2`.

`Qwen3.6-35B-A3B Uncensored` drafts with `incoai/Qwen3.6-35B-A3B-DFlash2`. Against the
previous `z-lab` drafter it is **5.3% faster** on agent-shaped prompts — code written against
a few thousand tokens of real source — and it won on every one of three such contexts, in
Python, Swift and Markdown, which is the bar a change has to clear here. It is slower on
prose, which is the trade worth knowing before pointing a chat at it.

The drafter runs as published. Quantising it looked like a win and was dropped: four bits
beat bf16 by 6% on one agent prompt and lost to it by 5% on another, which is noise wearing
a verdict. The experiment did settle something else — four bits leave a quarter of the
drafter's bytes, and dropping the other three quarters produced no consistent gain either
way, so a round is not bound by what the drafter reads. What it *is* bound by is not
settled: that needs the round split into drafting, selecting and verifying, and nothing
here measures that yet.

The knob that does move this workload is the draft width, and it is not in the catalog yet.
Pinned at three, the agent workload runs **117.7-118.9 tok/s against 86.6-88.4** at the
default — **+36%** — while short code falls from 221-229 to 154-157. The drafter accepts
about three tokens over a long context and drafts seven, so four verified rows per round
are thrown away; on a short templated prompt it accepts six and the same narrowing throws
the speedup away instead. Nothing in the loop reads the context length, which is what would
let both workloads have it.

Dropped on the same rule: Ornith-1.5-35B-A3B leads on a short templated prompt (257-261
against 225-230) and loses where an agent actually works — 34-40 against 51-52 on a long
context with code, and 70-73 against 75-76 on prose. It was listed for its tool discipline,
which stopped being a reason once the server learned the dialect the 35B-A3B speaks; an
entry nobody should download is 19.5 GB of temptation. Qwen3-Coder-Next decodes at 59-61
tok/s and costs 42 GB with no DFlash 2 drafter in existence; LFM2.5-8B-A1B is fast (204-208) but loops on an empty tool
result, which is where an agent actually lives; the 3.5 generation gained 1.0-1.4x on
DFlash 1 drafters that have no candidate selector.

### The small ones

**Qwen3.5-2B** is the one to hand small agent jobs to: summarise, extract, classify, look
something up with the read-only tools. 1.7 GB, and the fastest model here — 241 tok/s on
a short prompt, 123 on a 36k-token log, 77 at 100k, where the 35B-A3B manages 48-53 on an
8k-token summary. On an agent-shaped check — a value found with `read_file`, a file found
with `grep`, JSON pulled out of a message, five lines classified — it got all four right,
as did Qwen3.5-4B and the MoE; LFM2.5-8B-A1B returned nothing on the last two. Capped at
120,000 tokens, which keeps it under 8 GB (6.7 GB at 100k).

**Qwen3.5-4B** is for summaries, log digging and long documents on a machine that has to
keep its memory for something else: the whole process, weights and context, stays under
8 GB. 3 GB of weights, vision in the checkpoint, and three layers of four on linear
attention, so a long log costs about 75 MB per thousand tokens.

| Context | Footprint | Decode | Reading the prompt |
|---:|---:|---:|---:|
| short | 4.2 GB | 108-121 tok/s | — |
| 36k | 6.3 GB | 72 | 1060 |
| 55k | 7.6 GB | 57 | 895 |

A request is capped at 56,000 tokens, prompt and reply together; past that it is refused
with the numbers rather than allowed to spend the budget. It is not the fast option: on a
36k-token log the 35B-A3B MoE decodes 56-58 tok/s, and Qwen3.5-9B, measured for this slot,
only 31-47 — a dense model reads all its weights for every token, the MoE about 3B of 35B.
It runs without a drafter: the 9B's accepted 1.6-1.8 tokens a round on summaries and slowed
them down.

Weights that are already on disk are found before the app offers to download anything.
Models live in `~/Library/Application Support/Feynt/models`.

## What it does

- First-run wizard: memory and disk check, then a model choice, then a download of whatever
  is missing with progress in bytes actually written.
- Menu bar: state, live tok/s, accepted tokens per round, the port and a copy-URL button,
  load and unload, re-download, chat, log, idle timeout.
- Chat with streaming and a separate collapsible reasoning area, a model switcher and an
  unload button in its top bar. Images attach with the paperclip or by dropping them on the
  message field.
- Model tools, read-only on purpose: `read_file`, `list_files` and `grep` inside a folder you
  pick, and `web_fetch` for public pages. Nothing writes or runs — a local model is not
  trusted with that, and in agent mode nobody is there to approve it. The folder is the
  boundary (symlinks are resolved before the check), credential locations such as `~/.ssh`,
  `~/.aws` and `.env` files are refused in any folder, and `web_fetch` refuses loopback,
  private and link-local addresses, on every redirect too. At most six calls per answer.
  The chat has a **Tools** toggle and a folder menu; each call shows as a row you can open.
- Idle unload: the references to the weights are dropped, the MLX cache is cleared and the
  memory goes back to the system. The timeout runs from a minute to an hour, or off.
- Memory: MLX's cache of freed buffers is capped at 1 GB. Uncapped, one 36k-token prompt
  left the process at 93 GB, 89 of it cache; capped, 6.3 GB, at the same speed.
- Prefix cache: up to four conversations stay warm, capped at 12 GB. The memory is spent on
  purpose, so that the same prefill is never paid for twice.
- Update checks: shortly after launch and once a day Feynt asks GitHub for the latest
  release, and if it is newer says so — a system notification once per version, and an
  "Update available" item at the top of the menu. It does not replace itself: Feynt is a
  Homebrew cask, and an app that swapped its own bundle would leave Homebrew believing the
  old version is installed. Update with `brew upgrade --cask feynt`. The one request carries
  nothing but the app's version in its User-Agent; "Check for updates automatically" in the
  menu turns it off, and "Check for Updates…" asks on demand.
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

`messages`, `tools`, `max_tokens`, `stream`, `temperature` and
`chat_template_kwargs.enable_thinking` are honoured; unknown fields are ignored. Message
content may be a string or a list of typed parts, `image_url` parts included (see
[Vision](#vision)). Reasoning arrives separately, in `reasoning_content`.

Tool calls are OpenAI-shaped in both directions: `tools` go in, `tool_calls` come back on
the assistant message, and a `tool` message carries the result of one. Each model speaks
its own dialect - `xml_function` for the coding model, framed JSON for the 27Bs - and the
dialect is resolved from the checkpoint after loading, so a client never sees protocol text
in the answer.

```sh
curl http://127.0.0.1:19234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"hello"}],"max_tokens":256}'
```

The listener stays up when the idle timeout unloads the weights, and a completion request
loads them again on demand.

Speculation runs under greedy decoding; a request with `temperature > 0` is served by the
plain path without it. A request carrying `tools` used to take that plain path too, which
cost it the prefix cache — the expensive half of an agent's turn, since every step re-sends
the whole conversation. The speculative loop now decodes tool calls itself, so the same
3187-token conversation re-sent costs 1.00s instead of 2.96s. The trade is that a block of
16 drafted tokens does not pay for itself at the ~1.6 acceptance a tool-carrying prose
request sees: sustained decode drops from 64-68 to 47-49 tok/s, so the change wins up to
roughly 500 generated tokens per turn and draws level past that. An agent's turns are far
shorter than that.

## Vision

Every model in the catalog reads images: in the chat (the paperclip, or drop a picture on
the message field), over the OpenAI endpoint as `image_url` parts, through MCP `generate`
as `images: [{data, mimeType}]`, and over A2A as a `raw` part with an image `mediaType`.
Images go inline, base64. A URL to an image is refused rather than fetched, because a
server that fetches what a client names can be pointed at the local network.

The checkpoints carry a vision tower that a text-only load skips. Feynt loads it next to
the resident text model and points the vision model's language-model weights at the text
model's own arrays, so vision costs the tower alone — 0.89 GB on the 35B-A3B, 0.92 GB on
the 27Bs, 0.67 GB on the 4B — read in about 0.2 s. The text model stays the one the DFlash
drafter taps, so text keeps its speed; a request with an image decodes without
speculation, about 90 tok/s on the 35B-A3B and 22 on a 27B.

## MCP and A2A

The same listener on `127.0.0.1:19234` speaks two agent protocols, so another agent can
use the local model without being configured as an OpenAI client.

**MCP** — `POST /mcp`, revision 2026-07-28, with the `initialize` handshake of earlier
revisions answered too, since most hosts in use still open with it. Four tools:
`list_models` (what is downloaded, loaded and active), `load_model`, `unload_model`, and
`generate` (a prompt, an optional system prompt and model, an answer). `generate` lets the
model use the tools above: `workspace` names the folder it may read, `tools: false` turns
them off; the calls it made come back in `structuredContent.toolCalls`.

What `generate` takes from an agent, beyond the prompt:

- `files` — paths Feynt reads itself and puts in front of the prompt (text), or sends to
  the vision tower (images, recognised by their bytes). The contents never pass through
  the calling agent's context, and a small model gets the material up front instead of
  having to decide to look for it. Absolute paths, or relative to `workspace`; credential
  locations are refused as everywhere else.
- `json_schema` — a JSON Schema the answer must follow, enforced token by token with the
  grammar mask from [mac-mlx](https://github.com/magicnight/mac-mlx) (Apache 2.0, vendored in
  `Sources/Feynt/Vendor/MacMLXConstraint`). The parsed value comes back in
  `structuredContent.json`. Such a request decodes without speculation.
- A model that cannot answer from what it was given replies `INSUFFICIENT: …`, and the
  result says `insufficient: true`, so a missing fact is not dressed up as an answer.

Every result reports `usage` — prompt and generated tokens, seconds, time to first token,
tok/s — in `structuredContent`, and as a one-line second content block, so the answer
itself stays exactly what the model wrote. A client that sends a `progressToken` gets the
result as an SSE stream with progress notifications: about once a second while tokens
arrive, and every ten seconds while a long prompt is being read, so a slow local answer
does not trip the caller's timeout. `list_models` says what each model is for, its size,
context, and whether it reads images or speculates. Downloading is
deliberately not a tool: a model is 16–20 GB, which is not something an agent should start
without being asked.

```sh
claude mcp add --transport http feynt http://127.0.0.1:19234/mcp
```

**A2A** — agent card at `/.well-known/agent-card.json`, JSON-RPC at `POST /a2a`, protocol
1.0: `SendMessage`, `SendStreamingMessage`, `GetTask`, `CancelTask`. Messages that share a
`contextId` continue one conversation. `metadata.model` picks the model (`uncensored-moe`,
`uncensored`, `stock`); without it the active one answers. `metadata.workspace` and
`metadata.tools` do what they do for MCP `generate`. `CancelTask` stops the
generation itself — the next request is answered as on an idle machine — rather than only
marking the task.

Both are checked against their official clients: MCP with the Python SDK 2.3.0 over both
the current and the handshake path, A2A with `a2a-sdk` 1.2.2 with and without streaming.
Both refuse a request whose `Origin` is a website rather than this machine, which is what
keeps a page in a browser from driving them through DNS rebinding. Generation on either
waits its turn behind the OpenAI endpoint: there is one GPU.

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

Tests come in two kinds. `FEYNT_TESTS=1 swift test` checks, without a model and in under a second, what
decides what a model may touch: the folder boundary, credential locations, private
addresses, globs and the tool-call budget. None of it touches MLX, which from a SwiftPM test
run would not find its metallib. The variable keeps the test target out of a
release build, which Xcode otherwise compiles differently, 13 MB larger.

`Tests/e2e` drives a running Feynt over HTTP: both protocols, the official MCP and A2A
clients, the model's tools, and that a cancelled or abandoned request frees the GPU. It loads
and unloads models, so point it at a copy rather than the Feynt you use:

```sh
FEYNT_URL=http://127.0.0.1:19235 FEYNT_MODEL=uncensored-moe \
    uv run --with pytest --with mcp --with a2a-sdk pytest Tests/e2e
```

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
