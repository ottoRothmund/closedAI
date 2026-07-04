# closedAI

Run large open-weights language models by pooling compute across the machines you
and people you trust already own. One model, powered collectively — everyone who
contributes a slice of memory and compute can query it.

The idea is simple: a lot of the best open models are too big to run comfortably
on any single consumer machine, but not too big for a handful of them together.
closedAI lets a group stand up one shared model, split across their hardware, with
an Ollama-grade command line and an OpenAI-compatible API. Model choice is left
entirely to the group.

> **Status: early.** This repository is being built milestone by milestone. The
> first milestone — a working single-node inference core — is complete; the
> distributed layers are next. See the [roadmap](docs/plans/2026-07-03-closedai-roadmap.md).

## What works today

A single-node inference core that the distributed system builds on:

```bash
# build
cargo build --release

# run a local GGUF model, or pull one straight from Hugging Face
closedai run --model ./model.gguf "The capital of France is"
closedai run --model hf.co/bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf "hello"
```

Tokens stream to stdout; a `tokens/sec` line is printed to stderr when generation
finishes.

## How it's built

- **`closedai-engine`** — an `Engine` trait and generation types, deliberately
  shaped so the distributed executor can slot in later without changing callers;
  a `LlamaEngine` implementation backed by llama.cpp; model-reference parsing
  (local paths and `hf.co/<owner>/<repo>/<file.gguf>`); and model download from
  Hugging Face.
- **`closedai-cli`** — the `closedai` binary.

Models are always fetched from Hugging Face at runtime — the project never
redistributes weights.

## Where it's going

The design is a cooperative swarm: a group runs one model defined by a signed
manifest, a few capable always-on nodes form the inference pipeline, and everyone
else contributes redundancy and capacity. Contributing to a model earns the right
to query it. The full design is in
[`docs/specs/2026-07-03-closedai-design.md`](docs/specs/2026-07-03-closedai-design.md).

## Building from source

Requires a recent stable Rust toolchain and CMake (llama.cpp is built from
source through its Rust binding). GPU backends are available behind cargo
features (`cuda`, `metal`, `vulkan`); the default build is CPU-only.

```bash
cargo test --workspace   # offline; model-gated tests skip without a local model
cargo run -p closedai-cli -- run --model ./model.gguf "your prompt"
```

## License

[AGPL-3.0-or-later](LICENSE).
