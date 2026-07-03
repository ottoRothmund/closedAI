# closedAI Milestone 01 — Local Inference Core: Implementation Plan

> Implement this plan one task at a time. Each task's steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust workspace that runs a GGUF language model on one machine through a clean `Engine` abstraction, driven by a `closedai run <model>` CLI, with whole-file model download from Hugging Face.

**Architecture:** A Cargo workspace with two crates. `closedai-engine` wraps llama.cpp (via the `llama-cpp-2` binding) behind an `Engine` trait so that later milestones can substitute a distributed, layer-split executor without changing any caller. `closedai-cli` is the user-facing binary that resolves a model reference, downloads it if needed, and streams generated tokens. This milestone is deliberately single-node and single-process: it de-risks the engine binding and toolchain before any distribution logic exists.

**Tech Stack:** Rust (edition 2021), `llama-cpp-2` (llama.cpp binding), `hf-hub` (Hugging Face downloads), `clap` (CLI), `thiserror`/`anyhow` (errors), `encoding_rs` (token piece decoding), `tracing` (logging).

## Global Constraints

Every task's requirements implicitly include these. Exact values:

- **License:** AGPL-3.0-or-later. Every source file starts with `// SPDX-License-Identifier: AGPL-3.0-or-later`. A verbatim AGPL-3.0 `LICENSE` file lives at the repo root.
- **Consistent style** — one formatter (rustfmt), one naming convention, and uniform doc comments across the codebase.
- **Language:** Rust, edition 2021, stable toolchain. Formatting with `rustfmt`; linting with `clippy` at `-D warnings`.
- **Engine boundary:** All inference goes through the `Engine` trait. No caller may reference `llama_cpp_2` types directly outside `closedai-engine`.
- **llama.cpp pin:** `llama-cpp-2` is pinned to an exact version whose bundled llama.cpp is at or after build **b8492** (the CVE-2026-34159 fix) and new enough to load the flagship `deepseek2`/MLA architecture (GLM-4.7-Flash). Verified in Task 7. Never use llama.cpp's `rpc-server` or its RPC wire protocol.
- **No weight redistribution:** Models are fetched from Hugging Face at runtime by the tool; no model weights are committed to the repository or served by any closedAI-operated infrastructure.
- **GPU backends** are opt-in cargo features in this milestone (`cuda`, `metal`, `vulkan`); the default build is CPU-only. Single-binary auto-detection is deferred to Milestone 09.
- **Commits:** Conventional Commits; commit after each task's tests pass.

---

## File Structure

```
closedAI/
├── Cargo.toml                     # workspace manifest
├── LICENSE                        # AGPL-3.0 verbatim
├── rust-toolchain.toml            # pin stable channel
├── .github/workflows/ci.yml       # fmt + clippy + test
├── crates/
│   ├── closedai-engine/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs             # re-exports; crate docs
│   │   │   ├── engine.rs          # Engine trait, GenerationParams, GeneratedToken, outcomes
│   │   │   ├── error.rs           # EngineError
│   │   │   ├── model_ref.rs       # ModelRef parsing (local path vs hf.co ref)
│   │   │   └── llama.rs           # LlamaEngine: llama-cpp-2 implementation of Engine
│   │   └── tests/
│   │       └── generation.rs      # integration test with a tiny real GGUF
│   └── closedai-cli/
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs            # clap entrypoint, subcommand dispatch
│           └── run.rs             # `run` subcommand: resolve → download → stream
└── docs/...                        # spec, roadmap, this plan
```

**Responsibilities:**
- `engine.rs` — the public contract. Pure Rust; no llama.cpp types leak here.
- `error.rs` — one error enum for the crate.
- `model_ref.rs` — pure parsing/resolution, no I/O.
- `llama.rs` — the only file that touches `llama_cpp_2`.
- `closedai-cli` — argument parsing, download orchestration, and terminal streaming only; all inference is delegated to `closedai-engine`.

---

## Task 1: Workspace, tooling, license, CI

**Files:**
- Create: `Cargo.toml`, `LICENSE`, `rust-toolchain.toml`, `.gitignore`, `.github/workflows/ci.yml`
- Create: `crates/closedai-engine/Cargo.toml`, `crates/closedai-engine/src/lib.rs`
- Create: `crates/closedai-cli/Cargo.toml`, `crates/closedai-cli/src/main.rs`

**Interfaces:**
- Produces: a buildable workspace; `closedai_engine::hello_ok() -> bool` (a trivial function that exists only to anchor the first test and is deleted in Task 2).

- [ ] **Step 1: Create the workspace manifest**

`Cargo.toml`:
```toml
[workspace]
resolver = "2"
members = ["crates/closedai-engine", "crates/closedai-cli"]

[workspace.package]
version = "0.0.1"
edition = "2021"
license = "AGPL-3.0-or-later"
repository = ""

[workspace.dependencies]
thiserror = "1"
anyhow = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
clap = { version = "4", features = ["derive"] }
hf-hub = "0.3"
encoding_rs = "0.8"
# Pin exactly; Task 7 verifies the bundled llama.cpp is >= b8492 and loads deepseek2.
llama-cpp-2 = "=0.1.150"
```

- [ ] **Step 2: Add the toolchain pin, gitignore, and license**

`rust-toolchain.toml`:
```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
```

`.gitignore`:
```
/target
*.gguf
.closedai-cache/
```

Download the AGPL-3.0 text to `LICENSE`:
```bash
curl -fsSL https://www.gnu.org/licenses/agpl-3.0.txt -o LICENSE
```

- [ ] **Step 3: Create the engine crate with an anchor function**

`crates/closedai-engine/Cargo.toml`:
```toml
[package]
name = "closedai-engine"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
thiserror.workspace = true
tracing.workspace = true
llama-cpp-2.workspace = true
hf-hub.workspace = true
encoding_rs.workspace = true

[features]
default = []
cuda = ["llama-cpp-2/cuda"]
metal = ["llama-cpp-2/metal"]
vulkan = ["llama-cpp-2/vulkan"]
```

`crates/closedai-engine/src/lib.rs`:
```rust
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Inference engine for closedAI. All inference flows through the [`Engine`]
//! trait so later milestones can substitute a distributed executor.

/// Temporary anchor so the workspace has a test before real types exist.
/// Removed in Task 2.
pub fn hello_ok() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_builds_and_tests_run() {
        assert!(hello_ok());
    }
}
```

- [ ] **Step 4: Create the CLI crate stub**

`crates/closedai-cli/Cargo.toml`:
```toml
[package]
name = "closedai-cli"
version.workspace = true
edition.workspace = true
license.workspace = true

[[bin]]
name = "closedai"
path = "src/main.rs"

[dependencies]
closedai-engine = { path = "../closedai-engine" }
anyhow.workspace = true
clap.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
```

`crates/closedai-cli/src/main.rs`:
```rust
// SPDX-License-Identifier: AGPL-3.0-or-later

fn main() {
    println!("closedai {}", env!("CARGO_PKG_VERSION"));
}
```

- [ ] **Step 5: Add CI**

`.github/workflows/ci.yml`:
```yaml
name: ci
on: [push, pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - name: Install llama.cpp build deps
        run: sudo apt-get update && sudo apt-get install -y cmake libclang-dev
      - run: cargo fmt --all -- --check
      - run: cargo clippy --workspace --all-targets -- -D warnings
      - run: cargo test --workspace
```

- [ ] **Step 6: Verify the workspace builds and tests pass**

Run: `cargo test --workspace`
Expected: compiles; `workspace_builds_and_tests_run` passes. (First build compiles llama.cpp and is slow; this is normal.)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: scaffold Rust workspace, license, and CI"
```

---

## Task 2: Engine trait and generation types

**Files:**
- Create: `crates/closedai-engine/src/engine.rs`
- Create: `crates/closedai-engine/src/error.rs`
- Modify: `crates/closedai-engine/src/lib.rs` (remove `hello_ok`, add modules)

**Interfaces:**
- Produces:
  - `trait Engine { fn model_info(&self) -> &ModelInfo; fn generate(&mut self, prompt: &str, params: &GenerationParams, on_token: &mut dyn FnMut(GeneratedToken) -> std::ops::ControlFlow<()>) -> Result<GenerationOutcome, EngineError>; }`
  - `struct GenerationParams { pub max_tokens: usize, pub temperature: f32, pub top_p: f32, pub seed: u32, pub stop: Vec<String> }` with `Default`
  - `struct GeneratedToken { pub token_id: i32, pub text: String }`
  - `struct GenerationOutcome { pub tokens_generated: usize, pub stop_reason: StopReason, pub prompt_tokens: usize }`
  - `enum StopReason { Eos, MaxTokens, StopString, Cancelled }`
  - `struct ModelInfo { pub name: String, pub n_ctx_train: u32 }`
  - `enum EngineError` (in `error.rs`)

- [ ] **Step 1: Write the failing test for `GenerationParams` defaults**

Append to `crates/closedai-engine/src/engine.rs` (create the file with this test first):
```rust
// SPDX-License-Identifier: AGPL-3.0-or-later

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_params_have_sane_defaults() {
        let p = GenerationParams::default();
        assert_eq!(p.max_tokens, 512);
        assert!((p.temperature - 0.8).abs() < f32::EPSILON);
        assert!((p.top_p - 0.95).abs() < f32::EPSILON);
        assert!(p.stop.is_empty());
    }

    #[test]
    fn stop_reason_is_equatable() {
        assert_eq!(StopReason::Eos, StopReason::Eos);
        assert_ne!(StopReason::Eos, StopReason::MaxTokens);
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p closedai-engine generation_params_have_sane_defaults`
Expected: FAIL — `GenerationParams` not found.

- [ ] **Step 3: Write the types**

Prepend to `crates/closedai-engine/src/engine.rs` (above the `tests` module):
```rust
use std::ops::ControlFlow;

use crate::error::EngineError;

/// Sampling and length controls for one generation call.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u32,
    pub stop: Vec<String>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            seed: 0,
            stop: Vec::new(),
        }
    }
}

/// One decoded token handed to the streaming callback.
#[derive(Debug, Clone)]
pub struct GeneratedToken {
    pub token_id: i32,
    pub text: String,
}

/// Why a generation loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    MaxTokens,
    StopString,
    Cancelled,
}

/// Result of a completed (or cancelled) generation call.
#[derive(Debug, Clone)]
pub struct GenerationOutcome {
    pub prompt_tokens: usize,
    pub tokens_generated: usize,
    pub stop_reason: StopReason,
}

/// Static facts about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub n_ctx_train: u32,
}

/// The one boundary all inference crosses. Implementations may be single-node
/// (Milestone 01) or distributed (later milestones); callers never know which.
pub trait Engine: Send {
    fn model_info(&self) -> &ModelInfo;

    /// Generate from `prompt`, invoking `on_token` for each decoded token.
    /// Returning `ControlFlow::Break` from the callback stops generation with
    /// `StopReason::Cancelled`.
    fn generate(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        on_token: &mut dyn FnMut(GeneratedToken) -> ControlFlow<()>,
    ) -> Result<GenerationOutcome, EngineError>;
}
```

- [ ] **Step 4: Write the error type**

`crates/closedai-engine/src/error.rs`:
```rust
// SPDX-License-Identifier: AGPL-3.0-or-later

use thiserror::Error;

/// Errors from loading a model or generating text.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("failed to load model: {0}")]
    Load(String),

    #[error("failed to tokenize prompt: {0}")]
    Tokenize(String),

    #[error("decode step failed: {0}")]
    Decode(String),

    #[error("prompt is {prompt} tokens but the context window is {n_ctx}")]
    ContextOverflow { prompt: usize, n_ctx: usize },
}
```

- [ ] **Step 5: Wire modules and remove the anchor**

Replace `crates/closedai-engine/src/lib.rs` body (keep the SPDX line and crate doc):
```rust
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Inference engine for closedAI. All inference flows through the [`Engine`]
//! trait so later milestones can substitute a distributed executor.

pub mod engine;
pub mod error;

pub use engine::{
    Engine, GeneratedToken, GenerationOutcome, GenerationParams, ModelInfo, StopReason,
};
pub use error::EngineError;
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p closedai-engine`
Expected: PASS — both `engine::tests` cases green.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(engine): define Engine trait and generation types"
```

---

## Task 3: Model reference parsing

**Files:**
- Create: `crates/closedai-engine/src/model_ref.rs`
- Modify: `crates/closedai-engine/src/lib.rs` (add `pub mod model_ref;` and re-export)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `enum ModelRef { LocalPath(PathBuf), HuggingFace { repo: String, file: String } }`
  - `impl ModelRef { pub fn parse(input: &str) -> Result<ModelRef, ModelRefError>; }`
  - `enum ModelRefError` (variant `Malformed(String)`)

A Hugging Face reference has the form `hf.co/<owner>/<repo>/<file.gguf>` (the file path after the repo is the GGUF filename). Anything else is treated as a local filesystem path.

- [ ] **Step 1: Write the failing tests**

`crates/closedai-engine/src/model_ref.rs`:
```rust
// SPDX-License-Identifier: AGPL-3.0-or-later

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn parses_hugging_face_reference() {
        let r = ModelRef::parse("hf.co/bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf")
            .unwrap();
        assert_eq!(
            r,
            ModelRef::HuggingFace {
                repo: "bartowski/SmolLM2-135M-Instruct-GGUF".to_string(),
                file: "SmolLM2-135M-Instruct-Q4_K_M.gguf".to_string(),
            }
        );
    }

    #[test]
    fn parses_local_path() {
        let r = ModelRef::parse("./models/tiny.gguf").unwrap();
        assert_eq!(r, ModelRef::LocalPath(PathBuf::from("./models/tiny.gguf")));
    }

    #[test]
    fn rejects_hf_reference_without_file() {
        let err = ModelRef::parse("hf.co/owner/repo");
        assert!(matches!(err, Err(ModelRefError::Malformed(_))));
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p closedai-engine model_ref`
Expected: FAIL — `ModelRef` not found.

- [ ] **Step 3: Write the implementation**

Prepend to `crates/closedai-engine/src/model_ref.rs`:
```rust
use std::path::PathBuf;

use thiserror::Error;

/// Where a model's weights come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRef {
    LocalPath(PathBuf),
    HuggingFace { repo: String, file: String },
}

#[derive(Debug, Error)]
pub enum ModelRefError {
    #[error("malformed model reference: {0}")]
    Malformed(String),
}

impl ModelRef {
    /// Parse a user-supplied model reference. `hf.co/<owner>/<repo>/<file.gguf>`
    /// is a Hugging Face reference; everything else is a local path.
    pub fn parse(input: &str) -> Result<ModelRef, ModelRefError> {
        let trimmed = input.trim();
        if let Some(rest) = trimmed
            .strip_prefix("hf.co/")
            .or_else(|| trimmed.strip_prefix("https://huggingface.co/"))
        {
            let parts: Vec<&str> = rest.split('/').filter(|s| !s.is_empty()).collect();
            if parts.len() < 3 {
                return Err(ModelRefError::Malformed(format!(
                    "expected hf.co/<owner>/<repo>/<file.gguf>, got '{input}'"
                )));
            }
            let file = parts[parts.len() - 1].to_string();
            let repo = parts[..parts.len() - 1].join("/");
            if !file.ends_with(".gguf") {
                return Err(ModelRefError::Malformed(format!(
                    "expected a .gguf file at the end of '{input}'"
                )));
            }
            return Ok(ModelRef::HuggingFace { repo, file });
        }
        Ok(ModelRef::LocalPath(PathBuf::from(trimmed)))
    }
}
```

- [ ] **Step 4: Export the module**

Add to `crates/closedai-engine/src/lib.rs` after `pub mod error;`:
```rust
pub mod model_ref;
pub use model_ref::{ModelRef, ModelRefError};
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p closedai-engine model_ref`
Expected: PASS — all three cases green.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(engine): parse local and Hugging Face model references"
```

---

## Task 4: LlamaEngine — load and generate via llama.cpp

**Files:**
- Create: `crates/closedai-engine/src/llama.rs`
- Create: `crates/closedai-engine/tests/generation.rs`
- Modify: `crates/closedai-engine/src/lib.rs` (add `pub mod llama;` and re-export `LlamaEngine`)

**Interfaces:**
- Consumes: `Engine`, `GenerationParams`, `GeneratedToken`, `GenerationOutcome`, `StopReason`, `ModelInfo`, `EngineError` (Task 2).
- Produces:
  - `struct LlamaEngine { /* private */ }`
  - `impl LlamaEngine { pub fn load(model_path: &std::path::Path, n_ctx: u32, n_gpu_layers: u32) -> Result<LlamaEngine, EngineError>; }`
  - `impl Engine for LlamaEngine`

**Note on the binding:** The code below follows the `llama-cpp-2` API (`LlamaBackend`, `LlamaModel::load_from_file`, `LlamaContextParams`, `LlamaBatch`, `LlamaSampler`, `token_to_piece`). Import paths and one or two signatures can drift between crate versions — build against the pinned `=0.1.150` and reconcile any mismatch against its docs.rs page; do not upgrade the pin to resolve a compile error without redoing Task 7's verification.

- [ ] **Step 1: Write the failing integration test**

`crates/closedai-engine/tests/generation.rs`:
```rust
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Integration test against a tiny real GGUF. Set CLOSEDAI_TEST_MODEL to the
//! path of a small .gguf (see the plan's Task 4 for how to fetch one). The test
//! is skipped when the env var is unset so `cargo test` stays offline by default.

use std::ops::ControlFlow;
use std::path::PathBuf;

use closedai_engine::{Engine, GenerationParams, LlamaEngine, StopReason};

fn test_model() -> Option<PathBuf> {
    std::env::var_os("CLOSEDAI_TEST_MODEL").map(PathBuf::from)
}

#[test]
fn generates_tokens_from_a_prompt() {
    let Some(model) = test_model() else {
        eprintln!("skipping: set CLOSEDAI_TEST_MODEL to a small .gguf to run");
        return;
    };

    let mut engine = LlamaEngine::load(&model, 2048, 0).expect("load model");

    let params = GenerationParams {
        max_tokens: 16,
        temperature: 0.0, // greedy → deterministic
        ..GenerationParams::default()
    };

    let mut collected = String::new();
    let mut count = 0usize;
    let outcome = engine
        .generate("The capital of France is", &params, &mut |tok| {
            collected.push_str(&tok.text);
            count += 1;
            ControlFlow::Continue(())
        })
        .expect("generate");

    assert!(count > 0, "expected at least one token");
    assert_eq!(outcome.tokens_generated, count);
    assert!(matches!(
        outcome.stop_reason,
        StopReason::Eos | StopReason::MaxTokens
    ));
    assert!(!collected.trim().is_empty(), "expected non-empty output");
}
```

- [ ] **Step 2: Fetch a tiny test model and run the test to verify it fails**

```bash
mkdir -p .closedai-cache
curl -fsSL -o .closedai-cache/smol.gguf \
  https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf
CLOSEDAI_TEST_MODEL=.closedai-cache/smol.gguf cargo test -p closedai-engine --test generation
```
Expected: FAIL — `LlamaEngine` not found.

- [ ] **Step 3: Implement `LlamaEngine`**

`crates/closedai-engine/src/llama.rs`:
```rust
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::ops::ControlFlow;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use crate::engine::{
    Engine, GeneratedToken, GenerationOutcome, GenerationParams, ModelInfo, StopReason,
};
use crate::error::EngineError;

/// Process-global llama.cpp backend. `LlamaBackend::init()` may be called only
/// once per process, so it is shared across every `LlamaEngine` and every thread
/// (notably the parallel integration tests).
static BACKEND: OnceLock<Arc<LlamaBackend>> = OnceLock::new();

fn backend() -> Arc<LlamaBackend> {
    BACKEND
        .get_or_init(|| Arc::new(LlamaBackend::init().expect("llama.cpp backend init")))
        .clone()
}

/// Single-node llama.cpp-backed engine.
pub struct LlamaEngine {
    backend: Arc<LlamaBackend>,
    model: LlamaModel,
    n_ctx: u32,
    info: ModelInfo,
}

impl LlamaEngine {
    /// Load a GGUF model. `n_gpu_layers` = 0 forces CPU; a large value offloads
    /// all layers when a GPU backend feature is compiled in.
    pub fn load(
        model_path: &Path,
        n_ctx: u32,
        n_gpu_layers: u32,
    ) -> Result<LlamaEngine, EngineError> {
        let backend = backend();

        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| EngineError::Load(e.to_string()))?;

        let info = ModelInfo {
            name: model_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string(),
            n_ctx_train: model.n_ctx_train() as u32,
        };

        Ok(LlamaEngine {
            backend,
            model,
            n_ctx,
            info,
        })
    }

    fn build_sampler(params: &GenerationParams) -> LlamaSampler {
        if params.temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::top_p(params.top_p),
                LlamaSampler::temp(params.temperature),
                LlamaSampler::dist(params.seed),
            ])
        }
    }
}

impl Engine for LlamaEngine {
    fn model_info(&self) -> &ModelInfo {
        &self.info
    }

    fn generate(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        on_token: &mut dyn FnMut(GeneratedToken) -> ControlFlow<()>,
    ) -> Result<GenerationOutcome, EngineError> {
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(self.n_ctx));
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| EngineError::Load(e.to_string()))?;

        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| EngineError::Tokenize(e.to_string()))?;
        let prompt_tokens = tokens.len();
        if prompt_tokens >= self.n_ctx as usize {
            return Err(EngineError::ContextOverflow {
                prompt: prompt_tokens,
                n_ctx: self.n_ctx as usize,
            });
        }

        // Feed the prompt; request logits only for the final token.
        let mut batch = LlamaBatch::new(512, 1);
        let last = prompt_tokens - 1;
        for (i, tok) in tokens.iter().enumerate() {
            batch
                .add(*tok, i as i32, &[0], i == last)
                .map_err(|e| EngineError::Decode(e.to_string()))?;
        }
        ctx.decode(&mut batch).map_err(|e| EngineError::Decode(e.to_string()))?;

        let mut sampler = Self::build_sampler(params);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut n_cur = prompt_tokens as i32;
        let mut generated = 0usize;
        let mut acc = String::new();
        let mut stop_reason = StopReason::MaxTokens;

        while generated < params.max_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                stop_reason = StopReason::Eos;
                break;
            }

            let piece = self
                .model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(|e| EngineError::Decode(e.to_string()))?;

            generated += 1;
            if let ControlFlow::Break(()) = on_token(GeneratedToken {
                token_id: token.0,
                text: piece.clone(),
            }) {
                stop_reason = StopReason::Cancelled;
                break;
            }

            // Stop-string check on the accumulated tail.
            acc.push_str(&piece);
            if !params.stop.is_empty() && params.stop.iter().any(|s| acc.ends_with(s)) {
                stop_reason = StopReason::StopString;
                break;
            }

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| EngineError::Decode(e.to_string()))?;
            n_cur += 1;
            ctx.decode(&mut batch).map_err(|e| EngineError::Decode(e.to_string()))?;
        }

        Ok(GenerationOutcome {
            prompt_tokens,
            tokens_generated: generated,
            stop_reason,
        })
    }
}
```

- [ ] **Step 4: Export `LlamaEngine`**

Add to `crates/closedai-engine/src/lib.rs`:
```rust
pub mod llama;
pub use llama::LlamaEngine;
```

- [ ] **Step 5: Run the integration test to verify it passes**

Run: `CLOSEDAI_TEST_MODEL=.closedai-cache/smol.gguf cargo test -p closedai-engine --test generation`
Expected: PASS — `generates_tokens_from_a_prompt` produces non-empty output.

If a method name mismatches the pinned crate, open the `llama-cpp-2` `=0.1.150` docs on docs.rs, find the equivalent, and adjust — keep the surrounding structure. Most likely reconciliations: `is_eog_token` may be absent — fall back to `token == self.model.token_eos()` (that only catches the single EOS token, so multi-stop chat models like the flagship may generate to `max_tokens`, which is acceptable this milestone); `token_to_piece` may be named `token_to_str`; `with_n_ctx` expects `Option<NonZeroU32>`; `n_ctx_train` may return `u32` already (drop the cast).

- [ ] **Step 6: Run clippy and fmt**

Run: `cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(engine): llama.cpp-backed LlamaEngine with streaming generation"
```

---

## Task 5: Whole-file Hugging Face download

**Files:**
- Create: `crates/closedai-engine/src/download.rs`
- Modify: `crates/closedai-engine/src/lib.rs` (add module + re-export)

**Interfaces:**
- Consumes: `ModelRef` (Task 3), `EngineError` (Task 2).
- Produces:
  - `fn resolve_model(model_ref: &ModelRef) -> Result<PathBuf, EngineError>` — returns a local path, downloading the whole GGUF from Hugging Face if the ref is remote (partial/range download is Milestone 03).
  - Adds `EngineError::Download(String)` variant.

- [ ] **Step 1: Write the failing test**

Append to `crates/closedai-engine/src/download.rs`:
```rust
// SPDX-License-Identifier: AGPL-3.0-or-later

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_ref::ModelRef;
    use std::path::PathBuf;

    #[test]
    fn local_ref_resolves_to_its_own_path() {
        let r = ModelRef::LocalPath(PathBuf::from("/tmp/x.gguf"));
        assert_eq!(resolve_model(&r).unwrap(), PathBuf::from("/tmp/x.gguf"));
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p closedai-engine download`
Expected: FAIL — `resolve_model` not found.

- [ ] **Step 3: Implement resolution and download**

Prepend to `crates/closedai-engine/src/download.rs`:
```rust
use std::path::PathBuf;

use hf_hub::api::sync::Api;

use crate::error::EngineError;
use crate::model_ref::ModelRef;

/// Return a local path to the model's GGUF, downloading the whole file from
/// Hugging Face if needed. Downloads are cached by `hf-hub` under its default
/// cache directory; a local reference is returned unchanged.
pub fn resolve_model(model_ref: &ModelRef) -> Result<PathBuf, EngineError> {
    match model_ref {
        ModelRef::LocalPath(p) => Ok(p.clone()),
        ModelRef::HuggingFace { repo, file } => {
            let api = Api::new().map_err(|e| EngineError::Download(e.to_string()))?;
            let path = api
                .model(repo.clone())
                .get(file)
                .map_err(|e| EngineError::Download(e.to_string()))?;
            Ok(path)
        }
    }
}
```

- [ ] **Step 4: Add the `Download` error variant**

In `crates/closedai-engine/src/error.rs`, add to `EngineError`:
```rust
    #[error("failed to download model: {0}")]
    Download(String),
```

- [ ] **Step 5: Export the module**

Add to `crates/closedai-engine/src/lib.rs`:
```rust
pub mod download;
pub use download::resolve_model;
```

- [ ] **Step 6: Run the unit test to verify it passes**

Run: `cargo test -p closedai-engine download`
Expected: PASS — `local_ref_resolves_to_its_own_path`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(engine): resolve model refs, downloading whole GGUF from Hugging Face"
```

---

## Task 6: CLI `run` subcommand with streaming output

**Files:**
- Create: `crates/closedai-cli/src/run.rs`
- Modify: `crates/closedai-cli/src/main.rs` (clap parser, subcommand dispatch, logging init)

**Interfaces:**
- Consumes: `resolve_model`, `ModelRef`, `LlamaEngine`, `Engine`, `GenerationParams`, `GeneratedToken` (engine crate).
- Produces: the `closedai` binary with `closedai run` and `closedai --version`.

- [ ] **Step 1: Write the failing CLI test**

`crates/closedai-cli/tests/cli.rs`:
```rust
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_closedai"))
}

#[test]
fn version_flag_prints_version() {
    let out = bin().arg("--version").output().expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("closedai"));
}

#[test]
fn run_requires_a_model() {
    let out = bin().arg("run").output().expect("run");
    assert!(!out.status.success(), "run without --model should fail");
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p closedai-cli --test cli`
Expected: FAIL — `--version` not implemented (stub `main` prints but clap isn't wired) and `run` subcommand missing.

- [ ] **Step 3: Implement the CLI parser and dispatch**

Replace `crates/closedai-cli/src/main.rs`:
```rust
// SPDX-License-Identifier: AGPL-3.0-or-later

mod run;

use clap::{Parser, Subcommand};

/// closedAI — collectively run one open model.
#[derive(Parser)]
#[command(name = "closedai", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Load a model and generate text on this machine.
    Run(run::RunArgs),
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => run::run(args),
    }
}
```

- [ ] **Step 4: Implement the `run` subcommand**

`crates/closedai-cli/src/run.rs`:
```rust
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::io::Write;
use std::ops::ControlFlow;
use std::time::Instant;

use clap::Args;
use closedai_engine::{resolve_model, Engine, GenerationParams, LlamaEngine, ModelRef};

#[derive(Args)]
pub struct RunArgs {
    /// Model reference: a local .gguf path or hf.co/<owner>/<repo>/<file.gguf>.
    #[arg(long)]
    model: String,

    /// Prompt. If omitted, read a single line from stdin.
    prompt: Option<String>,

    #[arg(long, default_value_t = 512)]
    max_tokens: usize,

    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    #[arg(long, default_value_t = 0)]
    seed: u32,

    /// Context window size.
    #[arg(long, default_value_t = 4096)]
    ctx: u32,

    /// Layers to offload to GPU (needs a GPU cargo feature). 0 = CPU only.
    #[arg(long, default_value_t = 0)]
    gpu_layers: u32,
}

pub fn run(args: RunArgs) -> anyhow::Result<()> {
    let model_ref = ModelRef::parse(&args.model)?;
    let path = resolve_model(&model_ref)?;

    let prompt = match args.prompt {
        Some(p) => p,
        None => {
            let mut line = String::new();
            std::io::stdin().read_line(&mut line)?;
            line.trim_end().to_string()
        }
    };

    let mut engine = LlamaEngine::load(&path, args.ctx, args.gpu_layers)?;
    let params = GenerationParams {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        seed: args.seed,
        stop: Vec::new(),
    };

    let start = Instant::now();
    let mut stdout = std::io::stdout();
    let outcome = engine.generate(&prompt, &params, &mut |tok| {
        let _ = stdout.write_all(tok.text.as_bytes());
        let _ = stdout.flush();
        ControlFlow::Continue(())
    })?;
    let elapsed = start.elapsed();

    let tok_per_s = outcome.tokens_generated as f64 / elapsed.as_secs_f64().max(1e-9);
    eprintln!(
        "\n[{} prompt tokens, {} generated, {:.1} tok/s, {}]",
        outcome.prompt_tokens,
        outcome.tokens_generated,
        tok_per_s,
        format_stop(outcome.stop_reason),
    );
    Ok(())
}

fn format_stop(reason: closedai_engine::StopReason) -> &'static str {
    use closedai_engine::StopReason::*;
    match reason {
        Eos => "eos",
        MaxTokens => "max_tokens",
        StopString => "stop_string",
        Cancelled => "cancelled",
    }
}
```

- [ ] **Step 5: Run the CLI tests to verify they pass**

Run: `cargo test -p closedai-cli --test cli`
Expected: PASS — both cases green.

- [ ] **Step 6: Manual end-to-end check**

Run:
```bash
cargo run -p closedai-cli -- run --model .closedai-cache/smol.gguf --max-tokens 20 --temperature 0 "The capital of France is"
```
Expected: streams text to stdout, then a stats line on stderr like `[7 prompt tokens, 20 generated, N.N tok/s, max_tokens]`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(cli): add run subcommand with streaming output and tok/s stats"
```

---

## Task 7: Verify the llama.cpp pin (CVE floor and flagship architecture)

**Files:**
- Create: `docs/engine-pin.md`
- Modify: `crates/closedai-engine/tests/generation.rs` (add an architecture smoke test)

**Interfaces:**
- Consumes: `LlamaEngine` (Task 4).
- Produces: `docs/engine-pin.md` recording the exact `llama-cpp-2` version, the bundled llama.cpp build, and evidence it satisfies the two constraints (>= b8492; loads `deepseek2`/MLA).

- [ ] **Step 1: Record the bundled llama.cpp build**

Find the llama.cpp commit/build the pinned crate vendors:
```bash
find ~/.cargo -path '*llama-cpp-sys-2*/llama.cpp/*' -name 'build-info.*' 2>/dev/null | head
grep -RhoE 'b[0-9]{4,}' $(find ~/.cargo -path '*llama-cpp-sys-2*' -name '*.h' 2>/dev/null) 2>/dev/null | sort -u | tail
```
Record the discovered build number in `docs/engine-pin.md`. If it is below **b8492**, bump `llama-cpp-2` to the lowest version whose bundle is at or above b8492 and repeat.

`docs/engine-pin.md` (fill the bracketed values with what you observed):
```markdown
# Engine pin

- `llama-cpp-2` version: `=0.1.150`
- Bundled llama.cpp build: `[bXXXX]`
- CVE-2026-34159 (RCE in the RPC backend, fixed in b8492): satisfied because the
  bundled build is `[bXXXX] >= b8492`. closedAI additionally never uses the RPC
  backend, so the affected code path is unreachable.
- Flagship architecture (`deepseek2` / MLA, GLM-4.7-Flash): verified by the
  architecture smoke test below.
```

- [ ] **Step 2: Write the architecture smoke test**

Append to `crates/closedai-engine/tests/generation.rs`:
```rust
/// Verifies the pinned llama.cpp loads the flagship architecture family.
/// Set CLOSEDAI_ARCH_MODEL to a small `deepseek2`/MLA GGUF (for example a
/// GLM-4.7-Flash quant, or the smallest available DeepSeek-family GGUF).
#[test]
fn loads_flagship_architecture() {
    let Some(model) = std::env::var_os("CLOSEDAI_ARCH_MODEL").map(std::path::PathBuf::from) else {
        eprintln!("skipping: set CLOSEDAI_ARCH_MODEL to a deepseek2/MLA GGUF to run");
        return;
    };
    let engine = closedai_engine::LlamaEngine::load(&model, 2048, 0);
    assert!(engine.is_ok(), "pinned llama.cpp must load deepseek2/MLA: {engine:?}");
}
```

- [ ] **Step 3: Run the smoke test against a flagship-family model**

```bash
# Example: a small deepseek2/MLA GGUF. Substitute the smallest you can fetch.
CLOSEDAI_ARCH_MODEL=/path/to/glm-4.7-flash-Q4_K_M.gguf \
  cargo test -p closedai-engine --test generation loads_flagship_architecture
```
Expected: PASS — the model loads. If it fails with an unknown-architecture error, the pin is too old; bump `llama-cpp-2`, redo Step 1, and re-run.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: record and verify the llama.cpp pin (CVE floor and flagship arch)"
```

---

## Definition of Done (Milestone 01)

- `cargo test --workspace` passes offline (unit + CLI tests).
- With a tiny GGUF, `CLOSEDAI_TEST_MODEL=... cargo test -p closedai-engine --test generation` passes.
- `closedai run --model <path-or-hf.co-ref> "<prompt>"` streams coherent text and prints a tok/s stats line.
- `docs/engine-pin.md` records a bundled llama.cpp build `>= b8492` that loads the `deepseek2`/MLA flagship architecture.
- `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --all -- --check` are clean.
- All inference goes through the `Engine` trait; no `llama_cpp_2` types appear outside `closedai-engine`.

## Notes for later milestones

- The `Engine::generate` signature already streams tokens and reports a
  `GenerationOutcome` with timings-friendly counts — Milestone 06 extends the
  outcome with network-vs-compute timing without changing callers.
- `LlamaEngine::load` takes an explicit path; Milestone 03 replaces the whole-file
  path with a sparse, span-restricted file produced by the partial-GGUF loader,
  behind the same constructor shape.
- Keep `LlamaBackend` construction in one place — Milestone 02 will need a second
  process that constructs its own backend for its layer span.
