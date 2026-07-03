# Engine pin

This records the evidence that closedAI's pinned inference engine satisfies the
two Milestone 01 constraints: (1) the bundled llama.cpp is at or after the
CVE-2026-34159 fix, and (2) it loads the flagship `deepseek2`/MLA architecture
family that GLM-4.7-Flash uses.

- `llama-cpp-2` version: `=0.1.150` (`Cargo.toml:20`; `llama-cpp-sys-2` pulled in
  transitively at the same version, `Cargo.lock`).
- Bundled llama.cpp build: **no numeric `bXXXX` could be recovered from the
  vendored source** (see "Build number" below). Corroborating, dated evidence
  (crate publish date, upstream PR dates, and the compiled-in architecture set)
  places the bundle after both the CVE-2026-34159 fix and the GLM-4.7-Flash
  architecture support landing — see "CVE-2026-34159" below.
- CVE-2026-34159 (RCE in the RPC backend, fixed in b8492): satisfied on two
  independent grounds — bundle recency (below) AND closedAI never uses the RPC
  backend, so the vulnerable code path is unreachable regardless of build.
- Flagship architecture (`deepseek2`/MLA, GLM-4.7-Flash): verified statically by
  source grep (below) and, where practical, by the live architecture smoke test.

## Build number: why it can't be pinned exactly, and what stands in for it

The task brief's suggested lookup —
`find ~/.cargo -path '*llama-cpp-sys-2*/llama.cpp/*' -name 'build-info.*'` plus a
`grep` for `b[0-9]{4,}` in vendored headers — was run and confirms *why* no
number is available, rather than finding one:

- `llama.cpp/cmake/build-info.cmake` derives `BUILD_NUMBER` and `BUILD_COMMIT` by
  shelling out to `git rev-list --count HEAD` / `git rev-parse --short HEAD`
  **at CMake-configure time**, with the fallback `set(BUILD_NUMBER 0)` /
  `set(BUILD_COMMIT "unknown")` when git isn't available.
- The crates.io tarball ships no `.git` directory: `find <crate-root> -maxdepth 3
  -iname '.git*'` returns nothing. `llama.cpp/common/build-info.cpp.in` (the
  template that stamps `LLAMA_BUILD_NUMBER`/`LLAMA_COMMIT`/`llama_build_info()`
  into the compiled binary as `"b" + BUILD_NUMBER + "-" + BUILD_COMMIT`) is
  therefore filled in with the `0`/`"unknown"` fallback at our build time, not a
  real upstream build number — the `bXXXX` scheme genuinely is not recoverable
  from this source tarball, exactly as the task brief anticipated.

What the tarball *does* carry, as a substitute paper trail:

- `.cargo_vcs_info.json` at the crate root: `{"git":{"sha1":
  "5459e4dfe57bdb0de41bead8c7e8386ea7961368"},"path_in_vcs":"llama-cpp-sys-2"}`.
  This is the commit of the **wrapper repo** (`utilityai/llama-cpp-rs`, which
  vendors upstream `ggml-org/llama.cpp` as a submodule) that `cargo publish` was
  run from — not an upstream llama.cpp commit, but it identifies exactly which
  publish this is. It corresponds to `utilityai/llama-cpp-rs` PR #1045,
  "version-bump-0.1.150" ("Bumped version to 0.1.150").
- crates.io records `llama-cpp-2` `0.1.150` as published **2026-06-16T22:02:18Z**
  by `MarcusDunn` (checksum `e82ec86fd73aaa7d8f4898cc4693410c217431506ba3237ea5c856127f49d84d`,
  confirmed via `https://crates.io/api/v1/crates/llama-cpp-2/0.1.150`).
- The compiled-in architecture set itself is a dating signal: the vendored
  `llama.cpp/src/llama-arch.h` enum includes `LLM_ARCH_DEEPSEEK2OCR`,
  `LLM_ARCH_DEEPSEEK32`, `LLM_ARCH_GLM_DSA`, `LLM_ARCH_MISTRAL4`,
  `LLM_ARCH_JAIS2`, and a `models/kimi-linear.cpp` builder — architectures that
  postdate GLM-4.7-Flash's own upstream support (see below), which itself
  postdates b8492.

Net: the exact `bXXXX` is unavailable by construction (no git metadata in the
tarball), so per the task brief's fallback instruction this document relies on
the two corroborating, independently-dated facts below instead of a build
number.

## CVE-2026-34159

CVE-2026-34159 is a critical (CVSS 9.8) unauthenticated RCE in llama.cpp's RPC
backend: `deserialize_tensor()` skips bounds validation when a tensor's `buffer`
field is `0`, letting a remote peer read/write arbitrary process memory via
crafted `GRAPH_COMPUTE` messages (and leak pointers via `ALLOC_BUFFER`/
`BUFFER_GET_BASE` to defeat ASLR). Fixed upstream in build **b8492**
(GitHub advisory `GHSA-j8rj-fmpv-wcxw`, tagged around **2026-03-23**).

### (a) Bundle recency

- `ggml-org/llama.cpp` build **b8492** (the fix) was tagged **~2026-03-23**.
- Upstream support for GLM-4.7-Flash's `Glm4MoeLiteForCausalLM` landed in
  `ggml-org/llama.cpp` **PR #18936** ("support Glm4MoeLite"), merged
  **2026-01-19** — i.e. *before* b8492.
- Our pinned crate (`llama-cpp-2` `=0.1.150`) was published **2026-06-16**, ~3
  months *after* b8492, and its vendored `llama.cpp/src/` contains architecture
  families visibly newer than the January GLM-4.7-Flash PR (`DEEPSEEK2OCR`,
  `DEEPSEEK32`, `GLM_DSA`, `kimi-linear`, etc. — see the full enum excerpt
  below).
- Ordering: GLM-4.7-Flash support (2026-01-19) < CVE fix b8492 (~2026-03-23) <
  our pinned crate's publish (2026-06-16). The bundle we ship comfortably
  postdates both.

### (b) The vulnerable code path is unreachable regardless

closedAI never starts or connects to llama.cpp's `rpc-server`, and the safe
Rust wrapper we depend on doesn't even expose RPC bindings:

```
$ grep -rniE 'rpc[-_]?server|llama_rpc|--rpc\b|rpc_backend|ggml_backend_rpc|GGML_RPC' crates/
(no output — exit code 1)

$ grep -rniE 'rpc' ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/llama-cpp-2-0.1.150/src/
(no output — exit code 1)
```

No `rpc` feature flags appear in any workspace `Cargo.toml` either. The RPC
backend is compiled as part of upstream llama.cpp's optional `ggml-rpc`
component, but closedAI's build never invokes `rpc-server` or dials an
`--rpc`-style remote endpoint — the affected `deserialize_tensor()` path is
never reached by anything closedAI does, independent of the exact build.

## Flagship architecture: `deepseek2` / MLA (GLM-4.7-Flash)

### Compiled-in architecture enum and GGUF name strings

`llama.cpp/src/llama-arch.h` (enum `llm_arch`, excerpt):
```
80:    LLM_ARCH_DEEPSEEK,
81:    LLM_ARCH_DEEPSEEK2,
82:    LLM_ARCH_DEEPSEEK2OCR,
83:    LLM_ARCH_DEEPSEEK32,
84:    LLM_ARCH_CHATGLM,
85:    LLM_ARCH_GLM4,
86:    LLM_ARCH_GLM4_MOE,
87:    LLM_ARCH_GLM_DSA,
```

`llama.cpp/src/llama-arch.cpp` (the table mapping each enum value to the GGUF
`general.architecture` string llama.cpp matches against when loading a model):
```
77:    { LLM_ARCH_DEEPSEEK2,        "deepseek2"        },
78:    { LLM_ARCH_DEEPSEEK2OCR,     "deepseek2-ocr"    },
79:    { LLM_ARCH_DEEPSEEK32,       "deepseek32"       },
82:    { LLM_ARCH_GLM4_MOE,         "glm4moe"          },
```

These aren't dead code: `llama.cpp/src/CMakeLists.txt:9` glob-compiles every
model builder —
```
file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")
```
— and `llama-cpp-sys-2`'s `build.rs` drives this exact CMakeLists.txt via the
`cmake` crate (`build.rs:7` `use cmake::Config;`, `build.rs:560`
`Config::new(&llama_src)`, `build.rs:919` `config.build()`), so
`llama.cpp/src/models/deepseek2.cpp` and `glm4-moe.cpp` are compiled into every
build of this crate, ours included. `ls llama.cpp/src/models/` also shows
`deepseek.cpp`, `deepseek2.cpp`, `deepseek2ocr.cpp`, `deepseek32.cpp`,
`glm4.cpp`, `glm4-moe.cpp`, `glm-dsa.cpp` all present.

### MLA support in `deepseek2.cpp`

`llama.cpp/src/models/deepseek2.cpp` implements real Multi-head Latent
Attention, not a stub — e.g.:
```
15:    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
16:    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl, false);
17:    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl, false);
...
59:    const bool is_mla = hparams.is_mla();
...
104:        if (is_mla) {
105:            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, 0);
106:            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, 0);
```
backed by `llama_hparams::is_mla()` in `llama.cpp/src/llama-hparams.cpp:240`:
```
bool llama_hparams::is_mla() const {
    assert((n_embd_head_k_mla_impl == 0 && n_embd_head_v_mla_impl == 0) ||
           (n_embd_head_k_mla_impl != 0 && n_embd_head_v_mla_impl != 0));
    return n_embd_head_k_mla_impl != 0 && n_embd_head_v_mla_impl != 0;
}
```
`is_mla()`/`n_embd_head_k_mla()`/`n_embd_head_v_mla()` are also referenced from
`llama-kv-cache.cpp` (`const bool has_v = !is_mla;` at line 243 — MLA changes the
KV-cache layout) and `llama-model.cpp` (the `print_info` line that emits the
`n_embd_head_*_mla` values seen in the live run below), i.e. MLA is load-bearing
across the model, not an isolated helper.

### GLM-4.7-Flash specifically loads as `deepseek2`

GLM-4.7-Flash's Hugging Face architecture is `Glm4MoeLiteForCausalLM`. Upstream
`ggml-org/llama.cpp` PR #18936 ("support Glm4MoeLite", merged 2026-01-19) adds
conversion/loading support for it — and per that PR's own discussion, it is
mapped onto the **existing `deepseek2` architecture**, not a new one: one
reviewer describes Glm4MoeLite as "a renamed version of GLM4Moe with
DeepseekV3Attention (uses MLA) and an added dense expert at the start"; another
confirms "deepseekv3 also have some dense layer at the start, so ... it seems
like just deepseek renamed"; the conclusion in-thread is "from the GGUF
perspective it's just renamed deepseek." This is exactly why this document
treats "loads `deepseek2`/MLA" and "loads GLM-4.7-Flash" as the same claim: a
GLM-4.7-Flash GGUF declares `general.architecture = "deepseek2"` and is loaded
by the `deepseek2.cpp` model builder quoted above, including its MLA path.

(The separate `LLM_ARCH_GLM4_MOE` / `"glm4moe"` entry is also compiled in, and
covers the earlier GLM-4.5/4.6 MoE family, which the same PR discussion
describes as GLM-4.7-Flash's structural predecessor. Its presence is additional
corroborating evidence of a broad, current flagship-MoE architecture set, even
though it isn't the tag GLM-4.7-Flash itself uses.)

Caveat found in `llama.cpp/src/models/glm4-moe.cpp`'s
`load_arch_hparams` (this is about the GLM4_MOE family's *own* loader, reached
by the `"glm4moe"` tag, not the `deepseek2` path GLM-4.7-Flash actually takes):
its `n_layer()` switch only names `GLM-4.5-Air`/`GLM-4.5`/`Solar Open` layer
counts explicitly and falls back to `LLM_TYPE_UNKNOWN` otherwise. That fallback
only affects the human-readable size label used in logs/`llama_model_desc()`;
it does not gate whether the model loads. Noted here for completeness since it
was visible in the same file, not because it affects the `deepseek2` path this
document is about.

## Architecture smoke test

`crates/closedai-engine/tests/generation.rs::loads_flagship_architecture` loads
whatever GGUF `CLOSEDAI_ARCH_MODEL` points at (an **absolute path** — `cargo
test` runs test binaries with their working directory set to the package
directory, not the repo root, so a relative path will not resolve as expected)
through the same `LlamaEngine::load` used in production, and asserts the load
succeeds. It no-ops (prints a skip notice, does not fail) when the env var is
unset, so `cargo test --workspace` stays green and offline by default.

### Live run

Executed against a real `deepseek2`/MLA model — `DeepSeek-V2-Lite` (`Q2_K`,
6.0 GB), the smallest practical member of the `deepseek2` family (the 18 GB
GLM-4.7-Flash flagship was not downloaded here; it declares the same
`general.architecture = deepseek2` and loads through the identical
`deepseek2.cpp` builder and MLA path documented above).

```
$ CLOSEDAI_ARCH_MODEL=/Users/otto/Dev/closedAI/.closedai-cache/deepseek-v2-lite.Q2_K.gguf \
    cargo test -p closedai-engine --test generation loads_flagship_architecture -- --nocapture

running 1 test
llama_model_loader: - kv   0:              general.architecture str = deepseek2
llama_model_loader: - kv  14:      deepseek2.attention.kv_lora_rank u32 = 512
print_info: arch                  = deepseek2
print_info: n_embd_head_k_mla     = 192
print_info: n_embd_head_v_mla     = 128
test loads_flagship_architecture ... ok
```

The pinned engine loaded the model, resolved its `deepseek2` architecture, and
populated the MLA head dimensions (`n_embd_head_k_mla = 192`,
`n_embd_head_v_mla = 128`) with the low-rank KV projection (`kv_lora_rank = 512`)
— confirming the flagship `deepseek2`/MLA path is not merely compiled in but
actually load-bearing at runtime. `cargo test --workspace` (no
`CLOSEDAI_ARCH_MODEL` set) still passes offline: the test no-ops when the env
var is unset.
