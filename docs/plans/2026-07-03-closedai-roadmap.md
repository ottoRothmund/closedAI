# closedAI v1 — Implementation Roadmap

The [design spec](../specs/2026-07-03-closedai-design.md) covers several
independent subsystems. It is therefore built as a sequence of milestone plans,
each producing working, testable software on its own and building on the last.
Only the current milestone is planned in bite-sized detail; each subsequent plan
is written once its predecessor is built, so it reflects what the earlier work
actually revealed.

| # | Plan | Deliverable | Depends on |
|---|------|-------------|------------|
| 01 | **Local inference core** | Rust workspace + llama.cpp behind an `Engine` trait; `closedai run <model>` generates text on one machine; whole-file Hugging Face download. | — |
| 02 | Layer-span execution + local pipeline | Split a model into contiguous layer spans; two local processes each run a span and pass hidden states, producing output identical to single-process greedy decode. | 01 |
| 03 | Peer-local partial-GGUF loading | Parse the GGUF tensor directory, fetch only a span's byte ranges from Hugging Face (HTTP Range) into a sparse file, digest-verified, and load just that span. | 02 |
| 04 | iroh transport | Replace the local socket with iroh QUIC streams dialed by key; run the pipeline across two machines. | 03 |
| 05 | Manifest + coordinator + assignment | Signed manifest; membership; serving-slot vs replica span assignment; floating coordinator with lease and soft-state reconstruction; routing hints. | 04 |
| 06 | OpenAI-compatible API gateway | `/v1/chat/completions` and friends on localhost; pipeline head holds embeddings + LM head + sampling; streaming with tok/s and network-vs-compute telemetry. | 05 |
| 07 | Standing, access, preflight | Rolling-uptime standing; contribution-weighted priority; grace period; contribution dashboard; preflight probe predicting tok/s and warning on Wi-Fi / oversized models. | 06 |
| 08 | Fault tolerance + degradation | Client activation-replay journal; replica failover mid-generation; graceful degradation; engine-build and model-digest skew pinning. | 05 |
| 09 | CLI, packaging, flagship manifest | `swarm join/leave/found/status`; single-binary packaging per OS; the flagship community-swarm manifest (GLM-4.7-Flash uncensored). | 06, 07, 08 |

Milestones 02–09 are deferred; this repository currently plans **Milestone 01**
(`2026-07-03-closedai-01-local-inference-core.md`).
