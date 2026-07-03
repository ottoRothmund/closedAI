# closedAI — Design

**Status:** Draft for review
**Date:** 2026-07-03

## 1. Overview

closedAI is a free, open-source system for running large open-weights language
models on hardware nobody owns alone. Members install one daemon, contribute a
slice of memory and compute, and together host a single model that no member
could run by themselves. Contributing to a model earns the right to query it.

The unit of organisation is a **swarm**: a group of members that collectively run
exactly one model. Each member's daemon serves a contiguous span of the model's
layers and, in return, exposes an OpenAI-compatible API on `localhost` that routes
queries through the whole swarm. Any existing OpenAI client works unchanged.

The flagship is a single public community swarm running one frontier-class model,
including uncensored community fine-tunes. Anyone can also found their own swarm —
for a different model, a group of friends, or testing — with the same software.

closedAI is not competing on price or speed. Cloud providers serve the same open
weights faster and, per token, often cheaper than consumer electricity. The value
is different: run models that are otherwise out of reach, on hardware you and
people you trust control, with model choice left entirely to members. Speed is
honestly presented as a trade-off, never a promise.

### 1.1 Prior art this design learns from

- **Petals** proved WAN pipeline-parallel inference of 70B–405B models works
  (~1–6 tok/s) with elegant fault tolerance, then died from no incentives, a
  per-model porting burden, and maintainer departure. We copy its client-side
  design and reimplement its fault-tolerance idea; we do not inherit its codebase.
- **AI Horde** has run a public volunteer text/image swarm since 2022 on
  non-monetary "kudos" (earned by serving, spent as queue priority, never sold).
  It has no cross-machine sharding, so its largest models sit behind one GPU with
  thousand-deep queues — exactly the ceiling closedAI removes. We copy its
  reciprocity model and its one hard content line (CSAM), not a currency.
- **llama.cpp** is the inference engine. Its `rpc-server` is an explicitly
  insecure proof-of-concept with a 2026 critical RCE (CVE-2026-34159); we link
  llama.cpp as a library and build our own transport and executor instead of
  using its RPC wire protocol.
- **iroh** (1.0, June 2026, Rust) is the transport: dial-by-public-key over QUIC,
  ~90–95% hole-punch success, encrypted relay fallback over port 443, already used
  in production for distributed AI.

## 2. Goals and non-goals

### Goals

1. Let a group pool modest, mostly-spare resources to run one large model none of
   them could run alone.
2. Ollama-grade first-run experience: one install, one command to query.
3. Drop-in OpenAI API compatibility so the existing client ecosystem works day one.
4. Model choice is a member decision, including uncensored fine-tunes; the
   software is neutral.
5. Honest about performance, privacy, and trust. No false promises.
6. Built to be maintained: a small, boring, reliable core, permissive-to-copyleft
   licensing, and visible non-commercial governance.

### Non-goals (v1)

1. No custom inference engine — llama.cpp does the math.
2. No tensor parallelism over WAN — pipeline (layer-split) only.
3. No transferable credits, tokens, or cryptocurrency of any kind.
4. No hosted model registry or curated catalog — model references resolve to
   Hugging Face repositories.
5. No privacy claim beyond "every member of a swarm can see everything that
   passes through it."
6. No per-user durable sessions — queries are stateless.
7. No dense models above ~100B as a flagship — sparse Mixture-of-Experts only,
   because that is what makes small per-member contributions viable.

## 3. Core concepts

**Swarm.** A named group running one model, defined by a signed manifest. The
flagship community swarm is just the default one; the software treats all swarms
identically.

**Manifest.** The signed root-of-trust document for a swarm. It names the model
(Hugging Face repository, content digests, quantization), the pinned llama.cpp
build hash, the layer→span plan parameters, the membership policy, and a list of
bootstrap peer public keys. A manifest is a file, not a service — it can be hosted
on GitHub, IPFS, or shared as a link.

**Member.** A person running the daemon and contributing to a swarm, identified by
an Ed25519 key.

**Node roles.** A member's daemon plays one or both roles per swarm:

- **Serving slot** — holds a contiguous layer span and sits in the inference
  pipeline. The number of serving slots is the pipeline depth, which is the primary
  driver of per-token latency. The coordinator packs spans onto the fewest, most
  capable, most reliable nodes to keep this number small.
- **Replica** — also holds a span, but is not necessarily on any given query's
  critical path. Replicas provide failover, let the router choose the
  lowest-latency copy, and serve concurrent queries in parallel. Most spare or
  background contributors land here.

This split is the central design decision (see §7). It decouples "contribute a
little" from "add a pipeline hop", which is what makes a swarm of many small
contributors physically viable.

**Standing.** A member is in *good standing* when their node is reachable, serving
its assigned span, and has met a rolling uptime threshold. Good standing grants
query access; recent contribution weights queue priority. Standing is the entire
incentive mechanism — nothing is transferable, purchasable, or saleable.

**Query.** A stateless request. There are no durable per-user sessions; each query
is a fresh pass through the pipeline, and per-query KV cache on serving nodes is
released when the query completes.

## 4. Architecture

Three logical components, all shipped in one Rust binary.

### 4.1 Member daemon

The always-running process on each member's machine. Responsibilities:

- **Shard runner** — loads its assigned layer span (only that span's weights; see
  §8) and executes those layers via llama.cpp linked as a library, exposing a
  span-restricted forward pass over the transport.
- **API gateway** — serves the OpenAI-compatible API on `localhost`. When the local
  member queries, this gateway is the pipeline *head*: it holds the token
  embeddings and LM head locally, drives the pipeline, and does all sampling
  locally, so no other node ever sees logits or sampling parameters.
- **Contribution reporter** — reports health and serving activity to the current
  coordinator and attests to work it participated in.

### 4.2 Coordinator role

Control plane only. It never sees prompts or activations. Responsibilities:
membership registry (keys and declared budgets), span assignment with replication,
health and standing accounting, and latency-aware routing hints.

The coordinator is not a dedicated server. It runs *inside* member daemons:
always-on members advertise willingness to coordinate; one holds a short,
renewable lease while others stand by. Its state is soft and fully reconstructible
from member self-reports plus the manifest, so on coordinator loss another member
assumes the role and rebuilds state within seconds. This yields a tracker's
simplicity (a single authority computing assignments, no DHT eclipse surface) with
P2P's resilience and, critically, **no requirement to operate a VPS or any
dedicated infrastructure**. Bootstrap relies only on: a statically hosted manifest,
iroh's free public relays for NAT traversal, and volunteered always-on member nodes.

### 4.3 Transport

iroh everywhere. Every link is an authenticated, end-to-end-encrypted QUIC stream
dialed by public key. No component ever listens on a raw, unauthenticated TCP port,
which structurally eliminates the CVE-2026-34159 exposure class. Direct connections
are used when hole-punching succeeds (~90–95%); otherwise traffic falls back
transparently to encrypted relays over port 443. The transport is behind an
internal abstraction so it can evolve without touching the executor or scheduler.

### 4.4 Engine

llama.cpp is vendored and pinned to a build at or after the CVE fix (b8492). It is
linked as a library; closedAI drives it directly and never uses its `rpc-server` or
RPC wire protocol. Support launches for a single model-architecture family (the
flagship's). Additional families are added through small, per-family adapters over
llama.cpp's kernels — an explicitly budgeted, bounded maintenance cost, in
deliberate contrast to Petals' unbounded per-model PyTorch porting burden.

## 5. The inference path

1. **Query arrives** at the local daemon's OpenAI-compatible endpoint.
2. **Head setup.** The local daemon embeds the prompt tokens and prepares to sample.
3. **Pipeline assembly.** Using the coordinator's routing hints, the daemon selects
   one serving replica per span such that the chain minimises total measured RTT and
   avoids relayed hops mid-pipeline (a relayed hop caps the whole chain).
4. **Prefill.** Prompt activations are streamed span-to-span. Prefill cost scales
   with prompt length across every span boundary, so prefill uses chunking and
   per-node prompt caching from day one — this is the dominant contributor to
   time-to-first-token and is designed in, not deferred.
5. **Decode.** For each generated token, the head sends the current hidden state
   through the serving chain; the tail returns the final hidden state; the head
   applies the LM head and samples the next token locally. Only a small hidden-state
   vector crosses each boundary per token, so bandwidth is trivial and latency
   dominates.
6. **Completion.** The head streams tokens to the client with honest telemetry
   (tokens/second, and network-versus-compute time) in the response. Per-query KV
   cache on serving nodes is released.

### 5.1 Mixture-of-Experts handling

The flagship and all recommended models are sparse MoE. Splitting is still
contiguous layer-split: a node holding a layer span holds all experts for those
layers. Expert offload to CPU RAM, when needed, happens locally on each node; there
is no cross-node expert routing. This matters because MoE memory scales with total
parameters (what the swarm pools) while per-token compute scales with active
parameters (what each node and each hop must carry) — the property that lets modest
nodes participate at all.

## 6. Coordination and discovery

- **Bootstrap.** A joining daemon reads the swarm manifest, verifies its signature,
  and dials the listed bootstrap peers by key over iroh. No central endpoint is
  contacted; the manifest can live anywhere.
- **Membership.** The coordinator records member keys and declared budgets
  (memory, always-on or intermittent, link type). Members re-announce periodically;
  silent members are aged out of assignments.
- **Assignment.** The coordinator computes span placement (see §7) and distributes
  the current assignment map. Assignments change slowly, with hysteresis, to avoid
  thrashing when nodes flap.
- **Coordinator failover.** The lease-holding coordinator heartbeats to standbys.
  On lease expiry, a deterministic election among always-on members promotes a new
  coordinator, which rebuilds state from the next round of member announcements.

## 7. Contribution and access model

This section is the heart of the design, because the stated intent — many members
each contributing mostly spare, background resources — is in direct tension with
pipeline latency.

### 7.1 The physics

Per-token latency is approximately the sum of the round-trip times of the hops a
token traverses. A model split across *N* serving slots costs roughly *N* hop-RTTs
per token. Across households, a hop is ~15–50 ms. Naively splitting a 140 GB model
into 4 GB slices would mean ~35 serving slots, ~35 hops, and roughly 1 token/second
or worse — unusable.

### 7.2 The resolution: serving slots versus replicas

Latency is set by the *number of serving slots*, not the *number of members*. The
coordinator therefore:

- **Minimises pipeline depth** by packing spans onto the fewest, largest,
  most-reliable nodes that can cover the model. A few capable always-on nodes define
  the critical path.
- **Absorbs everyone else as replicas.** Spare and background contributors hold
  copies of spans off the critical path. They add failover, latency choice, and
  concurrent-query throughput — never pipeline depth.

So a swarm of 3 capable always-on nodes and 30 spare laptops runs as a ~3-hop
pipeline with ~10× replication, not a 33-hop chain. Every contribution counts
toward standing; only a few nodes determine speed.

**Honest consequence:** a swarm's achievable tokens/second is fundamentally bounded
by its best always-on hardware and the resulting pipeline depth. Spare resources
make a swarm more resilient and more scalable under concurrent load; they do not
make it faster. The product surfaces this rather than hiding it.

### 7.3 Access

- Good standing (reachable + serving + rolling uptime threshold) grants query
  access.
- Queue priority is weighted by recent contribution, so members who serve more wait
  less under contention.
- New members get a short grace period to reach standing before their access is
  gated.
- Nothing about standing is transferable, purchasable, or saleable — by design and
  by the project's terms — which removes the reseller and Sybil-farming incentives
  that monetary schemes attract.

## 8. Weights and models

- **One model per swarm.** The model is fixed in the manifest. To run a different
  model, join or found a different swarm. This keeps span assignment, verification,
  and caching simple and matches the "everyone powers the same model" intent.
- **Swappable flagship.** The flagship model is a single manifest field. The
  reference default targets the ~235B–355B sparse-MoE class (for example a Qwen3- or
  GLM-class model, or an abliterated variant), which pools across roughly a dozen
  modest nodes at a realistic few tokens/second. A swarm can launch with a smaller
  model and move up as its always-on capacity grows.
- **Peer-local span loading.** Each node loads only the weights for its assigned
  layer span. The daemon parses the GGUF tensor directory, computes the byte ranges
  for its span's tensors, and fetches only those ranges from Hugging Face via HTTP
  Range requests into a sparse local file, verifying content digests. This is a core
  build item (llama.cpp's default loader expects a whole-file model), and it is what
  makes contributing a slice — rather than downloading the entire model — possible.
- **No redistribution.** Every node fetches weights directly from Hugging Face under
  its own account and license acceptance. closedAI never mirrors or serves weights
  through any infrastructure it operates. This sidesteps model-license redistribution
  obligations (for example the Llama Community License) and keeps the project a
  neutral conduit. Recommended defaults are permissively licensed weights (MIT/
  Apache-class: DeepSeek, Qwen, Mistral) where mirroring would be clean anyway; the
  daemon surfaces each model's license metadata to swarm founders.

## 9. Trust, privacy, and safety

closedAI is honest about a hard fact: in any swarm, nodes serving the early layers
can reconstruct prompts from activations, and no consumer GPU offers a hardware
trusted-execution environment to prevent it. The design does not pretend otherwise.

- **Privacy.** The UI states plainly that every member of a swarm can see prompts
  and outputs that pass through it. The privacy story is social: you choose which
  swarm to join and who is in it. There is no cryptographic privacy claim in v1.
- **Output integrity (v1).** Lightweight redundant-replica sampling with
  disagreement flagging: a fraction of tokens or segments are computed by a second
  replica and compared, and persistent divergence flags a node for review. Naive
  bit-exact redundancy is explicitly avoided because floating-point non-associativity
  makes greedy decoding non-reproducible across heterogeneous hardware. Stronger,
  research-grade verification (activation-hash spot-checks in the TOPLOC family) is
  documented as the v3 public-swarm hardening path, not v1 scope.
- **Content.** Model choice is neutral, including uncensored fine-tunes. There is
  exactly one platform-wide, non-negotiable line: automated CSAM filtering, following
  AI Horde's operational precedent. Individual node operators may additionally
  decline to serve specific models with their hardware.
- **Legal posture.** v1 is trust-scoped: within a swarm, members are known to each
  other, weights are pulled by each node from Hugging Face directly, and no central
  service touches prompts or outputs. The terms make operator consent per model and
  the absence of privacy explicit. Broader public-swarm content and jurisdiction
  questions belong to the later public phase.

## 10. Failure handling and degradation

- **Node loss mid-generation.** Because closedAI owns its executor (rather than
  using llama.cpp RPC, which cannot recover mid-generation), the head journals the
  activations it has sent. On losing a serving node it re-routes to a replica and
  replays the journal to rebuild that span's attention state, resuming the generation
  rather than restarting it.
- **No available replica.** If a span has no surviving server, the query degrades to
  the largest model the surviving nodes can serve, or fails cleanly with an
  explanation — never a silent hang.
- **Preflight honesty.** Before any model download, the daemon probes per-pair RTT,
  bandwidth, and memory, then predicts tokens/second and time-to-first-token for the
  swarm's model. It warns hard on Wi-Fi links (observed ~10× throughput loss versus
  wired) and on configurations that cannot meet a usable floor.
- **Residency.** Model residency defaults to pinned/long, not a short keep-alive,
  because loading a large model across many nodes is expensive; `status` shows warm
  and cold state per node.
- **Version and model skew.** The manifest pins the engine build hash and model
  content digests. Joining nodes must match or are refused; `status` surfaces any
  skew explicitly. The engine is embedded in the daemon so members cannot
  accidentally mix incompatible builds.

## 11. UX, CLI, and API surface

The bar is Ollama's first-run feel; the divergence is that a swarm is multi-party
and must not hide that.

- **Install.** One self-contained binary per OS with automatic GPU detection
  (CUDA, ROCm, Metal, Vulkan via llama.cpp).
- **Query.** `closedai run` connects to the member's swarm and opens a chat; the
  first run joins, loads the assigned span, and begins serving in one step.
- **Swarm as a first-class noun.** `closedai swarm join <manifest>`,
  `swarm leave`, `swarm found`, `swarm status`. `status` shows the member's assigned
  span, per-node placement across the swarm, warm/cold state, link health, and the
  member's standing.
- **API.** OpenAI-compatible `/v1/chat/completions`, `/v1/completions`,
  `/v1/models`, and `/v1/embeddings` on `localhost`, so Open WebUI, IDE plugins, and
  other existing clients point at the swarm unchanged and see it as one large model.
- **Telemetry.** The streaming response carries tokens/second and network-versus-
  compute timing, making the distributed cost visible and honest.

## 12. Performance model

- Realistic single-query throughput for a large MoE across a modest cross-household
  swarm is on the order of a few tokens per second, bounded by pipeline depth and
  the slowest serving node. Wired links and fewer, larger serving slots move it
  toward the upper single digits; Wi-Fi and many small serving slots move it toward
  one.
- Time-to-first-token is dominated by distributed prefill and can reach tens of
  seconds on large prompts and deep pipelines; chunked prefill and prompt caching
  are the mitigations.
- The marketing and onboarding copy reflect this: "run what you otherwise could
  not," never a speed or price claim.

## 13. Security

- All inter-node traffic is authenticated and end-to-end encrypted via iroh; nodes
  are addressed by key.
- No raw, unauthenticated ports are ever exposed; the historically insecure
  llama.cpp RPC path is not used.
- llama.cpp is pinned past the CVE-2026-34159 fix and vendored, so members cannot
  run a vulnerable or mismatched build.
- Manifests are signed; joining nodes verify signatures before trusting assignments
  or bootstrap keys.
- A CI matrix exercises loading and serving of large (100 GB-plus) models split
  across several heterogeneous nodes, because the field's failure mode is large-model
  edge cases, not small-model demos.

## 14. Scope and phasing

**v1 — private and community swarms, the core loop.**
Rust daemon and CLI; iroh transport; floating coordinator with no required
infrastructure; peer-local span loading from Hugging Face; layer-split pipeline
inference with client-side sampling and activation-replay fault tolerance;
OpenAI-compatible API; standing-based access with a contribution dashboard; honest
preflight and telemetry; one model-architecture family; one signed community-swarm
manifest as the flagship. No credits, no public-swarm trust machinery, no registry.

**v2 — reach and ergonomics.**
Additional model-architecture families; self-hostable relays for swarms that want
zero third-party dependency; richer `swarm found` tooling and manifest publishing;
optional tensor-split within a single household for co-located nodes.

**v3 — public swarm (only if warranted).**
Open, permissionless swarms with the machinery that requires: non-monetary,
non-transferable credits in the AI Horde tradition; research-grade output
verification (activation-hash spot-checks with cohort-matched verifiers and adaptive
spot-check rates); reputation with decay; and the full content-moderation and relay-
economics story. Treated as a separate product, not an incremental toggle, and only
after the trust-scoped tiers have proven themselves.

## 15. Open questions and risks

1. **Peer-local partial GGUF loading** is the highest-risk build item: llama.cpp's
   loader assumes a whole-file model, so span-restricted loading and execution needs
   real work against its internals and must track upstream changes.
2. **Achievable tok/s with genuinely spare hardware** is unproven for this design;
   the serving-slot/replica split is the mitigation, but the flagship's realistic
   floor needs an early end-to-end benchmark on representative consumer nodes.
3. **Coordinator election and soft-state reconstruction** need a concrete protocol
   that stays correct under churn without becoming a consensus system.
4. **CSAM filtering for text** (versus AI Horde's image-oriented tooling) and its
   false-positive burden on uncensored-model users needs a concrete mechanism.
5. **License interpretation:** this document specifies AGPL-3.0-or-later as the
   copyleft license (network copyleft, validated in this space by AI Horde and Jan);
   confirm this matches the intended "copyleft" before finalising.
6. **Abandonment is the field's proven killer**, not technical inferiority. The
   mitigation is scope discipline and a published sustainability and governance
   stance from day one; it must be honoured, not just written down.
