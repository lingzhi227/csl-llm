# CSL-LLM

CSL kernels and distributed execution components for real-model inference on Cerebras WSE-3, developed with SDK 2.10.1. The target is **Qwen2.5-0.5B-Instruct**, using official weights, all 24 layers and the full vocabulary.

**This is an early research release of validated components, not a working end-to-end LLM runtime.** A complete native decoder layer0 now passes a bounded five-call SDK test; complete24-layer generation and full-model capacity acceptance remain under development. The project builds on Pragma HLS experience; this release contains handwritten CSL and Python planning/validation tools, not a complete HLS model compiler.

## Start here

- [What has passed, and at what scale](docs/STATUS.md)
- [Setup and reproduction](docs/REPRODUCING.md)
- [Kernel contracts](csl/kernels/README.md)
- [Acceptance requirements](docs/ACCEPTANCE.md) and [development stages](DEVELOPMENT-PLAN.md)
- [Evidence scope](validation/README.md) and [source attribution](THIRD_PARTY_NOTICES.md)

## Published components

| Component | Recorded SDK scope |
| --- | --- |
| Linear algebra | FP32 local GEMV; lossless packed-BF16 weights with FP32 accumulation; real 128×896 contraction over eight PEs, ten calls |
| Regional communication | Two translated eight-PE regions, repeated device epochs and reset sequences |
| Vector operators | 21 cases covering RMSNorm, residual addition, SwiGLU slices, finite softmax and selected alias behavior |
| Qwen RoPE | 14 query / 2 key heads; nine position/alias cases, including position 2047 |
| Embedding and selection | Seven cases for a 128×112 embedding tile and 128-logit argmax/pair merge; not the full vocabulary |
| Model reference | Verified checkpoint inventory and independent equations compared with official FP32 reference over 42 fixed-prefix steps |

Recorded SDK results belong to the original development snapshots. Publication changes make host paths portable; they have not received a fresh complete SDK qualification. See [release checks](release/CHECKS.md).

## New accepted milestones

Stateful GQA (34 real reference tokens), a separately scoped 2048-position KV diagnostic, sequential packet ACK completion and exact full-checkpoint BF16 packing are now included. [Read their acceptance boundaries](docs/MILESTONES.md). These are component milestones, not complete-model generation.

The full layer-0 **UP 4864×896 projection** also passes four input cases on 304 PEs. This is a complete single projection, not by itself a complete MLP. The published tool selects the accepted u16-transfer snapshot; later transport optimizations are not included.

The complete layer-0 **GATE 4864×896 projection** now also passes four cases on 304 PEs using paired u32 BF16 transfer and column readback. Its separate entry point preserves the historical UP/u16 qualification. Different UP/GATE runs do not establish a transport speedup ratio.

The complete layer-0 **DOWN 896×4864 projection** now also passes four cases on 308 PEs, including the padded final input shard. UP, GATE and DOWN have each passed separately; their device-side MLP composition subsequently passed in a separate static-weight run.

## Complete resident MLP milestone

**UP/GATE/SwiGLU/DOWN now run together with device-only intermediates**,916 resident weight tiles and four independently checked input cases. [Evidence, reproduction and limits](docs/RESIDENT-MLP.md). RMSNorm/residual were subsequently accepted in the bounded layer0 decoder; complete-model integration remains pending. Generated CSL weight literals are excluded from this repository.

An accepted **DSD-copy optimization reduces measured controller cycles by49.57% on the nonzero MLP cases**, with bit-identical results and protocol observations. This is a simulator interval comparison, not a hardware speedup. Both implementations and [comparison limits](docs/RESIDENT-MLP.md) are retained.

## Complete decoder layer milestone

**The complete native layer0 now passes five calls with causal GQA, persistent KV, both RMSNorm/residual paths and MLP.** [Scope and reproduction](docs/DECODER.md). The separate token-controller tests are scripted metadata only;24-layer layout checks are offline only. Neither is complete-model inference.

A **one-tile native-u8 static-weight SDK probe** now passes full before/after weight readback and four GEMVs. Related compact initializer and streamed-layer checks are explicitly compile-only. [Source and reproduction boundaries](docs/NATIVE-U8.md).

An **experimental two-PE intact-artifact composition** now passes isolated SDK runtime comparison.256/1024 real-tile compilation probes remain separately scoped compiler results with zero-filled placeholders. [Limits and source prerequisites](docs/PARTITION-EXPERIMENT.md).

The **eight-PE cross-partition line allreduce** now also passes actual SDK execution for a complete 128 × 896 contraction, with independent numerical checks and exact direct/composed output comparison. [Evidence and boundaries](docs/PARTITION-EXPERIMENT.md#eight-pe-cross-partition-collective-acceptance). Full-model acceptance remains open.

## Repository layout

```text
csl/kernels/       Reusable local CSL operators and contracts
csl/runtime/       Distributed linear execution and line collectives
src/csl_llm/       Region contracts and candidate model layout planner
tools/             Model/reference preparation, SDK probes and resource inspection
tests/             Host-side region, ELF and process-lifecycle tests
configs/           Pinned model, precision policy and reference prompts
docs/              Scope, reproduction, placement and acceptance
validation/reviews/ Selected independent historical review records
evidence/          Small checkpoint metadata; fresh runs are generated locally
release/           Source snapshot hashes and publication checks
```

The current full-model placement candidate is **192×210 application PE positions**. It is a static proposal, not an accepted full-model deployment. Batch 1, total context 2048 and up to 256 generated tokens are targets, not demonstrated capacity.

Model weights, activation traces, SDK binaries, compiled device artifacts and unfinished experiments are not distributed. Obtain the pinned model and your own SDK installation as described in the reproduction guide. Physical-wafer execution has not been validated.
