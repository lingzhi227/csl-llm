# CSL-LLM

CSL kernels and distributed execution components for real-model inference on Cerebras WSE-3, developed with SDK 2.10.1. The target is **Qwen2.5-0.5B-Instruct**, using official weights, all 24 layers and the full vocabulary.

**This is an early research release of validated components, not a working end-to-end LLM runtime.** The complete 151936-token tied embedding/head now passes three isolated SDK cases, and a complete native decoder layer0 passes a bounded five-call SDK test; complete24-layer generation and full-model capacity acceptance remain under development. The project builds on Pragma HLS experience; this release contains handwritten CSL and Python planning/validation tools, not a complete HLS model compiler.

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
| Embedding and selection | Original 151936 × 896 tied embedding/head, three isolated SDK cases, complete logits checked from actual stopped cores and ordinary device winner transfer; earlier tile tests retained |
| Model reference | Verified checkpoint inventory and independent equations compared with official FP32 reference over 42 fixed-prefix steps |

Recorded SDK results belong to the original development snapshots. Publication changes make host paths portable; they have not received a fresh complete SDK qualification. See [release checks](release/CHECKS.md).

## Latest accepted integration

**Original-weight layer 0 → layer 1 now passes position 0, cached position 1 and reset-to-position-0 in one persistent SDK instance.** All 2088 matrix blocks are original; 52 independent numerical checks pass and reset outputs repeat bit for bit. [Cached/reset source and evidence](docs/TWO-LAYER-CACHED.md). Full 24-layer inference and generation remain unfinished.

## Latest accepted component

**The complete original vocabulary now passes actual SDK execution.** All 9496 weight tiles are assembled; three cases verify all 151936 logits, zero/tie behavior, three original embedding rows and fresh-process bit-repeat. [Full evidence, source map and remaining limits](docs/FULL-VOCABULARY.md). Device final normalization, the integrated 24-layer model and persistent generation remain unfinished.

The **193 × 210 model layout now passes actual SDK initialization/reset/prepare** after independent static checks. Only 12 matrix blocks have original weights; no model computation was invoked. [Boot evidence and source boundaries](docs/MODEL-BOOT.md).

Eight published static model-weight partitions independently qualify original blocks **0–7679 of 34552**, including whole-ELF compatibility. They are not a fully assembled or executed model. [Partition evidence and tools](docs/MODEL-PARTITIONS.md).

The full 24-layer single-token **CPU reference** now passes 459 independent checks. Expanded static symbol checks and source-reviewed full-model runtime tools are also published, with their evidence levels kept separate. [Validation preparation and limits](docs/MODEL-VALIDATION-PREPARATION.md). These do not establish full-model SDK inference.

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

**Tagged streaming output on a 193 × 50 PE grid** now passes actual SDK transport and independent payload/completion checks. This uses diagnostic integer patterns, not model logits. [Accepted scope, failure history and reproduction](docs/STREAM-OUTPUT.md).

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
