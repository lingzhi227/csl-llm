# CSL-LLM

CSL kernels and distributed execution components for real-model inference on Cerebras WSE-3, developed with SDK 2.10.1. The target is **Qwen2.5-0.5B-Instruct**, using official weights, all 24 layers and the full vocabulary.

**This is an early research release of validated components, not a working end-to-end LLM runtime.** Full decoder layers, complete attention-stage acceptance and complete-model generation remain under development. The project builds on Pragma HLS experience; this release contains handwritten CSL and Python planning/validation tools, not a complete HLS model compiler.

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

The full layer-0 **UP 4864×896 projection** also passes four input cases on 304 PEs. This is a complete single projection, not a complete MLP. The published tool selects the accepted u16-transfer snapshot; later transport optimizations are not included.

The complete layer-0 **GATE 4864×896 projection** now also passes four cases on 304 PEs using paired u32 BF16 transfer and column readback. Its separate entry point preserves the historical UP/u16 qualification. Different UP/GATE runs do not establish a transport speedup ratio.

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
