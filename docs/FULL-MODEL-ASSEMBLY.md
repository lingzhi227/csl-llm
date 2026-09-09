# Complete original-model static assembly

All 34552 original matrix blocks are now assembled into one complete 193 × 210 application layout (40530 PE coordinates). The result contains 34714 intact application ELF files, with 5978 common coordinates. Both complete static composition and the resulting assembly's own diagnostic symbol contract are independently accepted. **The complete model has not yet received SDK loading, neural inference, generation or capacity acceptance.**

## Accepted evidence

Run `model-assembled-20260909T102914567058Z` took 1430.660860 seconds with peak process-tree RSS 5545356 KiB; all three tracked processes were absent after completion. It selects intact compiler artifacts without patching their bytes. This remains experimental composition, not a claim of an officially supported SDK linker.

The [assembly acceptance](../validation/reviews/s5-fullmodel-assembly-acceptance.json) binds the independent contract and symbol reviews. The contract audit checks exact source/output ELF equality, full original partition coverage, common allocated bytes, ordered loader segments, layout and SDK I/O/RPC metadata. Original weight correctness is inherited from independently accepted partition initializer audits plus exact copied ELF bytes. It is not a new device-memory readback.

The separate [actual assembly symbol review](../validation/reviews/model-assembled-20260909T102914567058Z-independent-symbol-review.json) checks 408684 logical-symbol/coordinate combinations and 577 private aliases across the assembled images. These are actual static ELF address/size/alignment/alias checks for this assembled artifact, not a live or stopped-core reader test. Earlier partial-layout symbol audits do not substitute for this new audit.

## Selected tools and provenance

The [public assembly entry point](../tools/model_assembly_probe.py) uses isolated [frozen helper dependencies](../support/model_assembly). The driver requires a complete collection of locally available accepted audit fixtures; missing/repeated/incomplete collections are rejected during preparation. Use `python3 tools/model_assembly_probe.py --help` to inspect the interface. Obtain the pinned SDK, model and compiled/audited fixtures separately; public source does not include those artifacts.

[Compact frozen metadata](../validation/frozen/model-assembled-20260909T102914567058Z) retains the original driver, selected generic source and explicitly derived execution/config/result/symbol/manifest summaries. The enormous complete manifest and detailed result/symbol tables are intentionally not copied into the public package. Every summary identifies the original file SHA-256, and selected source hashes retain their original meanings. No weights, generated initializer modules, ELF/core binaries or complete experiment trees are published.

The only public execution-helper adaptations use the existing `CSL_LLM_SDK_IMAGE` and `CSL_LLM_CS_PYTHON` overrides and select the isolated support root. These portable adaptations have not received a fresh assembly or SDK qualification. [Publication checks](../release/FULL-MODEL-ASSEMBLY-CHECKS.json) record remote complete-manifest verification and selected-source/acceptance bindings. Publication itself did not rerun assembly, compilation, SDK simulation or CPU model inference.

## Remaining runtime gates

Next are preparation and actual execution of the full 24-layer first-token SDK test, independent original-input layer/stage/final-logit verification, then persistent fixed-request generation, reset/EOS and target capacity. Source-only runtime tools are described in [validation preparation](MODEL-VALIDATION-PREPARATION.md). The accepted [two-layer cached/reset test](TWO-LAYER-CACHED.md) remains the latest bounded neural integration evidence; it does not qualify this complete model assembly. Hardware validation remains separate.
