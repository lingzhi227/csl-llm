# Full-model validation preparation

This publication separates three kinds of evidence. None is a completed full-model SDK inference run.

| Item | Accepted scope | Not established |
| --- | --- | --- |
| Full-model CPU reference | One raw token, all 24 original layers, final normalization and the complete 151936-token head; independent numerical validation | SDK inference, multi-token prompts, cached/reset generation, long context |
| Expanded static symbol contract | Actual ELF symbol address/size/alignment and alias checks over the full layout | Live/core reader execution, complete original-weight deployment, model inference |
| Future runtime integration source | Frozen source review of assembly, execution and diagnostics | Complete assembly, prepared runtime candidate, chosen run budget or SDK completion |

## Independently accepted CPU reference

Run `model-single-token-reference-20260909T010221089813Z` was produced by `model-reference-build-20260909T005940695866Z`. Input token **151644** produces winner **72030**, matching the official one-token greedy generation result. The original checkpoint revision remains `7ae557604adf67be50417f59c2c2f167def9a775`.

The official FP32 reference is checked against independently propagated NumPy float64 equations using raw original BF16 checkpoint values across all 24 layers and the tied full head. All **459 independent checks** pass. The owner's separate 100 reference checks must not be counted as SDK tests or added to the independent check count. [Independent review](../validation/reviews/model-single-token-reference-20260909T010221089813Z-independent-reference-review.json) retains individual errors, method and immutable source/output hashes; [reference report](../validation/frozen/model-single-token-reference-20260909T010221089813Z/report.json) retains the original official-reference configuration and outcome.

CPU execution took 15.248867 seconds with peak process-tree RSS 3496332 KiB, and tracked processes were absent afterward. These are CPU reference-construction measurements, not wafer inference performance. No CPU model calculation was rerun for publication; the frozen trace hash and existing review/build bindings were verified.

## Expanded static symbols

`model-symbols-20260909T004647566849Z` independently checks 1401 application ELF classes, 40530 coordinates, 408684 logical-symbol/coordinate combinations and 374 private aliases across images. The [independent review](../validation/reviews/model-symbols-20260909T004647566849Z-independent-symbol-review.json) verifies actual object tables and coordinate ownership. The offline run took 40.969613 seconds and 121308 KiB peak RSS. This is a static symbol contract for those frozen artifacts; it does not qualify an expanded actual-core reader or a complete original-weight model runtime.

## Source-only integration

The [frozen integration source](../validation/frozen/model-runtime-integration-source-20260909T011841165105Z) contains the future full-model assembly/runtime drivers, normal-I/O and stage validators, original-initializer audit support, and [diagnostic design](../validation/frozen/model-runtime-integration-source-20260909T011841165105Z/docs/FULL-MODEL-RUNTIME-VALIDATION.md). Its [source snapshot review](../validation/reviews/s5-full-model-runtime-integration-source-snapshot-review.json) binds all 21 original source files. The runtime/assembly driver review and 312-stage source review remain separate records under `validation/reviews`.

The diagnostic design checks every layer's intermediate stages, normal outputs and original model initializers. Its request-history rules distinguish logical cache reset from retained physical SRAM. These are reviewed future checks, not observed full-model runtime results. At the frozen review there was no complete original assembly, no prepared full-model runtime candidate and no selected execution time budget. Do not infer completion from the presence of the driver.

## Source and reproduction boundaries

[Publication checks](../release/MODEL-PREPARATION-CHECKS.json) bind six complete remote manifests and the exact selected local sources. Original manifests and reports are immutable; compact execution/config/result summaries explicitly identify their source SHA-256 and omitted private paths/tables.

The source snapshots are inspection material, not turnkey execution bundles. Each intentionally omits the original SDK helper containing private host paths, documented in the publication check record. The existing public `tools/partition_executor.py` demonstrates the portable SDK environment overrides, but substituting it would create a new local source identity requiring fresh verification. Raw tensors, checkpoint data, ELF/core/SDK binaries and the upstream Transformers source copy are omitted; the original manifest preserves their hashes. Obtain licensed upstream dependencies separately using the pinned versions in the report and [source notices](../THIRD_PARTY_NOTICES.md).

The latest actual neural SDK integration remains the [bounded two-layer cached/reset sequence](TWO-LAYER-CACHED.md). Complete 24-layer SDK inference, continuous generation, full capacity and hardware validation remain open.
