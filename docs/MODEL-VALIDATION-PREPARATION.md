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


## Lifecycle source successor — 2026-09-09

The [new frozen source](../validation/frozen/model-runtime-lifecycle-source-20260909T022128060374Z) supersedes the earlier 011841 source for future single-token runtime preparation. The earlier snapshot and all original hashes remain unchanged. A reproduced fault showed that failure while logging the stop stage could skip the actual SDK stop call. The shared lifecycle helper now attempts stop logging, actual runner stop, every reference close and evidence writing independently, preserving the original exception and recording cleanup failures. Single-token and generation bodies share this helper.

[Shared lifecycle review](../validation/reviews/s5-sdk-lifecycle-shared-source-review.json) binds the final source hashes. The [earlier generation finding and intermediate resolution](../validation/reviews/s5-model-generation-runtime-source-review.json) retain their historical hashes; the later shared review identifies the final frozen implementation. The [snapshot review](../validation/reviews/model-runtime-lifecycle-source-20260909T022128060374Z-source-snapshot-review.json) binds all 28 files.

The source also adds fixed-request admission, request sequencing and a persistent generation body. These remain source-only. There is still no guarded fixed-prompt executable entry point, complete original assembly, prepared runtime candidate or accepted execution budget. The admission review's 166-step schedule and 896-element boundary checks are host checks, not observed wafer generation.

Publication reran the supplied fault tests: **10 tests and 2 subtests passed**. Because the frozen original SDK helper is intentionally omitted for private paths, this host test used the previously published portable helper via `PYTHONPATH=support/two_layer_cached/tools`; no frozen source was edited. From the repository root, run:

```sh
PYTHONPATH=support/two_layer_cached/tools python3 -m pytest -q validation/frozen/model-runtime-lifecycle-source-20260909T022128060374Z/tests
```

The controller separately records three host fault tests under the SDK image's Python 3.11 interpreter. Neither test set constructs SdkRuntime or executes model arithmetic. [Publication bindings](../release/MODEL-LIFECYCLE-CHECKS.json) explicitly retain these limits.


## Guarded generation source successor — 2026-09-09

The [023236 snapshot](../validation/frozen/model-generation-integration-source-20260909T023236345664Z) adds a guarded fixed-four-request preparation/worker entry point and integration-admission helper. It follows the lifecycle successor above; the statement that no guarded entry point existed applies to the earlier 022128 snapshot. The new owner is authored and source-reviewed, but has not prepared or executed a full-model generation candidate.

Preparation requires accepted actual first-token full-model evidence: matching original assembly, complete 40530-coordinate core/normal checks, six output digests, clean lifecycle/process receipts, and 17 exact single-token source pins. Current process absence and independent acceptance remain dispatch prerequisites. The allowed time-budget range is 1800–86400 seconds with a 24 GiB process-tree limit and 16 threads; this is a validated input constraint, **not a chosen or authorized execution budget**.

The worker binds the assembly, original fixed references and frozen source before the persistent request sequence. Success is emitted only after the runtime object is released and full original-initializer/core/normal/history/protocol checks complete, with an exact output file set and hashes. These are source-level requirements, not observed generation results. There is still no complete original assembly, accepted full-model first-token SDK run, generation candidate or selected generation budget.

[Independent guard review](../validation/reviews/s5-model-generation-guard-source-review.json) reports 6 admission/guard tests and 7 subtests. [Snapshot review](../validation/reviews/model-generation-integration-source-20260909T023236345664Z-source-snapshot-review.json) binds all 32 files and confirms unchanged 17 single-token code pins. Publication reverified all 32 files locally/remotely and ran the complete included suite: **16 tests and 9 subtests passed**. Tests used the existing public portable SDK helper through `PYTHONPATH=support/two_layer_cached/tools`; the private-path original helper remains omitted and no frozen source was edited. No SdkRuntime or CPU neural calculation was run.

[Publication bindings](../release/MODEL-GENERATION-GUARD-CHECKS.json) record these checks. Original 011841 and 022128 snapshots and historical review hashes remain intact. All snapshots are inspection material requiring separately obtained SDK/model artifacts and reconstructed evidence; the presence of preparation code does not establish turnkey generation.
