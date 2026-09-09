# Runtime diagnostic exports and first full-model outcome

The first complete original-weight 24-layer SDK attempt reached its six-hour guard without a neural result. Subsequent two-PE experiments establish a narrower capability: the public SDK 2.10.1 `dump_elf_core` call can return a diagnostic file while computation or packet work remains unfinished, and selected saved values match independent numerical expectations.

**A validated diagnostic export is not a successful shutdown, a restart checkpoint or full-model inference.** The two active experiments still have `execution_success: false` because the later `stop()` did not return within the guard. That failure is retained alongside `snapshot_arrays_passed: true`.

## Accepted cases

| Case | Observed diagnostic state | Execution outcome |
| --- | --- | --- |
| Normal compute, 183022 | Both PEs complete 16 xorshift steps; value 3518474805. All 352 exported object bytes equal ordinary device readback. | Success; 2.043 s |
| Active compute, 183511 | Both PEs have issued and committed 907 of 1000000000 steps, done=0, value 1765088451; independent GF(2) oracle agrees. | Stop timeout; 34.559 s including cleanup |
| Normal communication, 183959 | Eight rounds complete; receiver's 31-word nonzero payload agrees with the independent sequence formula. All 352 object bytes equal ordinary readback. | Success; 4.088 s |
| Active communication, 184530 | Both sequence counters are 6 of 1000000000, done=0; receiver's retained sequence-5 payload is exact. A sender software receive-operation flag is set. | Stop timeout; 34.574 s including cleanup |

Each explicit core is 1449042 bytes. The active compute export returned between 1.285 and 1.300 seconds; active communication between 1.272 and 1.286 seconds. The file hashes remained unchanged after the tracked host processes exited. Independent reviewers used the actual compiled object addresses and SDK core reader, checking 352 bytes per case. These small-file costs do not predict full-model export cost.

The communication flag indicates unfinished software operation state. It does not establish physical wavelets in flight. Per-PE numerical agreement does not establish a globally atomic snapshot, saved queue state, simulator restoration, application continuation, full-model neural correctness, long-context capacity or physical-wafer execution.

The [unified acceptance](../validation/reviews/sdk-explicit-core-qualification-acceptance.json) indexes all four exact runtime and review identities. [Publication checks](../release/SDK-CORE-EXPORT-CHECKS.json) record remote manifest verification and hashes without distributing cores, tensors, ELFs or the SDK.

## First full-model experiment

Run `model-runtime-20260909T111353596920Z` used the [accepted complete static assembly](FULL-MODEL-ASSEMBLY.md). Its guard stopped it after 21604.825875 seconds including cleanup. Peak resident memory was 21708104 KiB, below the 24 GiB limit; this was a time-budget failure. The independently reviewed receipt contains 20997 resource samples and confirms all three tracked processes exited.

The only retained device array was `ready`, shape [210,193,1], with every entry equal to one. There was no final results report, neural output array, protocol report or `out.core`. This confirms readiness only. It cannot establish which layers completed correctly, why execution failed to finish, or any recoverable checkpoint. No first output token was verified.

The [derived outcome review](../validation/reviews/model-runtime-20260909T111353596920Z-independent-outcome-review-summary.json) retains the original review hash while replacing private paths. The [execution summary](../validation/frozen/model-runtime-20260909T111353596920Z/execution-summary.json) retains `success: false`. Prior layer-0, cached two-layer and isolated full-vocabulary acceptances are unaffected.

## Diagnostic source organization

- [Isolated support directory](../support/sdk_core_export/README.md): exact generic export driver, device sources and helpers, except the two documented SDK-path substitutions and derived provenance.
- [Frozen export source](../validation/frozen/sdk-core-export-source-20260909T183022454936Z/manifest-summary.json): eight original source identities; the private-path SDK helper is omitted here.
- [Frozen compile source](../validation/frozen/sdk-stop-compile-20260909T180325524415Z/manifest-summary.json): the earlier compiler driver and identical two-PE device sources.
- [Loop review](../validation/reviews/sdk-stop-compile180325-loop-review.json): independent inspection of retained device loop and store ordering. Compiled binaries/disassembly are not distributed.

The exporter reuses the accepted compile bundle with manifest `d19bd241da4aab2180525dea9b25d99d9a67863ec9e8e049451e6bb1f2301f3d`. Its admission guard remains exact. The publication does not replace that identity with a rebuilt artifact or claim the portable copy was rerun.

## Reproduction boundary

The frozen folders are historical inspection material, not self-contained runtime bundles. Core files, compiled images, normal arrays and the original private-path helper are intentionally omitted. Reading the Python/CSL sources and compact receipts requires no model download.

Reproducing the experiment needs Linux, an independently installed matching SDK and compiled artifacts with a newly reviewed provenance chain. The preserved exporter requires the exact historical compile identity; a fresh compilation or path change does not automatically satisfy it. Do not disable the pin to claim reproduction. The driver's `--inspect` path additionally requires the recorded host PID namespace, completed process-identity checks and the original core/output files; a container with a separate PID namespace cannot prove host process exit.

In the isolated helper, `CSL_LLM_SDK_IMAGE` and `CSL_LLM_CS_PYTHON` replace the original absolute SDK image and launcher paths. These are publication portability edits only. No SDK experiment was launched for publication.

## Failure and review history

Earlier active-compute and active-communication `stop()` tests timed out after approximately 124 seconds including cleanup and produced no core. A prior affine compute candidate had already finished before stopping; it did not qualify active-work preservation. The revised xorshift device loop was independently checked before the four explicit-export cases.

An [additional host PID namespace audit](../validation/reviews/sdk-stop-host-pid-namespace-audit.json) corrects the process-exit evidence for earlier small reviews that ran inside a separate container PID namespace. Original numerical/hash evidence is unchanged. The new explicit-core reviews directly check host process identities.

Next work is to qualify an explicit diagnostic export on the accepted full-model artifacts under reviewed resource limits, then inspect demonstrably completed layer states. This is future work; the small diagnostic acceptance does not authorize a claim of full-model continuation or successful inference.
