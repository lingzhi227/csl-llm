# Two-layer cached token and reset acceptance

The original-weight device chain `endpoint → layer 0 → layer 1 → endpoint` passes three calls in one persistent SDK 2.10.1 instance: position 0, cached position 1, then reset and position 0 again. The host supplies original token embeddings at the diagnostic endpoint; interlayer activations and neural computation remain on the device. The application uses 129 × 20 PEs and all 2088 original matrix blocks, two normalization/frequency controllers and 18 bias roots.

## Evidence and boundaries

The accepted run is `decoder-chain-cached-runtime-20260908T235719778700Z`. [Acceptance](../validation/reviews/decoder-chain-cached-runtime-20260908T235719778700Z-acceptance.json) binds the independent [numerical review](../validation/reviews/decoder-chain-cached-runtime-20260908T235719778700Z-independent-numeric-review.json) and [actual-core review](../validation/reviews/decoder-chain-cached-runtime-20260908T235719778700Z-independent-core-review.json).

All 52 independent original-BF16 float64 numerical checks pass: maximum absolute error 1.900818234e-5, relative L2 4.867778062e-6 and relative peak 8.095302587e-6. These independent metrics use a different reference from the owner's frozen-reference comparison in results.json; they must not be combined or relabeled. The separate cached CPU reference review is not an SDK execution.

Normal device transfers capture all three calls. Endpoint and both layer outputs after reset exactly repeat the first call, bit for bit. The final actual stopped-core audit covers 57 arrays, 2580 progress markers, 2579 identities, 205 protocol coordinates, all original initializers and 264 KV tiles. Logical reset does not erase physical slot 1: its retained values are separately checked, while untouched slots remain zero. The final core is the third call's state, not three separate core snapshots.

Execution took 2660.681181 seconds with peak process-tree RSS 1661068 KiB; all three tracked processes were absent after completion. Endpoint intervals are 743855, 746665 and 743855 simulator cycles. Layer-controller intervals include upstream waiting and must not be summed or presented as hardware throughput.

This qualifies the bounded two-layer cached/reset sequence only. Full 24-layer inference, integrated final normalization/vocabulary, persistent text generation, forced packet arrival ordering and 2048-context/256-generation capacity remain unaccepted. Physical hardware validation is separate.

## Published source and reproduction

- [Cached runtime entry point](../tools/decoder_chain_cached_probe.py) selects isolated [helper dependencies](../support/two_layer_cached), preserving the previously accepted position-zero helper versions.
- [Frozen reviewed source](../validation/frozen/decoder-chain-cached-runtime-source-20260908T235700578680Z) retains the exact selected Python/CSL source and original manifest. The original SDK helper is omitted because of host-specific paths; the isolated helper uses the same documented environment overrides as the existing partition executor.
- [Existing two-layer CSL and assembly](TWO-LAYER.md) remain the accepted device artifacts. This cached run reuses the accepted assembly unchanged.
- [Original run metadata](../validation/frozen/decoder-chain-cached-runtime-20260908T235719778700Z) includes the exact driver, precision contract, manifest and results. Execution/config summaries are explicitly derived, retain original SHA-256 values and omit private paths and large tables.
- [Publication checks](../release/TWO-LAYER-CACHED-CHECKS.json) record remote verification of all 17990 runtime manifest entries, all 126 frozen-source entries, every output hash including the actual core, and local normal/core reset checks.

These frozen snapshots are for inspection, not turnkey replay bundles. Original manifests also name intentionally omitted model-bearing initializers, raw tensors, binaries and assembled fixtures. Obtain the pinned model and SDK separately and reconstruct the hash-bound original assembly/reference prerequisites described by the driver. The source snapshot includes additional tools whose presence does not extend this runtime's acceptance.

Set `CSL_LLM_SDK_IMAGE` and `CSL_LLM_CS_PYTHON` as described in [reproduction](REPRODUCING.md). Inspect `python3 tools/decoder_chain_cached_probe.py --help` before preparing new local evidence. Published host-path adaptations have not received a fresh SDK run. Host validation: all 12 tests under `support/two_layer_cached/tests` pass.
