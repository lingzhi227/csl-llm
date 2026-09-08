# Tagged streaming output acceptance

Actual SDK transport passes on the 193 × 50 application PE grid (9650 PEs). This diagnostic generates tagged integer patterns on device; it does not execute neural kernels or export computed model logits.

## Accepted runs

| Run suffix | Application grid | Exports / tags | Total wall time | Peak process-tree RSS |
| --- | --- | --- | --- | --- |
| 154051566254Z | 17 × 2 | 6 / 2 | 9.183840 s | 368972 KiB |
| 154513964476Z | 193 × 50 | 4 / 2 | 581.535405 s | 4834956 KiB |

The full-grid run selects column 0, column 184, then column 0 again under tag 1; tag 2 resets and exports column 184. Each source emits 128 words. Column 0 contributes 50 rows (6400 words), column 184 contributes 49 rows (6272 words), excluding the padded final-row location. Independent checks verify every word's tag, row, column and position, selected-source completion, repeat/reset, and the completion counters of all 9650 PEs (3 after tag 1, 1 after tag 2). This is selected-column transport coverage on a full grid, not exhaustive export from every column.

Full-grid compilation has six application ELF classes, maximum static high-water 5056 bytes and four final DSR families (6 nodes, 2 checked interference edges). Tiny compilation has five ELF classes, the same maximum static high-water, and five DSR families (8 nodes, 3 edges). All seven/full and six/tiny tracked processes were absent at completion. Host API intervals are not device kernel cycles or hardware bandwidth; dynamic-stack and full-model memory fit remain unproven.

## Failure and qualified repair

The preceding `stream-output-20260908T153144064218Z` failed on the first export: zero bytes received, simulator signal 11, host exit 134. Independent SDK source inspection identified a circular command-stream wait: export held the RPC command while the queued D2H setup needed that command stream to release input backpressure.

The tiny retry changed only the PE source. It releases the RPC command after arming the asynchronous send; the send callback clears ownership and increments progress without a second unblock. Blocking receive, task wait and progress checks verify completion. Driver, layout, configuration and executor are byte-identical to the failed run. Passing retries support this protocol repair; they do not establish the simulator's internal signal-11 root cause. The original failure and pre-run candidate review remain immutable, separately labeled records.

## Selected source and reproduction

`tools/stream_output_probe.py` and `csl/programs/stream_output/pe.csl` preserve accepted logic; the public driver selects the existing portable executor. For preparation use `--prepare`, optionally `--full-fabric`, with the desired `--project-root` and `--source-root`; create the destination project's `evidence` directory first. The preparer expects a local `sdk-examples/tutorials/gemv-09-streaming` checkout beside the project and records four example hashes; accepted hashes are retained in the configuration summaries. Own SDK installation is required; use `CSL_LLM_SDK_IMAGE` and `CSL_LLM_CS_PYTHON` as documented for other probes.

Frozen snapshots are for inspection: host-specific config paths and execution commands are replaced by explicitly derived summaries, with original hashes retained. SDK binaries and original output arrays are omitted. Recreated fixtures have new manifests and require fresh local validation; publication checks do not claim a new SDK run.

## Separate compile-only support

`compact-vocabulary-compile-20260908T151839929786Z` qualifies original static tiles 1024 through 2047 on the full vocabulary layout. Exactly 1024 tiles contain original checkpoint data; 8472 remaining matrix tiles contain zeros. Compilation took 559.127585 s and 9695112 KiB peak RSS, with 1054 ELF classes and maximum static high-water 38768 bytes. Independent resource checks cover 1036 final DSR families, 19636 nodes and 118806 interference edges.

`tools/compact_vocabulary_interval_probe.py` retains that frozen interval generator with public encoder/executor imports. It requires locally reconstructed manifest-bound parent fixtures and model payloads, as do the earlier partial-weight probes. No weights or binaries are distributed. This is neither an execution result nor proof that all 9496 real matrix tiles are resident together. Full vocabulary and S0–S6 model acceptance remain open.
