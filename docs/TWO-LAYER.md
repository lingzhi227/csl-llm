# Original-weight two-layer device chain

This page retains the original position-zero acceptance. A subsequent [three-call cached/reset milestone](TWO-LAYER-CACHED.md) extends the bounded two-layer scope.

Two original-sized Qwen decoder layers now pass actual SDK execution in the chain `device endpoint → layer 0 → layer 1 → device endpoint`. Activations travel directly between device regions. The host supplies the initial embedding for token 151644 at causal position 0 and reads the final output and diagnostics. This is a single position-zero integration test, not a complete LLM or cached autoregressive sequence.

## Accepted runtime and numerical evidence

`decoder-chain-runtime-20260908T232256520820Z` is independently accepted against original-BF16 float64 equations and actual stopped-core evidence. All 35 SDK numerical checks pass, with maximum absolute error 1.499385465e-5, relative L2 1.654009143e-6 and relative peak 1.885599050e-6. The separate CPU reference had 54 independent equation checks; those CPU checks must not be counted as additional SDK executions.

The independent core audit checks 57 diagnostic arrays, all 2580 progress markers, 2579 identity coordinates, 205 protocol coordinates, all 2088 original matrix blocks, both original normalization/frequency controllers, 18 bias roots and 264 KV tiles. Ordinary final-output transfer matches the actual stopped-core endpoint output, which matches layer 1 output bit for bit. All tracked processes were absent after completion. The public acceptance record binds both independent numerical and core reviews.

Total execution time was 801.778596 seconds with peak process-tree RSS 1660820 KiB. The endpoint interval was 743855 simulator cycles. Layer-0 and layer-1 intervals were 374983 and 743759 cycles; these include waiting and must not be added as sequential isolated kernel costs. None of these measurements establishes hardware throughput.

The test does not establish cached second-token behavior, reset isolation, nonzero-position RoPE, 24-layer inference, final vocabulary logits, generation or 2048-context/256-generation capacity. Earlier component qualifications and later authoring do not extend this run's scope.

## Static prerequisites

The complete two-layer layout is 129 × 20 application PEs (2580 coordinates), with two original decoder regions and a diagnostic endpoint. Three independently inspected compiler batches cover matrix intervals 0–1023, 1024–2047 and 2048–2087. The completed assembly `decoder-chain-assembled-20260908T231810298397Z` contains all 2088 original matrix blocks, two norm/frequency controllers and 18 bias roots, with 492 common coordinates.

Independent assembly checks cover 2226 intact application ELF files and 26916 logical-symbol coordinate checks. Assembly took 105.360070 seconds and 288316 KiB peak RSS. Per-batch initializer/resource reviews and assembly contract/initializer/symbol reviews remain separate. This is experimental composition of unchanged compiler artifacts, not a claim of an officially supported SDK linker. `release/TWO-LAYER-LEDGER.json` binds the accepted reference, batches, assembly and runtime.

## Selected code and reproduction

- `csl/programs/two_layer/` contains ten exact weight-free device modules, including the endpoint and decoder, matched to the accepted compiler manifest and reviewed integration source.
- `tools/decoder_chain_assembly_probe.py` prepares assembly from locally completed batch audits; `tools/decoder_chain_runtime_probe.py` prepares the single-position SDK test from a completed assembly and the exact reference fixture.
- `support/two_layer/` isolates the frozen runtime/assembly helpers, layout definitions, precision policy and nine host rejection tests. It does not import evolving development sources.
- `validation/frozen/` retains inspection snapshots and explicitly derived compact reports; `validation/reviews/` retains independent original acceptance records.

The public entry points default to the isolated support root; the SDK executor changes only path selection through `CSL_LLM_SDK_IMAGE` and `CSL_LLM_CS_PYTHON`. Device and helper logic are unchanged. Publication checks inspect the already produced outputs and do not rerun SDK simulation. Synthetic host tests validate rejection behavior and are not SDK numerical evidence.

These tools require locally reconstructed original static declarations, completed batch compiler/audit artifacts, and the exact manifest-bound reference fixture. The runtime deliberately rejects a different reference identity. Create a writable project evidence directory and consult `--help` for required inputs. The entire static-input/reference preparation chain is not presented as a turnkey portable model runner.

Weights in CSL literals, original arrays, compiler/SDK binaries, cores and host-specific artifact paths are omitted. Compact summaries preserve original report hashes; original manifests still enumerate omitted files. Public inspection snapshots therefore cannot be executed without reconstructing their dependencies. The complete 122-file integration source manifest was reverified locally; selected actual runtime and assembly dependencies were also reverified on the evidence host. Historical reviews referring to the source before runtime acceptance remain unchanged.

## Separate cumulative model-partition support

The third and fourth full-model partitions now extend independently accepted original block coverage to 0–3583 of 34552. They retain initializer/resource/whole-ELF compatibility acceptance only; no complete full-model assembly or inference is inferred from the two-layer result. The fixed public ledger records all four intervals; prior two-partition reports remain historical records.

The independently accepted `model-symbols-20260908T214013350071Z` checks 312542 logical symbol/coordinate bindings across all 40530 full-layout coordinates and 1401 ELF files, including 337 observed private aliases. It took 32.635148 seconds and 117032 KiB peak RSS. Its original helper source and compact evidence are included for inspection. This is static symbol qualification for those artifacts, not a new actual-core reader execution or permission to transfer earlier boot acceptance to new artifacts.
