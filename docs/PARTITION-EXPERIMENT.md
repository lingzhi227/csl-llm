# Experimental intact-artifact composition

Two independent original-weight GEMVs passed actual SDK execution in isolated processes, comparing direct compilation with experimental composition of intact compiler-produced ELF artifacts. No ELF bytes were patched. This is an experimental workflow, not a claim of official SDK support.

Both variants used two original128×112 tiles and four input cases. Initial/final raw weight readback, independent original-weight `math.fsum` checks, zero/repeat/progress and direct/composed output bits passed. Maximum relative L2 was1.672e-7 and peak2.3331e-7. Each measured interval was19184cycles. Runtime45.886745s; peak process-tree RSS200824KiB. This does not test application collectives, full vocabulary or model inference.

The static parent145551 was independently checked for exact direct-baseline loader segments, allocated sections, coordinates and I/O/RPC metadata. The runtime manifest recursively binds those compiled artifacts. Compiled artifacts contain weights and are not public.

## Separately accepted compile-only scaling probes

| Probe | Actual checkpoint tiles | Zero-filled matrix positions | Application ELF classes | Compile wall time | Peak tree RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
|143328|256|9240|277|153.062s|2743492KiB|
|144210|1024|8472|1045|552.006s|9696772KiB|

Both were checked across9650 application coordinates and9496matrix weight locations, with28672byte/aligned4 weight symbols and max static38768bytes. This is not proof that all9496 real weight tiles fit/compile/run. Zero-filled placeholders must not be counted as loaded model weights.

## Source and reproduction boundary

`partition_compile_probe.py` and `partition_runtime_probe.py` retain the accepted generator/worker logic with isolated SDK path settings. `elf_partition.py` and `compare_elf_images.py` implement artifact selection and comparison. Use `--help` for required parent/template fixture arguments. The compact256/1024 generators and shared encoder are also included; these are compile-only workflows.

These tools require locally reconstructed, manifest-bound parent fixtures and locally compiled SDK artifacts; the entire input preparation chain is not supplied as a turnkey runner. Source snapshots in `validation/frozen/` are for inspection. Recreating artifacts requires one's own model/SDK and new verification. No weights, binaries or original tensors are distributed.

The earlier145950 same-process runtime exited134. Root cause remains unknown: the accepted retry changed both process isolation and1D buffers. Do not infer which change fixed it. The eight-PE collective is now separately accepted below. Full-model acceptance remains open.

## Eight-PE cross-partition collective acceptance

Runtime `partition-runtime-20260908T151505818904Z` passes the complete 128 × 896 group-0 contraction using eight original weight tiles, with line allreduce across four independently compiled two-PE partitions. Direct and composed variants execute in separate processes, four input cases each. All eight PE outputs pass independent original-weight `math.fsum` checks (maximum relative L2 1.1639505e-7, relative peak 1.4761303e-7). Initial/final weight readback, zero/repeat/progress and direct/composed output bits pass.

Measured simulator intervals are 19622 cycles at root and 19630 elsewhere. Total runtime is 111.095742 s, peak process-tree RSS 204880 KiB; all five tracked PIDs were absent after execution. These are simulator measurements, not hardware performance.

Static parent `partition-compile-20260908T150732487335Z` passed exact direct/composed ordered loader images, I/O and RPC comparison. Its eight application ELF classes have maximum static high-water 35744 bytes; eight final DSR families cover 56 nodes and 144 checked interference edges. Compilation took 21.418434 s and 379428 KiB peak RSS. Static resource analysis does not prove dynamic-stack or full-model fit.

Use `tools/partition_collective_compile_probe.py --collective` and `tools/partition_collective_runtime_probe.py` with the required reconstructed parent fixtures (`--help`). The selected PE/kernel/line sources live in `csl/programs/partition_collective/`. Frozen sources and three independent reviews are retained under `validation/`; `release/COLLECTIVE-CHECKS.json` binds publication checks to the originals. Public edits isolate tool imports and library paths, correct the generator description, and reuse the byte-matched portable executor. No fresh SDK run is claimed for those edits.

Snapshots intentionally omit weight initializers, raw tensors, binaries and host-specific composition/execution files. They are inspection snapshots, not complete executable fixtures. This establishes a bounded collective across experimental intact compiler artifacts, not official SDK composition support, full 151936-token vocabulary execution, or complete model inference. Full S0–S6 acceptance remains open.
