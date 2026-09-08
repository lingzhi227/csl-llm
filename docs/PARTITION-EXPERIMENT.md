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

The earlier145950 same-process runtime exited134. Root cause remains unknown: the accepted retry changed both process isolation and1D buffers. Do not infer which change fixed it. The150732 eight-PE collective remains outside this release's accepted scope. Full-model acceptance remains open.
