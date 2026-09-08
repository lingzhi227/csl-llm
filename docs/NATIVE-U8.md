# Native-u8 static initializer and supporting compiler checks

One original128×112 BF16 tile passed actual SDK load/run with a native-u8 byte-string initializer. No runtime weight H2D was used. The complete7168 raw u32 words were read before and after four GEMVs and matched the original checkpoint. Canonical/negative/zero/repeat outputs passed independent `math.fsum` checks (canonical relative L2 about1.502e-7, peak1.653e-7); zero and repeat are exact. Each measured controller interval is19184cycles.

The accepted run took20.388707seconds and241424KiB peak process-tree RSS. Independent checks bind the actual weight symbol at3348,28672bytes,4-byte alignment and loaded coordinate(4,1). Static SRAM34864bytes; one DSR family,6nodes,12edges. This is one tile, not full-vocabulary or full-model initialization/inference.

## Source and reproduction boundary

The accepted frozen driver/weight-free PE/kernel and small reports are included. `tools/native_u8_runtime_probe.py` exposes `--from-fixture`, `--worker` and `--execute`; its isolated executor retains the accepted resource limits and uses the documented SDK environment variables. Supporting `native_u8_compile_probe.py` and `compact_words_compile_probe.py` retain the original fixture-generation logic. Use `--help` for their arguments.

These generators require locally reconstructed parent compact/vocabulary fixtures, including original manifest-bound weights and input traces. Those fixtures are not distributed and this update does not provide a turnkey replacement for their entire preparation chain. A new fixture must preserve checkpoint identity and its own manifest; do not substitute historical hashes or label an unverified reconstruction accepted. Frozen directories are for inspection, not direct execution.

`weights.csl` byte strings, expected.bin and all other generated checkpoint payloads are excluded, regardless of extension. No NPY/NPZ, ELF or SDK binaries are public.

## Separately scoped supporting evidence

- The135146 byte-string/comptime-u32 initializer and141727 native-u8 initializer passed compilation and exact loaded ELF-byte checks. Those earlier runs did not execute neural arithmetic.
- The131131 streamed single-layer program passed compilation/static resources only. Its frozen source under `validation/frozen/` is an inspection snapshot, not a layer inference pass.
- Earlier134933 aggregate-bitcast compilation failed; later success does not rewrite it.
- Full-vocabulary124830 was explicitly stopped early after2605.708seconds without completing a full weight block; it was not a timeout or accepted numerical execution. The full151936-vocabulary objective remains open.

All these checks are supporting milestones. Complete24-layer/full-vocabulary generation and full-model capacity acceptance remain pending.
