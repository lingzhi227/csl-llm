# Publication checks — 2026-09-08

- Four host tests passed on Linux in an isolated temporary directory (4.049 seconds), including escaped-child cleanup and unrelated-process identity protection. On macOS, three passed and the Linux-only test was skipped.
- Selected Python files compile successfully. Reader Markdown links and public payload checks passed.
- The selected payload excludes model weights, NPZ activation/weight slices, SDK binaries, ELF/core files, personal workstation paths and private session instructions.
- Public SDK image/wrapper locations use `CSL_LLM_SDK_IMAGE` and `CSL_LLM_CS_PYTHON`. Reference preparation uses the documented local `evidence/reference-f32` directory. These are host portability edits; historical review hashes remain unchanged.
- The regional line collective is selected from the accepted frozen regional-linear run, excluding subsequent unqualified attention extensions. [KERNEL-MATCHES.json](KERNEL-MATCHES.json) records exact matches between published modules and selected original SDK inputs.

No fresh SDK simulations or full-model runs were launched for this publication. Historical reviews retain their bounded scope and do not certify a complete model or every published parameter combination. Original raw execution bundles are not distributed; new runs must generate and verify their own evidence.

`SOURCE-SNAPSHOT.json` identifies selected original inputs before publication edits. `MANIFEST.json` hashes every published file other than itself. `verify_manifest.py` checks that payload independently of the SDK.

## Stateful-component milestone update

The three new SDK run manifests match every original listed file, successful execution/result records match each other, and their hashes match the independent coordinator reviews. Fifteen locally retained raw output/cache NPZ files also match review hashes (see `RAW-OUTPUT-CHECKS-20260908.json`). This verifies recorded artifact identity, not a new numerical rerun.

Frozen CSL/driver inspection snapshots, small results/config/manifests and derived execution summaries are included. Public preparation tools use the accepted frozen driver and exact protocol dependencies, with explicit publication-only path selection. The earlier capacity preparation entry now has its accepted frozen implementation and required KV/collective dependencies. Earlier regional sum-only code remains unchanged.

After the update, all four host tests passed on Linux in 4.045 seconds. Python syntax, documentation links and public payload checks passed. No new SDK simulation was launched for publishing. Full-model acceptance remains open.

## Full native UP milestone

Verified every file in the accepted projection-up manifest, all four actual output hashes, independent review hashes and successful execution/results. Published the exact frozen kernel, collective and region helper; the public driver changes only reference/dependency paths and limits its selector to accepted UP. The newer u32/batched-read implementation is excluded. Python syntax and CLI checks pass; local host regression passes three tests with the Linux-only cleanup test skipped. Public payload and reader links checked; no new SDK simulation. See NATIVE-UP-CHECKS.json.

## Full native GATE milestone

Verified frozen GATE and paired-u32 selection manifests, successful execution/result identities, independent review hashes, and 11 actual output files against results. Shared kernel/region/collective and transport dependencies match exact frozen bytes. UP/u16 is preserved; GATE/u32 has its own preparation entry. Six local host tests pass, including BF16 bit-pattern/byte-order/shape tests; one Linux-only test is skipped. These host pattern tests do not constitute exhaustive SDK transport testing. Python syntax, CLI, reader links and public payload checks pass. No new SDK execution or ELF/DSR qualification was performed.

## Full native DOWN milestone

Verified all frozen manifest inputs, successful execution/results and independent numerical review hashes, four actual output hashes and the independently reviewed ELF report identity. Shared code dependencies match accepted frozen bytes. The new DOWN entry adapts paths/selector only. Six host tests pass with one Linux-only skip; Python syntax/CLI, payload and links pass. No new SDK run or publication-side ELF/DSR audit. Inputs remain host-prepared fixtures; separate UP/GATE/DOWN passes do not qualify a connected device MLP. See NATIVE-DOWN-CHECKS.json.

## Complete resident MLP milestone

All original manifests were verified for the accepted static MLP, packet fan-in, filtered-input and one-PE static-weight experiments, together with successful execution/results and matching independent reviews. Retained actual-output hashes match the results. Static MLP resource/initializer report hashes match their independent reviews. Publication did not rerun numerical or SDK experiments.

Published drivers select their own exact frozen libraries and region helper. Regenerating the full916-tile static layout from the portable driver produces byte-identical layout CSL. Six host tests pass, with one Linux-only skip; Python syntax, Markdown links and credential/personal-path checks pass. Weight-bearing generated source was excluded by explicit per-run selection and content inspection:916 weights modules, static1PE pe.csl, NPZ and ELFs are not published. Metadata lists original hashes but not weight values. No complete-layer/model, dynamic-stack or hardware-performance qualification is implied. Original H2D timeout and supporting compile failures remain disclosed.

## MLP DSD-copy performance milestone

Verified every frozen manifest entry, SDK/result/review bindings, same fixture/precision/layout/916weight modules, and directly compared all four actual output containers to baseline bitwise (excluding timing/cycle arrays). Resource report hashes match the independent review. Public generator reproduces the exact frozen static layout; original scalar-copy implementation is preserved. Six host tests pass/one Linux-only skip; syntax, links and source-content scans pass. Generated model-weight sources and binaries remain excluded. No new SDK experiment or hardware speedup claim. See MLP-DSD-CHECKS.json.

## Complete native decoder layer0 milestone

All1068 original manifest entries were verified read-only on the remote immutable decoder run, including generated weights omitted from the local mirror/public package. Mirrored source and metadata hashes, five actual output identities and final execution/results/review bindings also match. All supporting reviews match hashes in the final numerical acceptance. The9case scripted token-controller manifest/results/execution and resource bindings were separately checked.

The portable frozen layer0 generator reproduces the exact accepted layout. Eight host tests pass, including24-layer offline ownership; one Linux-only test is skipped. Python syntax, reader links and public content checks pass. The parameterized driver/source match their independently reviewed hashes but are identified as offline interfaces, not SDK-qualified multilayer execution. Weight/bias/controller-aux source payloads, NPZ and ELFs are excluded. No new SDK or publication-side numerical rerun; no full-model or physical-hardware acceptance claim.

## Native-u8 runtime and compiler-support update

Independently verified remote manifests for all four selected runs (9/7/7/14 files), including weight payloads intentionally absent from the public snapshot. Mirrored source/results/execution/review identities and four actual runtime output hashes match. Selected ELF/DSR metadata hashes match independent resource reviews. Runtime and compile-only outcomes remain separate. Eight host tests pass/oneLinux-only skip; Python syntax, runtime CLI, reader links and source-content scans pass.

New tools retain frozen generator/worker behavior with isolated SDK executor and host-path adaptations. Parent fixture reconstruction remains a documented prerequisite rather than a claimed turnkey reproduction. No SDK rerun; no full-vocabulary inference or large-scale initialization acceptance. Actual byte-string weight modules/expected.bin and all raw tensor/binary payloads are excluded.

## Isolated two-PE artifact-composition milestone

Independently checked all remote recursive manifests for runtime/static composition and256/1024compile probes (31/24/269/1037entries). Mirrored files and review/execution/results identities match; all8actual output hashes match and direct/composed non-timing arrays compare bitwise. Generators/helpers use frozen source with isolated public SDK paths. The256/1024encoder snapshots match. Eight host tests pass/oneLinux-only skip; syntax/runtime CLI, public content and reader links pass. No new SDK experiment or claim of official artifact-composition support. Source-only reproduction prerequisites, zero-placeholder limits, same-process failure134 and unqualified collective/model scope remain explicit.

Eight-PE collective publication: both frozen recursive manifests verified on the original evidence host; review/run hashes, eight actual-output hashes and non-timing array bit equality checked locally. See `COLLECTIVE-CHECKS.json`. No duplicate SDK execution.

Streaming publication: four original manifests checked on the evidence host; ten saved output payloads and four global progress arrays independently checked locally against tags/coordinates/counters and review hashes. Frozen tiny retry differs from failed run only in PE source. See `STREAM-CHECKS.json`; no new simulator run.

Full vocabulary publication: four complete recursive manifests rechecked on the original evidence host (28523 runtime, 38893 assembly, 14 contracted-reader and 17 topology-reader entries); aggregate review hashes and three case result/output identities verified locally. Exact 9496-tile coverage, zero/winner, all progress values and repeat-logit bits checked. Seven weight-free CSL modules match the frozen common batch hashes. See `VOCABULARY-CHECKS.json`; no new SDK run.

Model boot publication: 1113 boot, 272 compile and 1106 audit manifest entries checked on the original evidence host; review/execution/results identities verified. Actual output/core hashes, all 40530 progress markers, decoder zero status and ordinary controller metadata checked. No new SDK run. See `MODEL-BOOT-CHECKS.json`.

Model partition publication: six complete frozen manifests verified on the original evidence host; review/execution/results identities and exact accepted block intervals checked locally. No SDK or compiler rerun. See `MODEL-PARTITION-CHECKS.json`.

Two-layer publication: all selected execution/batch recursive manifests verified on the evidence host; complete 122-file integration source verified locally. Actual core/output hashes, 57 diagnostic arrays, ordinary readiness/progress and endpoint/layer1/final-transfer bit equality checked. Independent original-input/core reviews retained. Nine isolated support tests pass, plus the existing host suite; no new SDK run. See `TWO-LAYER-CHECKS.json`.


## Cached two-layer publication — 2026-09-09

See [cached source/evidence and validation](../docs/TWO-LAYER-CACHED.md). All 12 isolated host tests pass. Existing saved SDK outputs, complete remote manifest bindings and reset bit-repeat were checked without re-running the simulator. Model payloads and private paths are excluded.

Static partitions five and six: complete remote manifests, independent review bindings, local selected source hashes and exact original block intervals verified. No SDK/compiler rerun. See [publication records](MODEL-PARTITION-5-6-CHECKS.json).

Full-model preparation: six complete remote manifests, local source hashes, CPU trace/report/build and independent review bindings verified. Selected Python syntax checked; no new CPU model or SDK run. See [preparation records](MODEL-PREPARATION-CHECKS.json).

Partitions 7–8 and lifecycle successor: six complete static remote manifests plus all 28 frozen lifecycle source files verified. Host fault tests passed (10 tests, 2 subtests) using the existing portable SDK helper; no SDK/model run. See [lifecycle bindings](MODEL-LIFECYCLE-CHECKS.json) and [static bindings](MODEL-PARTITION-7-8-CHECKS.json).

Partitions 9–10 and guarded generation source: six complete static remote manifests and all 32 generation source files verified. Complete included host suite passes 16 tests and 9 subtests using the existing portable helper; no SDK/model run. [Guard publication records](MODEL-GENERATION-GUARD-CHECKS.json).

Partitions 11–12: six complete remote manifests, local selected source, exact original intervals and independent review/execution/results bindings verified. No compiler/SDK/CPU reference rerun. See [publication record](MODEL-PARTITION-11-12-CHECKS.json).

Partitions 13–14: six complete remote manifests, local selected source, exact original intervals and independent review/execution/results bindings verified. No compiler/SDK/CPU reference rerun. See [publication record](MODEL-PARTITION-13-14-CHECKS.json).

Partitions 15–16: six complete remote manifests, local selected source, exact original intervals and independent review/execution/results bindings verified. No compiler/SDK/CPU reference rerun. See [publication record](MODEL-PARTITION-15-16-CHECKS.json).

Partitions 17–18: six complete remote manifests, local selected source, exact original intervals and independent review/execution/results bindings verified. No compiler/SDK/CPU reference rerun. See [publication record](MODEL-PARTITION-17-18-CHECKS.json).
