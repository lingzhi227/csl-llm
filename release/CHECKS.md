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
