# Publication checks — 2026-09-08

- Four host tests passed on Linux in an isolated temporary directory (4.049 seconds), including escaped-child cleanup and unrelated-process identity protection. On macOS, three passed and the Linux-only test was skipped.
- Selected Python files compile successfully. Reader Markdown links and public payload checks passed.
- The selected payload excludes model weights, NPZ activation/weight slices, SDK binaries, ELF/core files, personal workstation paths and private session instructions.
- Public SDK image/wrapper locations use `CSL_LLM_SDK_IMAGE` and `CSL_LLM_CS_PYTHON`. Reference preparation uses the documented local `evidence/reference-f32` directory. These are host portability edits; historical review hashes remain unchanged.
- The regional line collective is selected from the accepted frozen regional-linear run, excluding subsequent unqualified attention extensions. [KERNEL-MATCHES.json](KERNEL-MATCHES.json) records exact matches between published modules and selected original SDK inputs.

No fresh SDK simulations or full-model runs were launched for this publication. Historical reviews retain their bounded scope and do not certify a complete model or every published parameter combination. Original raw execution bundles are not distributed; new runs must generate and verify their own evidence.

`SOURCE-SNAPSHOT.json` identifies selected original inputs before publication edits. `MANIFEST.json` hashes every published file other than itself. `verify_manifest.py` checks that payload independently of the SDK.
