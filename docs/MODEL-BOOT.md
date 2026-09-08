# Full model layout: static checks and SDK boot

The 193 × 210 application layout (40530 PEs) has passed compilation, independent static initializer/resource inspection and one actual SDK initialization/reset/prepare run. **No model compute was invoked.** Of 34552 matrix blocks, only 12 contain original data; the remaining matrix blocks are zero. All 25 normalization controllers (including final normalization), frequencies on 24 decoder controllers and three selected bias roots have original data. This is not full original-weight deployment or neural inference.

## Accepted evidence

| Stage | Frozen run | Wall time | Peak process-tree RSS |
| --- | --- | ---: | ---: |
| Compilation | model-compile-20260908T182818553172Z | 133.591230 s | 1364424 KiB |
| Offline initializer/resource audit | model-elf-audit-20260908T194004676554Z | 130.470166 s | 105348 KiB |
| Actual SDK boot | model-boot-20260908T195258420390Z | 1094.791799 s | 22337684 KiB |

Static inspection covers all 40530 coordinates, 388 application ELF classes, original selected data and zero placeholders with four-byte alignment. Maximum static high-water is 45456 bytes; 210 final DSR families contain 2818 nodes and 12362 checked interference edges. These bounds do not prove dynamic-stack use or fully initialized model capacity.

The SDK run loaded the layout, initialized coordinate metadata, supplied prompt/control metadata, reset the model and invoked prepare. All 40530 progress values equal 1, decoder logical status is zero, and all coordinate identities and selected original/zero initializers match the independent expectations. Ordinary controller progress/control/summary transfer matches actual stopped-core values; prompt metadata is checked through ordinary transfer. The actual core is 2251540088 bytes and is independently parsed as runtime memory, with compiled ELF used only for symbol addresses and coordinates. All three tracked processes were absent at completion.

The layout includes 3168 KV tiles. Reset logical status does not qualify physical KV capacity, context-length behavior or persistent generation. Simulator host memory and wall time are not hardware utilization or throughput. Full original model deployment, neural computation across 24 layers and generation remain open. The previously accepted full-vocabulary component remains a separate run with complete original vocabulary weights.

## Selected source and tools

`csl/programs/model_boot/` contains 13 exact, weight-free CSL modules from the accepted compilation. They include computation code, but this milestone qualifies their compilation and boot behavior only. The frozen generated layout is retained under `validation/frozen/model-compile-20260908T182818553172Z/`; static-data files it references are omitted.

`support/model_boot/` isolates the frozen layout/coordinate definitions, initializer auditor, static resource readers and stopped-core reader. This prevents evolving model work from changing older qualified tools. Its own publication manifest is explicitly distinct from the original evidence manifest.

`tools/model_elf_audit_probe.py` prepares an offline audit from locally reconstructed compiler output; `tools/model_boot_probe.py` prepares a boot run from a locally completed audit. Both default to this isolated support package. Use `--help` for required local fixture arguments and create the destination project's evidence directory. SDK paths use `CSL_LLM_SDK_IMAGE` and `CSL_LLM_CS_PYTHON`. The boot preparer selects the portable equivalent executor; support computation logic is unchanged. Publication edits have not received a fresh SDK execution.

The original compile driver is preserved as inspection source only: its evolving model-layout/static-input preparation chain is not presented as a complete portable compiler frontend. Reproduction requires one's own SDK/checkpoint and reconstructed manifest-bound static declarations and compiled artifacts. Frozen snapshots intentionally omit weights, bias/norm/frequency literal modules, raw arrays, ELF and core binaries. Their original manifests enumerate omitted files; snapshots alone are not executable fixtures. Independent review hashes and derived execution summaries preserve provenance without host-specific paths.
