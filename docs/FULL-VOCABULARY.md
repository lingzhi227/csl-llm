# Complete tied embedding and vocabulary head

The original 151936 × 896 tied embedding/head passes three actual SDK cases on a 193 × 50 application PE grid. This is a complete vocabulary component, not complete Qwen inference. The head input is supplied from the verified reference after final normalization; device final RMS, all 24 layers together, persistent autoregressive generation/reset and full-model capacity remain unaccepted.

## What ran and passed

| Case | Original embedding row | Head input | Winner | Controller interval |
| --- | ---: | --- | ---: | ---: |
| 0 | 0 | Official final-normalization output | 22 | 262444 cycles |
| 1 | 151645 | Zero | 0 | 268359 cycles |
| 2 | 151935 | Repeat of case 0 head input | 22 | 262444 cycles |

All 896 values in each selected embedding row are exact. All 151936 head outputs pass an independently derived original-BF16 float64 reference: maximum absolute error 5.9501414e-6, relative L2 2.5468193e-7, relative peak 2.6342210e-7 for the nonzero cases. The zero case produces only zero logits and selects the smallest index. Cases 0 and 2 produce bit-identical logits in separate fresh processes; their embedding queries differ. This does not qualify repeat/reset in a persistent model process.

The winning token returns through ordinary device-to-host transfer. Complete logits, all 9496 original matrix weight blocks, all 9650 PE identity/control/progress records and lookup data are verified from the actual stopped simulator core, separately checked against saved outputs and ordinary winner transfer. Initial compiled images supply symbol addresses; they are not substituted for actual runtime memory. Full-logit normal streaming and the complete token-generation interface are not qualified by this test.

The full run `vocabulary-runtime-20260908T173751421842Z` completed in 6399.905247 seconds with peak process-tree RSS 5620748 KiB; all six tracked processes were absent at completion. Controller cycle intervals are simulator measurements; total wall time includes loading, transfers and validation, and neither is hardware throughput.

The aggregate acceptance record and its hash-bound final numerical and three independent core reviews are in `validation/reviews/`. Earlier case-0/case-1 receipts correctly retain their original pending-whole-run status; the final acceptance record supersedes that pending state without rewriting history. `sdk_complete` remains false.

## Static assembly and ten compiler batches

All 9496 original weight tiles were compiled across ten independently audited batches. Per-batch unused positions were zeros; the assembled artifact selects the original-weight tile for every matrix coordinate. Thus the assembled program has complete original vocabulary weights, not the zero-filled partial coverage of individual batches.

`vocabulary-assembled-20260908T172933194810Z` passed independent static checks: 9499 unchanged application ELF files, 9650 unique application coordinates, exact loaded original weight bytes, required symbol contracts, common ordered I/O/RPC images and 18995 source allocation-graph copies. Maximum static high-water is 38768 bytes. Assembly took 288.673652 seconds and 1006540 KiB peak RSS, separately from compiling the batches and preparing their snapshots.

The public batch ledger retains all ten bundle identities, exact tile intervals and original evidence hashes. Original initializer/resource reviews remain separate. This workflow composes intact compiler artifacts without modifying ELF bytes; it is experimental and does not claim an officially supported SDK linker. Static high-water does not establish dynamic stack or full-model memory fit.

## Source organization and reproduction prerequisites

- `csl/programs/resident_vocabulary/`: seven exact, weight-free CSL modules from the common accepted batch program (PE, GEMV, embedding, reduction, argmax, input and packets).
- `tools/vocabulary_assemble_probe.py`: validates and composes locally available accepted batch artifacts; use repeated `--batch` arguments with `--prepare`.
- `tools/vocabulary_runtime_probe.py`: prepares and executes the three-case component workflow; `--help` describes its assembled-artifact and original-fixture arguments.
- `tools/vocabulary_elf_partition.py`, `vocabulary_compare_elf_images.py`, `vocabulary_symbols.py`: frozen assembly helpers; the preparer copies them under their original worker module names.
- `tools/vocabulary_sdk_core.py`: exact reader used by the accepted vocabulary runtime, separate from subsequent reader improvements.
- `tools/core_contract_probe.py` and `contract_sdk_core.py`: independently accepted offline contracted reader; `core_topology_probe.py` and `topology_sdk_core.py` retain the earlier topology diagnostic separately.

Public drivers change source paths/imports to select these isolated helpers and the existing portable `partition_executor.py`. Frozen helper logic is unchanged. Set `CSL_LLM_SDK_IMAGE` and `CSL_LLM_CS_PYTHON` for one's own pinned SDK installation. No complete SDK rerun is claimed after these publication edits.

Reproduction requires locally reconstructed original fixtures, one's own checkpoint and SDK, complete batch outputs and their resource reports, and a writable project evidence directory. These are inspection-oriented research tools, not a turnkey complete model package. Frozen manifests enumerate original omitted artifacts; snapshots cannot pass their original manifest verification without reconstruction. Weights (including CSL initializer byte strings), ELF/core binaries and raw input/output tensors are intentionally absent. Host-specific composition/symbol receipts are omitted; their original hashes, acceptance and scoped metadata remain available. Required dependencies must not be inferred to exist merely because a manifest lists them.

## Offline reader support and failure history

The existing eight-PE runtime's actual core had already passed independent output/progress/weight inspection; this was offline evidence, not another SDK run. The full-topology audit `core-topology-audit-20260908T162341268872Z` then verified all 9650 progress values and 6272 final tagged-stream words from its actual stopped core in 4.075441 seconds. Its frozen timing text mentions concurrent compilation; an explicit correction records that compilation had finished and ELF/DSR audits were concurrent. The original report is preserved alongside that correction.

`core-contract-audit-20260908T190039713537Z` uses the already accepted vocabulary case-0 core and a hash-bound symbol contract, avoiding repeated parsing of 9499 ELF symbol tables. All weights, metadata, logits, lookup and winner comparisons pass. Total time is 43.021839 seconds, peak RSS 862344 KiB; proof/JSON 7.260565 s, core load 0.252945 s, read/compare 32.465207 s. Another SDK case ran concurrently, so this is not an isolated benchmark or device-inference speedup. Host tests reject changed image hashes, incomplete/duplicate coordinates, invalid SRAM ranges and unbound artifact receipts.

Earlier failures are retained as limitations: 161657 could not access an external parent through the container mount; 162043 used an unavailable unexported symbol name, later replaced by the explicit compiler-generated output symbol. The 155803 public `read_symbol` attempt failed because its required symbol table was unavailable; ordinary composed-case transfers preceding it passed, but the baseline was not reached. This is not evidence that ELF `.symtab` was absent, nor that the public API now works. The successful readers inspect stopped cores offline and do not qualify live memory access.
