# Model weight partition qualification

Two independently compiled partitions now cover original matrix blocks 0–1535, out of 34552 required by the full model. Their initializer, static resource and whole-ELF compatibility checks pass. They have not been assembled into a fully initialized model and do not add a neural runtime result.

| Original block interval | Original blocks in this partition | Compile time | Compile peak RSS | Application ELF classes |
| --- | ---: | ---: | ---: | ---: |
| 0–511 | 512 | 392.903389 s | 5881412 KiB | 889 |
| 512–1535 | 1024 | 636.782091 s | 10493376 KiB | 1401 |

Each partition retains the complete 193 × 210 application layout, with unselected matrix blocks represented by zeros. Both have maximum static high-water 45456 bytes. The first resource audit checks 710 final DSR families, 7784 nodes and 32857 interference edges; the second checks 1222 families, 12681 nodes and 52133 edges. Original selected matrix data, 25 normalization controllers, selected bias roots and placeholder bytes/alignment are independently checked. The second partition has nine selected original bias roots; this is not complete original-bias deployment.

The offline initializer/resource runs took 143.693764 s / 121532 KiB and 152.811075 s / 137560 KiB respectively. The separate compatibility runs took 8.208980 s / 218732 KiB and 9.221093 s / 199408 KiB. Process cleanup was independently confirmed.

## What compatibility establishes

The first partition contributes 512 selected ELF files and the second 1024. Each has 162 common ELF files covering 5978 common coordinates and 215 placeholder ELF files. Independent inspection checks that entire selected artifacts belong to the requested tile set, common allocated bytes and ordered loader segments match the baseline, and output metadata plus ordered SDK I/O/RPC remain compatible. These checks support later composition; they do not prove that all 34552 original blocks have been composed or run.

`release/MODEL-PARTITION-LEDGER.json` records the two accepted intervals and their original compile/audit/contract identities. The development ledger continues to grow; this public selection is a fixed historical snapshot. Independent reviews and selected original manifests, configuration and results are retained under `validation/`.

## Selected tools and reproduction boundary

`tools/model_partition_contract_probe.py` prepares a bounded offline compatibility audit from locally completed candidate and baseline initializer/resource audits. Its default source root is `support/model_partition/`, an isolated copy of the second accepted partition's contract helper, ELF readers and layout definitions. The only public changes select that support root and portable SDK paths. Use `--help` for fixture arguments and `CSL_LLM_SDK_IMAGE` / `CSL_LLM_CS_PYTHON` for one's own SDK.

The original compile and audit drivers are inspection snapshots with original evidence hashes; the entire evolving static-input/model-layout preparation chain is not supplied as a standalone frontend. Reproduction needs local checkpoint declarations, compiled ELF artifacts and completed audit fixtures. Public manifests list original omitted dependencies. Weight literal modules, raw arrays, ELF binaries and personal paths are excluded, so frozen snapshots alone are not runnable. Publication verification did not rerun compilation or SDK simulation.

The previous [full-layout boot](MODEL-BOOT.md) remains a separate accepted run with only 12 original matrix blocks. Its runtime result must not be transferred to these larger static partitions. Complete original-model composition, computation, persistent generation and full capacity remain open.

## Subsequent accepted coverage

The third (1536–2559) and fourth (2560–3583) partitions extend original static coverage to 3584 of 34552 blocks. Their initializer/resource/whole-ELF compatibility reviews and frozen evidence accompany the [two-layer release](TWO-LAYER.md). The table above retains the first publication scope; that publication contained four accepted intervals. Full-model assembly and execution remain open.


## Fifth and sixth partitions — 2026-09-09

Two further independently accepted intervals, 3584–4607 and 4608–5631, extend the public ledger to **5632 of 34552 original matrix blocks**. This is static block coverage, not a percentage of completed LLM development. Each candidate has 1401 application ELF classes, 45456-byte maximum static high-water, and independently checked original initializers across the 40530-coordinate layout. Each whole-ELF contract verifies 1024 selected original blocks, 162 common ELF classes across 5978 coordinates, and ordered SDK I/O/RPC compatibility.

| Interval | Compile time | Compile peak RSS | Initializer/resource audit | Whole-ELF contract |
| --- | ---: | ---: | --- | --- |
| 3584–4607 | 648.850070 s | 10493152 KiB | model-elf-audit-20260909T000133487095Z | model-contract-20260909T002817703824Z |
| 4608–5631 | 648.773233 s | 10494512 KiB | model-elf-audit-20260909T013324108058Z | model-contract-20260909T014038336694Z |

The [public interval ledger](../release/MODEL-PARTITION-LEDGER.json) links exact compile/audit/contract identities and all independent reviews. [Publication checks](../release/MODEL-PARTITION-5-6-CHECKS.json) bind the six newly verified complete remote manifests and original result hashes. Selected frozen Python drivers/dependencies accompany each run; runtime helpers from older milestones remain unchanged. Compact result/config/execution reports are derived and explicitly labeled; original manifests retain references to intentionally omitted model data and binaries. No SDK simulation or compiler was rerun for publication. Full 34552-block assembly and model inference remain unaccepted.


## Seventh and eighth partitions — 2026-09-09

Accepted intervals 5632–6655 and 6656–7679 extend original static coverage to **7680 of 34552 matrix blocks** across eight intervals. This is not a completion percentage for the LLM. Both retain 1401 ELF classes and maximum static high-water 45456 bytes; whole-ELF compatibility checks include 5978 common coordinates and ordered SDK I/O/RPC metadata.

| Interval | Compile time | Compile peak RSS | Initializer/resource audit | Whole-ELF contract |
| --- | ---: | ---: | --- | --- |
| 5632–6655 | 679.507279 s | 10495080 KiB | model-elf-audit-20260909T020338568912Z | model-contract-20260909T020826832138Z |
| 6656–7679 | 677.500871 s | 10494576 KiB | model-elf-audit-20260909T021121915327Z | model-contract-20260909T021804090838Z |

The [ledger](../release/MODEL-PARTITION-LEDGER.json) and [new publication checks](../release/MODEL-PARTITION-7-8-CHECKS.json) bind both intervals and their independent reviews. All six remote manifests were reverified; missing local metadata for the eighth compile was retrieved read-only into a temporary publication area, without changing development evidence. The [paired-compiler outcome review](../validation/reviews/s5-fullmodel-compile-pair-012655-014628-outcome-review.json) only qualifies receipts, resources and process cleanup; it is not neural inference performance. No compiler or SDK run was repeated for publication. Full original assembly and model SDK execution remain open.


## Ninth and tenth partitions — 2026-09-09

Original intervals 7680–8703 and 8704–9727 independently close, extending static coverage to **9728 of 34552 matrix blocks** across ten intervals. Each candidate has 1401 ELF classes and 45456-byte maximum static high-water; the independent initializer checks cover all 40530 coordinates, while compatibility verifies 5978 common coordinates and ordered SDK I/O/RPC metadata.

| Interval | Compile time | Compile peak RSS | Initializer/resource audit | Whole-ELF contract |
| --- | ---: | ---: | --- | --- |
| 7680–8703 | 679.560693 s | 10494312 KiB | model-elf-audit-20260909T023751025697Z | model-contract-20260909T024355577708Z |
| 8704–9727 | 674.298945 s | 10493788 KiB | model-elf-audit-20260909T024607209690Z | model-contract-20260909T025150485738Z |

The [ten-interval ledger](../release/MODEL-PARTITION-LEDGER.json) and [new publication checks](../release/MODEL-PARTITION-9-10-CHECKS.json) bind exact original intervals, source, execution/result hashes and independent reviews. Six complete remote manifests were reverified. The [second paired-compiler review](../validation/reviews/s5-fullmodel-compile-pair-021832-021844-outcome-review.json) covers resource/receipt/cleanup behavior only, not neural inference speed. Static block coverage is not a percentage of completed LLM work; complete original assembly and full-model SDK inference remain open.
