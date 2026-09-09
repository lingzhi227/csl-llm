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


## Eleventh and twelfth partitions — 2026-09-09

Original intervals 9728–10751 and 10752–11775 independently close, extending static coverage to **11776 of 34552 matrix blocks** across twelve intervals. Both have 1401 ELF classes, 45456-byte maximum static high-water and 1222 DSR families. Independent initializer checks cover all 40530 application coordinates; whole-ELF compatibility verifies selected original blocks, 5978 common coordinates and ordered SDK I/O/RPC metadata. Process cleanup is independently recorded.

| Interval | Compile time | Compile peak RSS | Initializer/resource audit | Whole-ELF contract |
| --- | ---: | ---: | --- | --- |
| 9728–10751 | 681.745346 s | 10496696 KiB | model-elf-audit-20260909T031146048688Z | model-contract-20260909T031703587119Z |
| 10752–11775 | 679.747520 s | 10495392 KiB | model-elf-audit-20260909T032652818925Z | model-contract-20260909T033259893327Z |

The [fixed ledger](../release/MODEL-PARTITION-LEDGER.json) and [publication checks](../release/MODEL-PARTITION-11-12-CHECKS.json) bind both new intervals, their independent reviews, original manifests and execution/result identities. Six complete remote manifests were reverified. Selected source and derived summaries exclude model literals, raw tensors, binaries and private paths; original hashes retain their meanings. No compiler, SDK simulation or CPU model calculation was repeated. This remains static deployment coverage, not full-model assembly or neural inference acceptance.


## Thirteenth and fourteenth partitions — 2026-09-09

Original intervals 11776–12799 and 12800–13823 independently close, extending static coverage to **13824 of 34552 matrix blocks** across fourteen intervals. Both have 1401 ELF classes, 45456-byte maximum static high-water and 1222 DSR families. Independent initializer checks cover all 40530 application coordinates; whole-ELF compatibility verifies selected original blocks, 5978 common coordinates and ordered SDK I/O/RPC metadata. Process cleanup is independently recorded.

| Interval | Compile time | Compile peak RSS | Initializer/resource audit | Whole-ELF contract |
| --- | ---: | ---: | --- | --- |
| 11776–12799 | 684.678763 s | 10495032 KiB | model-elf-audit-20260909T035446861737Z | model-contract-20260909T035944013621Z |
| 12800–13823 | 683.831056 s | 10494616 KiB | model-elf-audit-20260909T040300560930Z | model-contract-20260909T040841471502Z |

The [fixed ledger](../release/MODEL-PARTITION-LEDGER.json) and [publication checks](../release/MODEL-PARTITION-13-14-CHECKS.json) bind both new intervals, their independent reviews, original manifests and execution/result identities. Six complete remote manifests were reverified. Selected source and derived summaries exclude model literals, raw tensors, binaries and private paths; original hashes retain their meanings. No compiler, SDK simulation or CPU model calculation was repeated. This remains static deployment coverage, not full-model assembly or neural inference acceptance.


## Fifteenth and sixteenth partitions — 2026-09-09

Original intervals 13824–14847 and 14848–15871 independently close, extending static coverage to **15872 of 34552 matrix blocks** across sixteen intervals. Both have 1401 ELF classes, 45456-byte maximum static high-water and 1222 DSR families. Independent initializer checks cover all 40530 application coordinates; whole-ELF compatibility verifies selected original blocks, 5978 common coordinates and ordered SDK I/O/RPC metadata. Process cleanup is independently recorded.

| Interval | Compile time | Compile peak RSS | Initializer/resource audit | Whole-ELF contract |
| --- | ---: | ---: | --- | --- |
| 13824–14847 | 672.427124 s | 10493680 KiB | model-elf-audit-20260909T042756388119Z | model-contract-20260909T043227875920Z |
| 14848–15871 | 675.515288 s | 10493828 KiB | model-elf-audit-20260909T043504093883Z | model-contract-20260909T044054425908Z |

The [fixed ledger](../release/MODEL-PARTITION-LEDGER.json) and [publication checks](../release/MODEL-PARTITION-15-16-CHECKS.json) bind both new intervals, their independent reviews, original manifests and execution/result identities. Six complete remote manifests were reverified. Selected source and derived summaries exclude model literals, raw tensors, binaries and private paths; original hashes retain their meanings. No compiler, SDK simulation or CPU model calculation was repeated. This remains static deployment coverage, not full-model assembly or neural inference acceptance.


## Seventeenth and eighteenth partitions — 2026-09-09

Original intervals 15872–16895 and 16896–17919 independently close, extending static coverage to **17920 of 34552 matrix blocks** across eighteen intervals. Both have 1401 ELF classes, 45456-byte maximum static high-water and 1222 DSR families. Independent initializer checks cover all 40530 application coordinates; whole-ELF compatibility verifies selected original blocks, 5978 common coordinates and ordered SDK I/O/RPC metadata. Process cleanup is independently recorded.

| Interval | Compile time | Compile peak RSS | Initializer/resource audit | Whole-ELF contract |
| --- | ---: | ---: | --- | --- |
| 15872–16895 | 671.538693 s | 10493140 KiB | model-elf-audit-20260909T045932334038Z | model-contract-20260909T050349517847Z |
| 16896–17919 | 682.741879 s | 10494308 KiB | model-elf-audit-20260909T050655191197Z | model-contract-20260909T051245189026Z |

The [fixed ledger](../release/MODEL-PARTITION-LEDGER.json) and [publication checks](../release/MODEL-PARTITION-17-18-CHECKS.json) bind both new intervals, their independent reviews, original manifests and execution/result identities. Six complete remote manifests were reverified. Selected source and derived summaries exclude model literals, raw tensors, binaries and private paths; original hashes retain their meanings. No compiler, SDK simulation or CPU model calculation was repeated. This remains static deployment coverage, not full-model assembly or neural inference acceptance.


## Nineteenth and twentieth partitions — 2026-09-09

Original intervals 17920–18943 and 18944–19967 independently close, extending static coverage to **19968 of 34552 matrix blocks** across twenty intervals. Both have 1401 ELF classes, 45456-byte maximum static high-water and 1222 DSR families. Independent initializer checks cover all 40530 application coordinates; whole-ELF compatibility verifies selected original blocks, 5978 common coordinates and ordered SDK I/O/RPC metadata. Process cleanup is independently recorded.

| Interval | Compile time | Compile peak RSS | Initializer/resource audit | Whole-ELF contract |
| --- | ---: | ---: | --- | --- |
| 17920–18943 | 683.816164 s | 10494792 KiB | model-elf-audit-20260909T053140055002Z | model-contract-20260909T053618409132Z |
| 18944–19967 | 675.511452 s | 10493496 KiB | model-elf-audit-20260909T053922183984Z | model-contract-20260909T054533797924Z |

The [fixed ledger](../release/MODEL-PARTITION-LEDGER.json) and [publication checks](../release/MODEL-PARTITION-19-20-CHECKS.json) bind both new intervals, their independent reviews, original manifests and execution/result identities. Six complete remote manifests were reverified. Selected source and derived summaries exclude model literals, raw tensors, binaries and private paths; original hashes retain their meanings. No compiler, SDK simulation or CPU model calculation was repeated. This remains static deployment coverage, not full-model assembly or neural inference acceptance.


## Twenty-first and twenty-second partitions — 2026-09-09

Original intervals 19968–20991 and 20992–22015 independently close, extending static coverage to **22016 of 34552 matrix blocks** across twenty-two intervals. Both have 1401 ELF classes, 45456-byte maximum static high-water and 1222 DSR families. Independent initializer checks cover all 40530 application coordinates; whole-ELF compatibility verifies selected original blocks, 5978 common coordinates and ordered SDK I/O/RPC metadata. Process cleanup is independently recorded.

| Interval | Compile time | Compile peak RSS | Initializer/resource audit | Whole-ELF contract |
| --- | ---: | ---: | --- | --- |
| 19968–20991 | 676.578832 s | 10492076 KiB | model-elf-audit-20260909T060451410173Z | model-contract-20260909T060952504758Z |
| 20992–22015 | 676.675186 s | 10494084 KiB | model-elf-audit-20260909T061310544381Z | model-contract-20260909T061920006420Z |

The [fixed ledger](../release/MODEL-PARTITION-LEDGER.json) and [publication checks](../release/MODEL-PARTITION-21-22-CHECKS.json) bind both new intervals, their independent reviews, original manifests and execution/result identities. Six complete remote manifests were reverified. Selected source and derived summaries exclude model literals, raw tensors, binaries and private paths; original hashes retain their meanings. No compiler, SDK simulation or CPU model calculation was repeated. This remains static deployment coverage, not full-model assembly or neural inference acceptance.


## Twenty-third and twenty-fourth partitions — 2026-09-09

Original intervals 22016–23039 and 23040–24063 independently close, extending static coverage to **24064 of 34552 matrix blocks** across twenty-four intervals. Both have 1401 ELF classes, 45456-byte maximum static high-water and 1222 DSR families. Independent initializer checks cover all 40530 application coordinates; whole-ELF compatibility verifies selected original blocks, 5978 common coordinates and ordered SDK I/O/RPC metadata. Process cleanup is independently recorded.

| Interval | Compile time | Compile peak RSS | Initializer/resource audit | Whole-ELF contract |
| --- | ---: | ---: | --- | --- |
| 22016–23039 | 685.762207 s | 10493596 KiB | model-elf-audit-20260909T063833559841Z | model-contract-20260909T064531185909Z |
| 23040–24063 | 679.691890 s | 10493652 KiB | model-elf-audit-20260909T065159479171Z | model-contract-20260909T065846307824Z |

The [fixed ledger](../release/MODEL-PARTITION-LEDGER.json) and [publication checks](../release/MODEL-PARTITION-23-24-CHECKS.json) bind both new intervals, their independent reviews, original manifests and execution/result identities. Six complete remote manifests were reverified. Selected source and derived summaries exclude model literals, raw tensors, binaries and private paths; original hashes retain their meanings. No compiler, SDK simulation or CPU model calculation was repeated. This remains static deployment coverage, not full-model assembly or neural inference acceptance.


## Twenty-fifth and twenty-sixth partitions — 2026-09-09

Original intervals 24064–25087 and 25088–26111 independently close, extending static coverage to **26112 of 34552 matrix blocks** across twenty-six intervals. Partition 25 has 1400 ELF classes (1024 selected, 162 common, 214 placeholders); partition 26 has 1401 (1024 selected, 162 common, 215 placeholders). This difference is accepted composition, not a missing artifact. Both have 45456-byte maximum static high-water and 1222 DSR families. Independent initializer checks cover all 40530 application coordinates; whole-ELF compatibility verifies selected original blocks, 5978 common coordinates and ordered SDK I/O/RPC metadata. Process cleanup is independently recorded.

| Interval | Compile time | Compile peak RSS | Initializer/resource audit | Whole-ELF contract |
| --- | ---: | ---: | --- | --- |
| 24064–25087 | 684.176311 s | 10498172 KiB | model-elf-audit-20260909T072112386150Z | model-contract-20260909T072656215516Z |
| 25088–26111 | 752.608032 s | 10599600 KiB | model-elf-audit-20260909T073027719980Z | model-contract-20260909T073616638123Z |

The [fixed ledger](../release/MODEL-PARTITION-LEDGER.json) and [publication checks](../release/MODEL-PARTITION-25-26-CHECKS.json) bind both new intervals, their independent reviews, original manifests and execution/result identities. Six complete remote manifests were reverified. Selected source and derived summaries exclude model literals, raw tensors, binaries and private paths; original hashes retain their meanings. No compiler, SDK simulation or CPU model calculation was repeated. This remains static deployment coverage, not full-model assembly or neural inference acceptance.


## Twenty-seventh and twenty-eighth partitions — 2026-09-09

Original intervals 26112–27135 and 27136–28159 independently close, extending static coverage to **28160 of 34552 matrix blocks** across twenty-eight intervals. Both have 1401 ELF classes (1024 selected, 162 common, 215 placeholders), 45456-byte maximum static high-water and 1222 DSR families. Independent initializer checks cover all 40530 application coordinates; whole-ELF compatibility verifies selected original blocks, 5978 common coordinates and ordered SDK I/O/RPC metadata. Process cleanup is independently recorded.

| Interval | Compile time | Compile peak RSS | Initializer/resource audit | Whole-ELF contract |
| --- | ---: | ---: | --- | --- |
| 26112–27135 | 744.405442 s | 10598580 KiB | model-elf-audit-20260909T080320354782Z | model-contract-20260909T080832266258Z |
| 27136–28159 | 748.266003 s | 10598024 KiB | model-elf-audit-20260909T081218408065Z | model-contract-20260909T082617594657Z |

The [fixed ledger](../release/MODEL-PARTITION-LEDGER.json) and [publication checks](../release/MODEL-PARTITION-27-28-CHECKS.json) bind both new intervals, their independent reviews, original manifests and execution/result identities. Six complete remote manifests were reverified. Selected source and derived summaries exclude model literals, raw tensors, binaries and private paths; original hashes retain their meanings. No compiler, SDK simulation or CPU model calculation was repeated. This remains static deployment coverage, not full-model assembly or neural inference acceptance.
