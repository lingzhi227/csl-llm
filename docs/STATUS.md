# Release status — 2026-09-09

**Latest runtime outcome:** the first complete original-weight 24-layer attempt timed out after six hours with readiness only and no accepted neural output. Small two-PE explicit core exports now pass independent compute/communication state checks; active-case shutdown still times out. [Diagnostics and limits](SDK-CORE-EXPORT.md). Restoration and full-model inference remain unqualified.

**Complete static matrix coverage:** all 34552 original blocks across 35 intervals independently accepted. Complete static assembly and its own diagnostic ABI are now independently accepted; neural SDK acceptance remains open. [Assembly evidence](FULL-MODEL-ASSEMBLY.md). [Complete coverage evidence](MODEL-PARTITIONS.md).

Latest integration: the original-weight two-layer chain passes position 0, cached position 1 and reset-to-position-0 in one SDK instance, with 52 independent numerical checks and bit-exact reset outputs. [Scope, source and evidence](TWO-LAYER-CACHED.md). Full-model generation remains unaccepted.

Latest acceptance: the complete original 151936 × 896 tied embedding/head passes three actual SDK cases on 193 × 50 PEs. Complete original weights, logits and metadata are checked from stopped cores; ordinary winner transfer, zero/tie and fresh-process bit-repeat pass. [Full-vocabulary evidence](FULL-VOCABULARY.md). Earlier partial-vocabulary limitations below describe historical milestones. Full S0–S6 / 24-layer persistent generation remains open.

**Earlier accepted integration: one complete native decoder layer0**, with three consecutive canonical positions and two changed tokens after reset. The full24-layer model has not run end to end. [Decoder evidence and reproduction](DECODER.md).

| Area | Accepted evidence | Remaining boundary |
| --- | --- | --- |
| Model/reference | Pinned494,032,768-parameter checkpoint; BF16 and FP32 CPU reference; independent semantic checks | CPU reference is not SDK inference |
| Linear/MLP | Full native UP/GATE/DOWN separately; complete device-composed resident MLP; bit-exact DSD-copy variant | MLP cycle improvement is a simulator controller interval, not hardware throughput |
| Layer0 decoder | RMSNorm,biasedQKV,RoPE,causalGQA,persistentKV,output projection,both residual paths andMLP;5calls | Cache contents observed in occupied stripe0 per head; not long-context/full-model acceptance |
| Token control |9scripted metadata cases,EOS/reset/repeat and2048/256counters | No neural inference or KV capacity qualification |
| Layer parameterization | Offline24-layer tensor/ownership coverage, default layout matches accepted layer0 | No actual24-layer routes/execution; layer23 unexecuted |
| Resources | Accepted layer0 static application ELF/DSR/initializer checks | No dynamic-stack, full-model fit or hardware proof |

Complete24-layer/full-vocabulary generation and full-model2048total/256generated acceptance remain pending. A bounded layer0 pass does not close all S4 coverage or S5–S6. The full-vocabulary113409 attempt stopped during compilation at its24GiB process-tree memory guard, before any application ELF or numerical output. The initialization approach is being revised without reducing the151936-token vocabulary target. This does not invalidate the accepted decoder.

Earlier bounded components, failed attempts and baseline versions are retained with their original review scope. Historical run manifests list omitted raw inputs, model-bearing generated modules and binaries; these are not directly runnable public bundles. Public preparation tools create new local evidence using an independently obtained checkpoint and SDK. Publication host-path edits have not received a fresh SDK rerun.

Native-u8 static initialization now has a one-tile SDK pass with4GEMVs and complete before/after weight readback. Supporting compact/streamed-layer checks are compile-only. The later full-vocabulary124830 attempt was explicitly stopped early without a complete weight-block result; it is not a timeout or pass. See [native-u8 scope](NATIVE-U8.md).

Experimental intact-artifact composition passes two independent GEMVs in isolated SDK processes, not application collectives.256/1024 real-tile compilation probes include zero-filled placeholders; they do not qualify full9496real-tile vocabulary deployment. [Exact limits](PARTITION-EXPERIMENT.md).

Eight-PE cross-partition line allreduce is now accepted for the full group-0 128 × 896 contraction, four cases per isolated variant. This supersedes the earlier statement that no application collective is qualified; full vocabulary/model inference remains unaccepted. [Scoped evidence](PARTITION-EXPERIMENT.md#eight-pe-cross-partition-collective-acceptance).

193 × 50 tagged streaming transport is independently accepted for selected-column ordering, padding exclusion, repeat/reset and all 9650 completion counters. Tiny support and original tiles 1024–2047 compile-only evidence accompany it. No neural/full-vocabulary runtime acceptance is inferred. [Evidence](STREAM-OUTPUT.md).

Full 193 × 210 model layout: independent static checks and SDK boot accepted for 40530 PEs. Only 12 of 34552 matrix blocks contain original weights; reset/prepare passes, no compute. [Exact scope](MODEL-BOOT.md).

## Historical progress entries

The entries below preserve earlier milestones in order; “latest” and pending assembly statements refer to their historical recording time. Current assembly and runtime status are given above.

Original model blocks 0–1535 now pass two independent initializer/resource/whole-ELF compatibility audits. Full 34552-block assembly and neural execution remain open. [Partition ledger and limits](MODEL-PARTITIONS.md).

Latest published full-model static coverage: six accepted intervals cover original blocks 0–5631 of 34552. Full-model assembly and SDK inference remain open. [Evidence and limits](MODEL-PARTITIONS.md).

Full-model validation preparation is published separately: independently accepted single-token CPU reference, expanded static symbols and source-only future integration tools. [Evidence levels and boundaries](MODEL-VALIDATION-PREPARATION.md). None extends the bounded two-layer neural SDK acceptance.

Latest static ledger covers eight intervals, original blocks 0–7679 of 34552. The [runtime lifecycle source successor](MODEL-VALIDATION-PREPARATION.md) fixes exception-safe cleanup and includes host-tested request helpers; complete model SDK execution remains open.

Latest static coverage: ten intervals, original blocks 0–9727 of 34552. Guarded generation preparation now has a reviewed source-only owner and host admission tests; no actual full-model first-token or generation SDK acceptance yet. [Source/evidence levels](MODEL-VALIDATION-PREPARATION.md).

Latest published static ledger: twelve closed intervals, original blocks 0–11775 of 34552. [Evidence and limits](MODEL-PARTITIONS.md). Full-model assembly and SDK inference remain unaccepted; generation snapshot023236 remains source-only.

Latest published static ledger: fourteen closed intervals, original blocks 0–13823 of 34552. [Evidence and limits](MODEL-PARTITIONS.md). Full-model assembly and SDK inference remain unaccepted; generation snapshot023236 remains source-only.

Latest published static ledger: sixteen closed intervals, original blocks 0–15871 of 34552. [Evidence and limits](MODEL-PARTITIONS.md). Full-model assembly and SDK inference remain unaccepted; generation snapshot023236 remains source-only.

Latest published static ledger: eighteen closed intervals, original blocks 0–17919 of 34552. [Evidence and limits](MODEL-PARTITIONS.md). Full-model assembly and SDK inference remain unaccepted; generation snapshot023236 remains source-only.

Latest published static ledger: twenty closed intervals, original blocks 0–19967 of 34552. [Evidence and limits](MODEL-PARTITIONS.md). Full-model assembly and SDK inference remain unaccepted; generation snapshot023236 remains source-only.

Latest published static ledger: twenty-two closed intervals, original blocks 0–22015 of 34552. [Evidence and limits](MODEL-PARTITIONS.md). Full-model assembly and SDK inference remain unaccepted; generation snapshot023236 remains source-only.

Latest published static ledger: twenty-four closed intervals, original blocks 0–24063 of 34552. [Evidence and limits](MODEL-PARTITIONS.md). Full-model assembly and SDK inference remain unaccepted; generation snapshot023236 remains source-only.

Latest published static ledger: twenty-six closed intervals, original blocks 0–26111 of 34552. [Evidence and limits](MODEL-PARTITIONS.md). Full-model assembly and SDK inference remain unaccepted; generation snapshot023236 remains source-only.

Latest published static ledger: twenty-eight closed intervals, original blocks 0–28159 of 34552. [Evidence and limits](MODEL-PARTITIONS.md). Full-model assembly and SDK inference remain unaccepted; generation snapshot023236 remains source-only.

Latest published static ledger: thirty closed intervals, original blocks 0–30207 of 34552. [Evidence and limits](MODEL-PARTITIONS.md). Full-model assembly and SDK inference remain unaccepted; generation snapshot023236 remains source-only.

Latest published static ledger: thirty-two closed intervals, original blocks 0–32255 of 34552. [Evidence and limits](MODEL-PARTITIONS.md). Full-model assembly and SDK inference remain unaccepted; generation snapshot023236 remains source-only.

Latest published static ledger: thirty-four closed intervals, original blocks 0–34303 of 34552. [Evidence and limits](MODEL-PARTITIONS.md). Full-model assembly and SDK inference remain unaccepted; generation snapshot023236 remains source-only.
