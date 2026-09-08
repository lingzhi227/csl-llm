# Release status — 2026-09-08

**Latest accepted integration: one complete native decoder layer0**, with three consecutive canonical positions and two changed tokens after reset. The full24-layer model has not run end to end. [Decoder evidence and reproduction](DECODER.md).

| Area | Accepted evidence | Remaining boundary |
| --- | --- | --- |
| Model/reference | Pinned494,032,768-parameter checkpoint; BF16 and FP32 CPU reference; independent semantic checks | CPU reference is not SDK inference |
| Linear/MLP | Full native UP/GATE/DOWN separately; complete device-composed resident MLP; bit-exact DSD-copy variant | MLP cycle improvement is a simulator controller interval, not hardware throughput |
| Layer0 decoder | RMSNorm,biasedQKV,RoPE,causalGQA,persistentKV,output projection,both residual paths andMLP;5calls | Cache contents observed in occupied stripe0 per head; not long-context/full-model acceptance |
| Token control |9scripted metadata cases,EOS/reset/repeat and2048/256counters | No neural inference or KV capacity qualification |
| Layer parameterization | Offline24-layer tensor/ownership coverage, default layout matches accepted layer0 | No actual24-layer routes/execution; layer23 unexecuted |
| Resources | Accepted layer0 static application ELF/DSR/initializer checks | No dynamic-stack, full-model fit or hardware proof |

Complete24-layer/full-vocabulary generation and full-model2048total/256generated acceptance remain pending. A bounded layer0 pass does not close all S4 coverage or S5–S6. The active vocabulary experiment is not a published pass.

Earlier bounded components, failed attempts and baseline versions are retained with their original review scope. Historical run manifests list omitted raw inputs, model-bearing generated modules and binaries; these are not directly runnable public bundles. Public preparation tools create new local evidence using an independently obtained checkpoint and SDK. Publication host-path edits have not received a fresh SDK rerun.
