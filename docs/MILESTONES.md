# Accepted component milestones — 2026-09-08

| Milestone | Accepted scope | Remaining limitation |
| --- | --- | --- |
| Complete resident layer-0 MLP | UP/GATE/SwiGLU/DOWN on device;916 weight tiles;4 cases;1280 readiness/progress locations | Not RMSNorm/residual/decoder/model; static initialization only |
| Full layer-0 DOWN projection | Complete 896×4864 matrix, 308 PEs/7 roots, 44 input shards with padded tail; 4 cases | Input is a host fixture; not device-composed SwiGLU or complete MLP |
| Full layer-0 GATE projection | Complete 4864×896 matrix, 304 PEs, 38 roots; 4 original-input cases; paired u32 BF16 transfer and column reads | Not whole MLP; no new independent ELF/DSR audit; not every replica output |
| Paired BF16 transport | Exact u32 transfer/readback of one real 128×112 tile; 7 lookup/selection cases | Not exhaustive SDK bit-pattern coverage or full-vocabulary selection |
| Full layer-0 UP projection | Complete 4864×896 matrix, resident BF16 on 304 PEs, 38 roots covering all outputs, 4 input cases | Not whole MLP/layer/model; all PE progress but only root output vectors read |
| Stateful causal GQA | 34 real reference tokens, reset then 3 changed tokens; independent 14-head causal results at 8 root checkpoints | Cache content sampled in 3 stripes per KV head; not every-position readback or full layer |
| KV capacity diagnostic | 2048, then reset to 65 and 1 positions; all 132 PE counts verified, root contexts and 3 cache stripes per head checked | Inputs are device-generated diagnostic trajectories from real seeds, not 2048 actual model-generated tokens |
| Packet completion handshake | 2 sessions × 8 sequential opposite-corner packets, lengths 1/2/7/30/31, ACK, exact payload/padding/progress/reset | Not concurrent senders or complete-model transport; preceding compile/progress failures are retained in PACKET-FAILURES.json |
| Resident BF16 packing | All 494,032,768 checkpoint parameter words and zero padding checked; 34,552 matrix tiles | Host input artifact verification, not SDK inference; weights are not published |

Original manifests, results, frozen CSL/driver inputs and selected independent reviews are under `validation/`. Derived execution summaries deliberately omit personal host paths. Raw NPZ inputs/outputs, original executors and SDK artifacts are absent, so these are inspection snapshots, not directly executable bundles. Use the public preparation tools to produce fresh bundles.

The old sum-only regional collective is retained. The accepted real-trace and capacity attention snapshots use separately named collective/KV modules, so later protocol changes do not silently replace the earlier qualified code.

S3 has component milestones but is not closed. Other native projections and complete decoder layers, 24-layer/full-vocabulary generation and full-model 2048/256 acceptance remain pending. Physical hardware validation remains separate.
