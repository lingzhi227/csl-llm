# Accepted component milestones — 2026-09-08

| Milestone | Accepted scope | Remaining limitation |
| --- | --- | --- |
| Stateful causal GQA | 34 real reference tokens, reset then 3 changed tokens; independent 14-head causal results at 8 root checkpoints | Cache content sampled in 3 stripes per KV head; not every-position readback or full layer |
| KV capacity diagnostic | 2048, then reset to 65 and 1 positions; all 132 PE counts verified, root contexts and 3 cache stripes per head checked | Inputs are device-generated diagnostic trajectories from real seeds, not 2048 actual model-generated tokens |
| Packet completion handshake | 2 sessions × 8 sequential opposite-corner packets, lengths 1/2/7/30/31, ACK, exact payload/padding/progress/reset | Not concurrent senders or complete-model transport; preceding compile/progress failures are retained in PACKET-FAILURES.json |
| Resident BF16 packing | All 494,032,768 checkpoint parameter words and zero padding checked; 34,552 matrix tiles | Host input artifact verification, not SDK inference; weights are not published |

Original manifests, results, frozen CSL/driver inputs and selected independent reviews are under `validation/`. Derived execution summaries deliberately omit personal host paths. Raw NPZ inputs/outputs, original executors and SDK artifacts are absent, so these are inspection snapshots, not directly executable bundles. Use the public preparation tools to produce fresh bundles.

The old sum-only regional collective is retained. The accepted real-trace and capacity attention snapshots use separately named collective/KV modules, so later protocol changes do not silently replace the earlier qualified code.

S3 has component milestones but is not closed. Full native projections, composed FFN, complete decoder layers, 24-layer/full-vocabulary generation and full-model 2048/256 acceptance remain pending. Physical hardware validation remains separate.
