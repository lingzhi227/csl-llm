# Release status — 2026-09-08

S0 is accepted for checkpoint identity, official CPU reference and the initial precision contract. The model has 290 independent tensors and 494,032,768 parameters. Official BF16 and FP32 references each cover three prompts and 42 steps. CPU results are not CSL inference.

S1 remains open for full-model deployment feasibility. Local and eight-PE linear computations have recorded SDK passes, as do selected regional communication probes. A 192×210 layout enumerates weights and KV ownership; complete code, memory, transport and lifecycle fit are not qualified. A 64×64 replicated probe sampled only four corner outputs and is not full-grid numerical acceptance.

S2 has initial vector, RoPE and embedding/selection component passes. These do not qualify arbitrary dimensions, all aliases or the complete operator library. Embedding tests cover 112 elements from a selected row; selection tests cover 128 logits and pair merging, not 151,936-way vocabulary selection.

S3 now has accepted bounded real-trace GQA and separate KV capacity diagnostics; selected frozen implementations are included. The complete stage remains open. See [milestones](MILESTONES.md). S4 complete layer, S5 complete model generation and S6 capacity/performance acceptance remain pending. No model-generated tokens from a complete CSL pipeline are claimed.

The independent review files retain original evidence hashes and exact measured scope. Full original binary outputs and execution snapshots are not included. This publication is a selected checkpoint, not a live development mirror.
