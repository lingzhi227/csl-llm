# Release status — 2026-09-08

S0 is accepted for checkpoint identity, official CPU reference and the initial precision contract. The model has 290 independent tensors and 494,032,768 parameters. Official BF16 and FP32 references each cover three prompts and 42 steps. CPU results are not CSL inference.

S1 remains open for full-model deployment feasibility. Local and eight-PE linear computations have recorded SDK passes, as do selected regional communication probes. A 192×210 layout enumerates weights and KV ownership; complete code, memory, transport and lifecycle fit are not qualified. A 64×64 replicated probe sampled only four corner outputs and is not full-grid numerical acceptance.

S2 has initial vector, RoPE and embedding/selection component passes. These do not qualify arbitrary dimensions, all aliases or the complete operator library. Embedding tests cover 112 elements from a selected row; selection tests cover 128 logits and pair merging, not 151,936-way vocabulary selection.

S3 now has accepted bounded real-trace GQA and separate KV capacity diagnostics; selected frozen implementations are included. The complete stage remains open. See [milestones](MILESTONES.md). S4 complete layer, S5 complete model generation and S6 capacity/performance acceptance remain pending. No model-generated tokens from a complete CSL pipeline are claimed.

The independent review files retain original evidence hashes and exact measured scope. Full original binary outputs and execution snapshots are not included. This publication is a selected checkpoint, not a live development mirror.

The complete layer-0 UP4864×896 projection now has a recorded SDK pass: 304 resident-weight PEs, 38 output roots, four input cases. Canonical relative L2 is 2.4136e-7; last-feature one-hot and zero are exact. GATE subsequently passed in its separately recorded run; DOWN subsequently passed separately; the full FFN subsequently passed separately; complete-layer integration remains pending. All 304 progress values are checked; output replicas beyond the 38 roots are not read back.

Full native GATE4864×896 passes four inputs: canonical relative L2 2.2213e-7, peak 5.0350e-7; one-hot and zero are exact. Thirty-eight roots cover final outputs and all304PE progress values pass. Paired-u32 transfer and column readback are bound to this frozen run. No new independent ELF/DSR audit is claimed. The complete MLP subsequently passed separately; layer/model remain pending.

Full native DOWN896×4864 passes canonical, last-feature one-hot, zero and repeated canonical input. Relative L2 is 1.1876e-7, peak 7.9498e-8; one-hot/zero are exact. Seven roots cover896 outputs, all308PE progress values pass;44 shards include a final48 valid features plus64 padding. Input is prepared on the host from independently checked trace-derived SiLU, not a device-composed SwiGLU result. Independent static application-ELF review covers44 classes with maximum35712 bytes; no dynamic stack, IO-class, DSR or composed-MLP fit claim.

The complete resident layer-0 MLP is now accepted for four inputs and device-only intermediate computation. See [resident MLP](RESIDENT-MLP.md). S2 as a whole, complete layers and full-model S0–S6 acceptance remain open.
