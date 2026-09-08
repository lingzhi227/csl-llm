# CSL operator contracts

These modules are bounded research implementations. Consult source comments and the historical reviews for exact shapes, precision, DSR leases and alias rules.

- `local_gemv_f32.csl`: FP32 memory-DSD accumulation, adapted from Cerebras SDK gemv-02-memory-dsds (Copyright 2025 Cerebras Systems, Apache-2.0). Reusable pointer API, descriptor reset and output clearing are project adaptations.
- `local_gemv_bf16_f32.csl`: packed BF16 weights with lossless expansion and FP32 accumulation.
- `local_gemv_bf16_f32_colmajor.csl`: column-major storage, synchronous source/destination DSR banks 3/4. Callers must finish earlier asynchronous users before entry. Inputs and output must be disjoint. Local 128×112 and eight-shard 128×896 cases have SDK evidence.
- `vector_f32.csl`: SDK BLAS-inspired reduction structure (Apache-2.0), RMSNorm, residual, stable-sign SwiGLU and finite softmax. Initial 21 SDK cases pass; compile-time bounds do not imply all shapes have been validated.
- `qwen_position_f32.csl`: half-split Qwen RoPE; nine position/alias cases pass, including position 2047. This does not implement causal attention or cache lifecycle.
- `tied_embedding_f32.csl` and `argmax_f32.csl`: exact lookup from a selected embedding tile and deterministic local selection/pair merging. Seven cases pass; full embedding assembly and full-vocabulary selection remain pending.

The license text for attributed upstream material is retained in [LICENSE](LICENSE). All performance numbers concern the tested component and SDK configuration, not full-model latency.
