# Reference and precision

The model is Qwen/Qwen2.5-0.5B-Instruct, revision `7ae557604adf67be50417f59c2c2f167def9a775`. The reference uses Transformers 4.51.3, CPU PyTorch 2.7.1, eager attention and deterministic execution. Original BF16 checkpoint values are promoted exactly to FP32. Canonical generation uses raw-logit greedy selection with repetition penalty 1.0 and deterministic smallest-index tie handling. These override the checkpoint default repetition penalty of 1.1 explicitly.

Official BF16 and FP32 arithmetic are separate comparisons: historical fixed-prefix top-1 agreement was 41/42, with maximum relative logit L2 difference 0.0465134. The device target is canonical FP32, with exact accepted greedy tokens and the fixed limits in `configs/precision.json`; these differences are not permission to relax acceptance.

`reference_semantics.py` implements independent model equations/control flow for RMS, QKV bias, half-split RoPE, GQA, causal softmax, residuals and SwiGLU using the original weights. It shares PyTorch arithmetic primitives with the official reference; it is not an independent BLAS implementation. CPU reference completion does not qualify any CSL layer.

The complete model and raw per-step tensors are deliberately obtained/generated locally. See [reproduction](REPRODUCING.md). Model terms and notices are obtained with the pinned upstream files.
