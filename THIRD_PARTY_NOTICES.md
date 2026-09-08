# Attribution and licensing

The FP32 GEMV adapts the Cerebras SDK `tutorials/gemv-02-memory-dsds` memory-DSD pattern (Copyright 2025 Cerebras Systems). The vector reduction structure follows the SDK benchmark BLAS reference. Retained notices and Apache-2.0 license text are in `csl/kernels/`. Upstream: https://github.com/Cerebras/sdk-examples . SDK modules are imported from the user's installation and are not vendored.

The model target is Qwen/Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`. Model files are downloaded from upstream under their own terms; this repository does not distribute weights or tokenizers. The official reference uses Hugging Face Transformers 4.51.3 and PyTorch.

This publication does not assign a new blanket license to original project code. The included Apache license applies to the attributed material; public repository visibility should not be read as a blanket licensing grant. No upstream endorsement is implied.
