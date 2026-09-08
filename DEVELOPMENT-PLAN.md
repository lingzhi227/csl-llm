# Staged implementation plan

Each stage has implementation, independent acceptance and a measured next-step decision. The original plan is a hypothesis, not a fixed compiler architecture. Update this document as evidence changes the best design, retaining decision history.

| Stage | Deliverable | Exit evidence |
| --- | --- | --- |
| S0 Reproducible model/reference | Fetch pinned official weights/config/tokenizer; lock reference environment; record tensor inventory, tokenizer/template, prompts and official greedy output; explicit dtype/error policy | Hash-verified artifacts, reproducible reference on available host, model semantic tests, recorded inference outputs; no SDK claim |
| S1 SDK deployment and cost feasibility | Characterize SDK2.10.1 environment, representative real-size kernels, perPE layout, resident weights/KV/code/scratch and resource plan; estimate full-model simulation cost from measurements | Actual compile/link and simulator probes at target local dimensions, explicit memory/resource accounting, feasible full-model execution plan and identified bottlenecks |
| S2 Minimum reusable CSL library | Accurate f32 accumulation linear algebra, bias, RMSNorm, stable softmax, SiLU, residual, Qwen RoPE, embedding/gather and deterministic argmax; reuse prior validated code with attribution | SDK numerical checks using real weights/activations and adversarial cases, input/alias contracts, repeated calls, mapped resource ownership |
| S3 Stateful causal GQA |14Q/2KV mapping, causal/block prefill, stable online softmax if useful, persistent KV append/read/position/reset | Per-head attention and cache checks vs independent reference; incremental vs full-prefix equivalence; block/capacity/reset/isolation cases |
| S4 Complete decoder layer | Native model dimensions and real layer weights, full attention/MLP/residual composition with resident data | Prefill and several cached decode steps, all intermediate stages, resource/lifecycle audits and cycles |
| S5 Complete model generation | Embedding→24layers→finalnorm→full LM head→deviceargmax; actual weights and tokenizer; autoregressive generation | Actual complete-model SDK run, original-input reference/logits/token checks, persistent cache, no hidden host neural compute; short prompts are first milestone |
| S6 SDK acceptance and optimization | Reach declared context2048 and generation256 capacity, short/medium/boundary prompts, EOS, reset and long-run behavior; improve measured latency and library reuse without weakening correctness | End-to-end SDK evidence, fixed numerical/generation gates, complete reproducible runner, benchmarks separating host/simulator/device cycles, tested deployment/resource plan |
| H Deferred hardware qualification | Use same model/contracts on provided physical wafer, actual placement and instrumentation | Only after user supplies access: real-device correctness and TTFT/throughput/memory; never substitute simulator estimates |

The current SDK completion boundary is S0–S6. H is a separately pending validation, not a reason to abandon SDK work. No unrelated numerical kernels, training, serving concurrency, quantization or alternate-model expansion before this target unless justified and explicitly scoped.

First concrete work: S0 artifact acquisition and official reference; in parallel only low-cost read-only SDK capability/resource assessment, not competing full simulations. Record measured full-model feasibility early. Maintain machine-readable phase status and exact next action.
