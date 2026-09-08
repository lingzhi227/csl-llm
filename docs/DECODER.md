# Accepted native decoder layer 0

The complete native first layer passed SDK 2.10.1: input RMSNorm, biased Q/K/V projections, Qwen RoPE, causal GQA with persistent KV, output projection, both residual connections, post-attention RMSNorm and the complete MLP. All neural intermediate calculations are on the device.

Five calls cover canonical positions0/1/2, then reset and two changed tokens at0/1. Thirteen stages and official outputs pass independent original-BF16→float64 checks; maximum reported relative L2 across checked stages is8.167e-6. All1280PE readiness/progress and132KV statuses pass. Actual K/V contents are read from occupied stripe0 per head at each sequence end; this is not exhaustive long-context cache readback.

Controller cycles:358403/359807/361208/358403/359801. SDK wall time2004.405641s; peak process-tree RSS11671412KiB. Static review covers1180 application ELF classes (max44032bytes) and1180 DSR families/50226 edges. All1044 matrix tiles and bias/norm/frequency initialization are checked against their original source and loader coordinates. This does not prove dynamic stack bounds, hardware performance or full-model fit.

## Reproduction

Prepare the official model and FP32 reference as described in [REPRODUCING.md](REPRODUCING.md), then:

```sh
python tools/resident_decoder_probe.py --prepare
python tools/resident_decoder_probe.py --execute evidence/resident-decoder-REPLACE_WITH_PRINTED_TIMESTAMP
```

The published executable is derived from the accepted frozen layer-0 driver, with only reference/dependency paths adapted. Its exact weight-free PE/libraries live under `csl/programs/resident_decoder/`. The generator creates weight-bearing `weights_*.csl`, bias modules and `controller_aux.csl` locally; none are public payloads. Do not upload generated runs. The original110618 run timed out; the later accepted113615 run is a separate immutable result, not a retroactive success of the first run.

## Supporting token control and offline interfaces

`tools/token_sequence_probe.py --prepare` and `--execute` validate a scripted metadata controller: prompt/generated selection, two EOS IDs, reset/repeat and2048total/256generated counters across9cases. Its accepted runtime uses17008 static bytes and one DSR node. It performs no neural inference and does not qualify KV capacity or full-model generation.

`src/csl_llm/decoder_layout.py` has independently checked offline24-layer tensor/ownership mapping:25056 matrix tiles and3168KV owners. The newer parameterized driver is retained for source inspection under `validation/offline/layer-parameterization/`, not offered as a qualified runtime. The default emitter matches the accepted layer-0 layout; neither layer23 nor a24-layer execution has been accepted.

Complete24-layer/full-vocabulary generation, full-model2048/256 acceptance and hardware execution remain pending. The subsequent full-vocabulary113409 attempt hit its24GiB compile-memory guard before producing any application ELF or numerical output. Initialization changes are under investigation; this is not a decoder regression or an SDK numerical failure.
