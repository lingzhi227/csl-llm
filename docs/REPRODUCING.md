# Setup and reproduction

## Host checks

Use Python 3.11 on Linux for the intended reference and SDK workflow. Install dependencies in a virtual environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
```

The process identity/cleanup test needs Linux `/proc` and is skipped elsewhere. These host tests do not execute CSL.

## Official checkpoint and CPU reference

Run from the repository root. Download weights directly from the pinned upstream revision; this requires network access and space for weights and substantially larger activation traces.

```sh
python tools/fetch_model.py configs/target.json models/qwen2.5-0.5b-7ae5576
python tools/model_inventory.py models/qwen2.5-0.5b-7ae5576/model.safetensors evidence/local-model-inventory.json
python tools/reference_run.py models/qwen2.5-0.5b-7ae5576 configs/prompts.json evidence/reference-f32 --dtype float32
python tools/reference_semantics.py models/qwen2.5-0.5b-7ae5576 evidence/reference-f32 evidence/reference-semantics
```

The component preparation scripts use `evidence/reference-f32/arithmetic/step-000.npz` and subsequent recorded steps. Generate that reference before preparing real-weight probes. Outputs must be fresh directories; never overwrite historical evidence. Canonical arithmetic is FP32 with exact BF16 weight expansion and raw-logit greedy selection; BF16 generation can differ.

## SDK component probes

Provide an independently installed SDK 2.10.1, the pinned image, and its `cs_python` wrapper. No SDK is included. Set absolute paths:

```sh
export CSL_LLM_SDK_IMAGE=/absolute/path/to/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif
export CSL_LLM_CS_PYTHON=/absolute/path/to/cs_python
python tools/vector_probe.py --prepare
```

Preparation prints a new `evidence/vector-f32-...` directory. Execute that exact directory once:

```sh
python tools/vector_probe.py --execute evidence/vector-f32-REPLACE_WITH_PRINTED_TIMESTAMP
```

The same `--prepare` / `--execute` workflow is available in `distributed_linear_probe.py`, `regional_linear_probe.py`, `position_probe.py` and `selection_probe.py`. Each freezes its sources, inputs and SDK identity, then records compilation, actual output comparisons and process completion. The Linux execution supervisor measures and limits its own process tree. Run one simulation at a time until resource costs are understood.

Inspect `results.json` and `execution.json` together. Compile success, a progress marker or one numerical output alone is not a pass. SDK results in the published historical reviews are not a claim that these portable entry points were rerun during packaging.

## Placement proposal

```sh
python tools/plan_layout.py evidence/model-inventory.json evidence/local-layout.json
```

This is static accounting, not deployment or end-to-end inference. There is no complete-model runner in this release.
