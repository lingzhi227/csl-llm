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

## Stateful attention, packet transport and weight packing

After generating the same FP32 reference, `attention_probe.py`, `attention_capacity_probe.py` and `packet_probe.py` use the same `--prepare` / `--execute` pattern. The capacity experiment uses synthetic diagnostic trajectories from real projection seeds. Keep it separate from real-model inference claims. Frozen inspection snapshots under `validation/frozen/` omit inputs/executors and must not be launched directly.

Generate a local placement plan using the command above, then pack locally:

```sh
python tools/pack_weights.py --plan evidence/local-layout.json
```

Verify the printed pack directory with `python tools/verify_weight_pack.py evidence/resident-weights-REPLACE_WITH_PRINTED_TIMESTAMP models/qwen2.5-0.5b-7ae5576/model.safetensors`. Packing produces approximately 1 GB of weights plus metadata; those outputs remain local and ignored. Packing success is not device execution.

## Full native UP projection

After preparing the same reference, run `python tools/projection_probe.py --prepare --projection up`, then `python tools/projection_probe.py --execute evidence/projection-up-REPLACE_WITH_PRINTED_TIMESTAMP`. Only UP is exposed by the published preparation CLI. It retains the accepted u16 weight-transfer path and separately names its frozen region/collective dependencies. The original SDK experiment took about 877 seconds; that wall time is not a full-model performance measurement.

## Full native GATE and paired transport

`python tools/gate_projection_probe.py --prepare --projection gate` creates a frozen GATE bundle; execute its printed directory using the same script with `--execute`. The accepted implementation packs original BF16 words into u32 transfers and reads roots by column. `selection_u32_probe.py` offers the same preparation/execution workflow for the separately accepted local transport/lookup probe. UP retains `projection_probe.py` and its u16 transport. These different workloads are not a controlled speedup comparison.

## Full native DOWN

Run `python tools/down_projection_probe.py --prepare --projection down`, then execute the printed directory with the same tool and `--execute`. The host prepares this standalone projection input from the retained reference. This does not demonstrate a device-connected UP/GATE/SwiGLU/DOWN pipeline. The resource report covers static application ELF sections, not complete runtime memory safety.
