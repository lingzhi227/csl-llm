# Accepted native resident MLP

The complete layer-0 UP → GATE/SiLU → elementwise product → DOWN numerical block passed SDK 2.10.1 using original Qwen BF16 weights and device-only intermediate computation. This is a complete MLP block, not a full Transformer layer.

- 916 resident weight tiles; 1280 application PE status locations checked.
- Four inputs: canonical, negative, zero, canonical repeat. Independent original-input `math.fsum` projections and scalar `math.exp` SiLU checked all UP/GATE/activation/output values.
- Canonical final relative L2: 1.9494378843e-7 against independent full mathematics; 2.5970086418e-7 against the official reference stage.
- All readiness/progress checks passed;64 tail values remain zero; DOWN-root/final bits agree; repeated intermediate/output bits agree.
- Controller cycles:247737/247737/237753/247737. Wall time943.054859 seconds; peak process-tree RSS14223864KiB (shared pages may be counted more than once). This is not hardware performance.
- Independent static review:921 application ELF classes, maximum39008bytes,920 final DSR families and34998 interference edges. Initializer/ELF/loader-coordinate checks cover all916 tiles. These checks do not establish dynamic stack bounds or full34552tile model fit.

## Reproduce with local model files

Follow the repository reference setup first. From the repository root:

```sh
python tools/resident_mlp_probe.py --prepare
python tools/resident_mlp_probe.py --prepare-static-from evidence/resident-mlp-REPLACE_WITH_PRINTED_TIMESTAMP
python tools/resident_mlp_probe.py --execute evidence/resident-mlp-static-REPLACE_WITH_SECOND_PRINTED_TIMESTAMP
```

The first command prepares fixtures only. The accepted execution route is the second, statically initialized bundle. Do not execute the initial H2D bundle as if it were the accepted route. The generator produces916 checkpoint-bearing `weights_*.csl` modules locally. They are model payloads and are intentionally not public. The weight-free PE template and libraries are included under `csl/programs/resident_mlp/`. Public preparation paths select exact frozen dependencies; no publication-side SDK rerun is claimed.

Supporting `packet_fanin_probe.py` and `filtered_input_probe.py` use `--prepare` then `--execute` on the printed fresh directory. The one-PE `static_weights_probe.py --prepare` consumes an existing frozen native projection fixture; it is a separate initialization experiment. Its generated `pe.csl` embeds weights and is excluded too.

## Failures and limits

The earlier runtime-H2D MLP run `resident-mlp-20260908T095813614456Z` hit its1800-second budget without a retained actual numerical result. The API within the coarse stage was not localized. Do not call it an established deadlock, infer a specific SDK cause, or attribute the static success to a proven deadlock fix. Its independent timeout review is retained. Earlier fan-in and filtered-input compile failures are disclosed in their reviews.

RMSNorm, residual connection, attention integration, complete decoder layers and24-layer/full-vocabulary generation remain pending. The new DSD-copy experiment is not part of this accepted release. Static initialization of this block does not qualify full-model loading.
