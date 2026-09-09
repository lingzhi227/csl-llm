"""Fresh official full-model single-token reference for initial SDK integration.

All 24 original layers and the full tied vocabulary execute on the CPU only
to construct a frozen oracle. This module must never run in an SDK worker.
"""
import datetime
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import sys


def prepare(project, source):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen2 import modeling_qwen2
    from sdk_probe import sha
    from resident_decoder_probe import explicit_layer
    from decoder_chain_reference import metric
    if importlib.metadata.version('transformers') != '4.51.3':
        raise ValueError('Pinned official Transformers 4.51.3 is required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model_path = project/'models/qwen2.5-0.5b-7ae5576'
    previous = project/'evidence/reference-f32-greedy-20260908T0554'
    if sha(model_path/'model.safetensors') != 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe':
        raise ValueError('Original model checkpoint changed')
    prior = json.loads((previous/'report.json').read_text())
    if sha(previous/'report.json') != 'ed7a56c88ffe6b6b629715b847fbc7415bbc82e6856c48ebbc03a6a080d16340':
        raise ValueError('Accepted official reference changed')
    if not sha(model_path/'manifest.json') == prior['model_manifest_sha256'] == '9e4305503478f0f118ced0c5b87b148b73ae5823504f5636fb7c778e1252c75e':
        raise ValueError('Model manifest differs from accepted original identity')
    manifest = json.loads((model_path/'manifest.json').read_text())
    for row in manifest['files']:
        if sha(model_path/row['path']) != row['sha256']:
            raise ValueError('Pinned model file changed: '+row['path'])
    implementation = Path(inspect.getfile(modeling_qwen2))
    if sha(implementation) != prior['files']['modeling_qwen2.py']:
        raise ValueError('Installed official Qwen implementation differs from accepted source')
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, attn_implementation='eager').eval()
    assert len(model.model.layers) == 24 and model.config.vocab_size == 151936
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    assert model.config.rms_norm_eps == 1e-6 and model.config.rope_theta == 1e6
    captured = {}
    selected = {'model.embed_tokens', 'model.norm'} | {f'model.layers.{i}' for i in range(24)}
    def hook(name):
        def capture(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value.detach().float().numpy().copy()
        return capture
    handles = [module.register_forward_hook(hook(name)) for name, module in model.named_modules() if name in selected]
    token = 151644
    ids = torch.tensor([[token]], dtype=torch.long)
    with torch.inference_mode():
        result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, return_dict=True)
    for handle in handles:
        handle.remove()
    logits = result.logits[0, -1].float().numpy().copy()
    assert logits.shape == (151936,) and np.isfinite(logits).all()
    winner = int(np.argmax(logits))
    captured['logits'] = logits
    for layer in range(24):
        for kind, values in [('key', result.past_key_values.key_cache), ('value', result.past_key_values.value_cache)]:
            value = values[layer].float().numpy().copy()
            assert value.shape == (1, 2, 1, 64)
            captured[f'cache.{layer}.{kind}'] = value
    # Verify canonical generate policy separately, before releasing the model.
    with torch.inference_mode():
        generated = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=1,
            do_sample=False, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=0,
            pad_token_id=model.generation_config.pad_token_id, use_cache=True)[0, 1:].tolist()
    assert generated == [winner]
    checks = []
    prior_trace = previous/'arithmetic/step-000.npz'
    assert sha(prior_trace) == prior['files']['arithmetic/step-000.npz']
    with np.load(prior_trace, allow_pickle=False) as original:
        for name in sorted(selected):
            checks.append(dict(stage=name, comparison='accepted causal prompt position zero',
                **metric(captured[name], original[name][:, :1])))
    # Explicit equations propagate their own original-input output through all
    # layers; they never substitute an SDK or official intermediate boundary.
    hidden = torch.from_numpy(captured['model.embed_tokens'][0].copy())
    weights = model.state_dict()
    with torch.inference_mode():
        for layer in range(24):
            nodes, keys, values, _ = explicit_layer(weights, hidden, layer)
            for name, value in nodes.items():
                captured[f'independent.layers.{layer}.{name}'] = value.numpy().copy()
            checks.append(dict(stage=f'layer_{layer}', comparison='independent full-prefix equations',
                **metric(nodes['output'].numpy(), captured[f'model.layers.{layer}'][0])))
            for name, value in [('key', keys), ('value', values)]:
                checks.append(dict(stage=f'cache_{layer}_{name}', comparison='independent full-prefix equations',
                    **metric(value.numpy(), captured[f'cache.{layer}.{name}'][0])))
            hidden = nodes['output']
        normalized = hidden*torch.rsqrt(hidden.square().mean(-1, keepdim=True)+1e-6)*weights['model.norm.weight']
        explicit_logits = torch.nn.functional.linear(normalized, weights['model.embed_tokens.weight'])[0].numpy()
    checks.append(dict(stage='final_norm', comparison='independent equations',
                       **metric(normalized.numpy(), captured['model.norm'][0])))
    checks.append(dict(stage='logits', comparison='independent equations', **metric(explicit_logits, logits)))
    assert int(np.argmax(explicit_logits)) == winner
    assert all(np.isfinite(value).all() for value in captured.values())
    out = project/'evidence'/('model-single-token-reference-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    (out/'single-token').mkdir(parents=True)
    np.savez_compressed(out/'single-token/step-000.npz', **captured)
    shutil.copy2(Path(__file__), out/'driver.py')
    shutil.copy2(implementation, out/'modeling_qwen2.py')
    shutil.copy2(source/'tools/resident_decoder_probe.py', out/'equations.py')
    shutil.copy2(source/'configs/precision.json', out/'precision.json')
    report = dict(success=True, dtype='float32', revision=manifest['revision'],
        model_manifest_sha256=sha(model_path/'manifest.json'), model_sha256=sha(model_path/'model.safetensors'),
        eager_attention=True, tied_embedding=True, deterministic=True, threads=1,
        versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'safetensors')},
        prompts=[dict(id='single-token', prompt_tokens=[token], generated_tokens=[winner],
            official_generate_exact=True, steps=[dict(step=0, context_tokens=1,
                selected_token=winner, trace='step-000.npz')])], cpu_checks=checks,
        accepted_official_report_sha256=sha(previous/'report.json'),
        scope='Fresh official original24layer/full151936 CPU reference, single raw token151644 and one greedy output; not chat/cached/reset/capacity or SDK inference.',
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(files={str(p.relative_to(out)): sha(p)
        for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)
    return out
