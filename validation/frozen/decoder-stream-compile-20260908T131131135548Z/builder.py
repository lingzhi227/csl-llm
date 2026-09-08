"""Native selected-layer candidate with persistent KV and reset diagnostics."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def geometry(origin=(0, 0)):
    from csl_llm.decoder_layout import geometry as layer_geometry
    return layer_geometry(origin)


def make_layout(regions, *, origin=(0, 0), layer=0, fragment=False, next_destination=None, initialization='static'):
    """Emit one region, translating routes as well as PE/tensor ownership.

    A fragment is embedded in a caller-owned rectangle and export namespace.
    Its controller uses device activation packets for both boundaries. Only
    the default standalone layer has SDK numerical qualification so far.
    """
    from csl_llm.decoder_layout import model_origin
    model_origin(layer)  # Validate pinned tensor index even for alternate origins.
    bx, by = origin
    if regions != geometry(origin):
        raise ValueError('Regions must match the complete translated decoder geometry')
    if not fragment and (origin != (0, 0) or next_destination is not None):
        raise ValueError('Standalone decoder routes require origin (0,0) and host output')
    if fragment and (next_destination is None or len(next_destination) != 2 or
                     any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in next_destination)):
        raise ValueError('Composed decoder requires a nonnegative next-controller coordinate')
    if initialization not in ('static', 'sdk'):
        raise ValueError('Initialization must be static or sdk')
    controller = (bx+62, by)
    prefix = f'layer_{layer}/' if fragment else ''
    roles = dict(up=1, gate=2, down=3, q=5, k=6, v=7, o=8, kv=9)
    occupied = {}
    for name, rs in regions.items():
        for group, region in enumerate(rs):
            for ordinal in range(region.participants):
                x, y = region.point(ordinal)
                assert (x, y) not in occupied
                packet_group = 7 if name == 'k' else 8 if name == 'v' else group
                stripe = (y-region.y)*11+x-region.x if name == 'kv' else 0
                destination = regions['gate'][group].point(0) if name == 'up' else controller
                auxiliary = (f'{prefix}bias_{name}_{group}.csl' if name in ('q', 'k', 'v') and ordinal == 0 else '<empty>')
                weights = f'{prefix}weights_{x-bx}_{y-by}.csl' if name != 'kv' else '<empty>'
                occupied[x, y] = (roles[name], packet_group, ordinal, region.participants,
                                  *region.neighbors(ordinal), stripe, *destination, weights, auxiliary)
    assert len(occupied) == 1176 and controller not in occupied
    lines = ['const memcpy=@import_module("<memcpy/get_params>",.{.width=64,.height=20});',
             'layout{@set_rectangle(64,20);'] if not fragment else []
    for y in range(by, by+20):
        for x in range(bx, bx+64):
            default = (4 if (x, y) == controller else 0, 0, 0, 2, 'WEST', 'EAST', 0, *controller,
                       '<empty>', prefix+'controller_aux.csl' if (x, y) == controller else '<empty>')
            role, group, ordinal, count, negative, positive, stripe, dx, dy, weight, aux = occupied.get((x, y), default)
            if initialization == 'sdk':
                group, dx, dy, weight, aux = 0, 0, 0, '<empty>', '<empty>'
            params = (f'.memcpy_params=memcpy.get_params({x}),.role={role},.group={group},.ordinal={ordinal},'
                      f'.participants={count},.negative={negative},.positive={positive},.stripe={stripe},'
                      f'.destination_x={dx},.destination_y={dy},.weight_file="{weight}",.auxiliary_file="{aux}"')
            if initialization == 'sdk':
                params += ',.static_weights=false,.runtime_identity=true'
            if fragment and (x, y) == controller:
                params += (f',.streamed_input=true,.streamed_output=true,'
                           f'.next_x={next_destination[0]},.next_y={next_destination[1]}')
            lines.append(f'@set_tile_code({x},{y},"pe.csl",.{{{params}}});')
    # Each tree's receivers are explicit; intermediate rows forward only.
    trees = [(3, 0, 31, 0, 18, 896, 112), (4, 32, 53, 0, 13, 4928, 112),
             (5, 54, 61, 0, 8, 896, 112), (6, 32, 53, 14, 19, 1152, 576),
             (7, 54, 61, 9, 15, 896, 112)]
    for color, left, right, top, bottom, period, extent in trees:
        lines.append(f'@set_color_config({bx+62},{by},@get_color({color}),.{{.routes=.{{.rx=.{{RAMP}},.tx=.{{WEST}}}}}});')
        for y in range(bottom+1):
            for x in range(left, 62 if y == 0 else right+1):
                receiving = x <= right and top <= y <= bottom
                directions = []
                if y == 0 and x > left:
                    directions.append('WEST')
                if x <= right and y < bottom:
                    directions.append('SOUTH')
                if receiving:
                    directions.append('RAMP')
                assert directions
                routes = '.rx=.{' + ('EAST' if y == 0 else 'NORTH') + '},.tx=.{' + ','.join(directions) + '}'
                filtering = ''
                if receiving:
                    shard = x % 8 if color == 3 else (y % 2)*22+x-32 if color == 4 else (x-32)//11 if color == 6 else x-54
                    counter = (period-shard*extent) % period
                    filtering = f',.filter=.{{.kind=.{{.counter=true}},.count_data=true,.init_counter={counter},.max_counter={extent-1},.limit1={period-1}}}'
                lines.append(f'@set_color_config({bx+x},{by+y},@get_color({color}),.{{.routes=.{{{routes}}}{filtering}}});')
    if fragment:
        return '\n'.join(lines) + '\n'
    for name, kind in [('weights', 'u32'), ('input', 'f32'), ('result', 'f32'), ('context', 'f32'),
                       ('activation', 'f32'), ('hidden', 'f32'), ('output', 'f32'), ('progress', 'u32'),
                       ('status', 'u32'), ('timing', 'u16'), ('cache_k', 'f32'), ('cache_v', 'f32')]:
        lines.append(f'@export_name("{name}",[*]{kind},false);')
    if initialization == 'sdk':
        for name, kind in [('identity', 'u32'), ('norms', 'u32'), ('frequency', 'f32'), ('bias', 'u32')]:
            lines.append(f'@export_name("{name}",[*]{kind},false);')
    for name in ['initialize', 'reset_model', 'prepare', 'compute']:
        lines.append(f'@export_name("{name}",fn()void);')
    return '\n'.join(lines + ['}']) + '\n'


def explicit_layer(weights, hidden, layer=0):
    """Independent CPU equations; this function is never called by the SDK worker."""
    import torch
    import torch.nn.functional as functional
    if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < 24:
        raise ValueError('Layer must be in [0,24)')
    prefix = f'model.layers.{layer}.'
    w = lambda name: weights[prefix+name].float()
    norm = lambda x, gamma: x*torch.rsqrt(x.square().mean(-1, keepdim=True)+1e-6)*gamma
    result = {'input_norm': norm(hidden, w('input_layernorm.weight'))}
    for name in ['q', 'k', 'v']:
        result[name] = functional.linear(result['input_norm'], w(f'self_attn.{name}_proj.weight'), w(f'self_attn.{name}_proj.bias'))
    q, k, v = [result[name].reshape(-1, heads, 64).transpose(0, 1) for name, heads in [('q', 14), ('k', 2), ('v', 2)]]
    frequency = 1.0/(1e6**(torch.arange(0, 64, 2).float()/64))
    angles = torch.arange(len(hidden)).float()[:, None]*frequency[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    rotate = lambda x: x*angles.cos()+torch.cat([-x[..., 32:], x[..., :32]], dim=-1)*angles.sin()
    q, k = rotate(q), rotate(k)
    scores = q@k.repeat_interleave(7, dim=0).transpose(-2, -1)*0.125
    mask = torch.arange(len(hidden))[None, :] > torch.arange(len(hidden))[:, None]
    probability = scores.masked_fill(mask, float('-inf')).softmax(-1)
    context = probability@v.repeat_interleave(7, dim=0)
    result['context'] = context.transpose(0, 1).reshape(-1, 896)
    result['o'] = functional.linear(result['context'], w('self_attn.o_proj.weight'))
    result['attention_residual'] = hidden+result['o']
    result['post_norm'] = norm(result['attention_residual'], w('post_attention_layernorm.weight'))
    result['up'] = functional.linear(result['post_norm'], w('mlp.up_proj.weight'))
    result['gate'] = functional.linear(result['post_norm'], w('mlp.gate_proj.weight'))
    result['activation'] = functional.silu(result['gate'])*result['up']
    result['down'] = functional.linear(result['activation'], w('mlp.down_proj.weight'))
    result['output'] = result['attention_residual']+result['down']
    return result, k, v, frequency


def prepare(source_root=None, compile_parallelism=8, layer=0):
    if compile_parallelism < 1:
        raise ValueError('compile_parallelism must be positive')
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, str(ROOT / 'tools'))
    from bf16_transport import pack_bf16_pairs
    source_root = ROOT if source_root is None else source_root
    sys.path.insert(0, str(source_root / 'src'))
    from csl_llm.decoder_layout import model_origin
    model_origin(layer)  # Validate before loading model data.
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    model_dir = ROOT / 'models/qwen2.5-0.5b-7ae5576'
    reference = ROOT / 'evidence/reference-f32-greedy-20260908T0554'
    trace = reference / 'arithmetic/step-000.npz'
    source = np.load(trace)
    state = load_file(model_dir / 'model.safetensors')
    report = json.loads((reference / 'report.json').read_text())
    english = next(row for row in report['prompts'] if row['id'] == 'english')
    changed_ids = english['prompt_tokens'][-8:-6]
    input_key = 'model.embed_tokens' if layer == 0 else f'model.layers.{layer-1}'
    canonical = torch.from_numpy(source[input_key][0, :3].copy())
    prefix = f'model.layers.{layer}'
    mapping = dict(input_norm='input_layernorm', q='self_attn.q_proj', k='self_attn.k_proj', v='self_attn.v_proj',
                   o='self_attn.o_proj', post_norm='post_attention_layernorm', up='mlp.up_proj', gate='mlp.gate_proj', down='mlp.down_proj')
    official_changed = {}
    official_model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, attn_implementation='eager', local_files_only=True).eval()
    cfg = official_model.config
    assert (cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads) == (896, 4864, 14, 2)
    assert cfg.rms_norm_eps == 1e-6 and cfg.rope_theta == 1e6
    handles = []
    def hook(name):
        def capture(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            official_changed[name] = value.detach()[0].clone()
        return capture
    def capture_input(module, args):
        official_changed['layer_input'] = args[0].detach()[0].clone()
    for name, module in official_model.named_modules():
        if name == prefix:
            handles.append(module.register_forward_pre_hook(capture_input))
        if name == prefix or name in {prefix+'.'+suffix for suffix in mapping.values()}:
            handles.append(module.register_forward_hook(hook(name)))
    with torch.inference_mode():
        official_model(input_ids=torch.tensor(changed_ids)[None, :], use_cache=True, return_dict=True)
    for handle in handles:
        handle.remove()
    del official_model
    changed = official_changed['layer_input']
    if layer == 0:
        assert torch.equal(changed, state['model.embed_tokens.weight'][changed_ids].float())
    assert not torch.equal(canonical[:2], changed)
    references, cache_k, cache_v, inputs, cases, cpu_checks, official_outputs = [], [], [], [], [], [], []
    for sequence, hidden in enumerate([canonical, changed]):
        with torch.inference_mode():
            nodes, keys, values, frequency = explicit_layer(state, hidden, layer)
        for name in list(mapping)+['output']:
            key = prefix if name == 'output' else prefix+'.'+mapping[name]
            expected = torch.from_numpy(source[key][0, :3].copy()) if sequence == 0 else official_changed[key]
            delta = nodes[name].double()-expected.double()
            absolute = float(delta.abs().max())
            relative = float(torch.linalg.vector_norm(delta)/max(float(torch.linalg.vector_norm(expected.double())), 1e-30))
            peak = absolute/max(float(expected.abs().max()), 1e-30)
            assert absolute <= 1e-5 or (relative <= 2e-5 and peak <= 3e-5), (sequence, name, absolute, relative)
            cpu_checks.append(dict(sequence=sequence, stage=name, max_abs=absolute, relative_l2=relative, relative_peak=peak))
        official_layer = torch.from_numpy(source[prefix][0, :3].copy()) if sequence == 0 else official_changed[prefix]
        for position in range(len(hidden)):
            inputs.append(hidden[position].numpy())
            official_outputs.append(official_layer[position].numpy())
            references.append({name: value[position].numpy() for name, value in nodes.items()})
            cache_k.append(keys[:, :position+1].numpy())
            cache_v.append(values[:, :position+1].numpy())
            cases.append(dict(sequence=sequence, position=position, reset=position == 0,
                              cache_read=position == len(hidden)-1,
                              label=f'{"canonical-prefix" if sequence == 0 else "changed-reset"}-{position}'))
    out = ROOT / 'evidence' / ('resident-decoder-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    data = {'inputs': np.stack(inputs), 'official_output': np.stack(official_outputs)}
    for name in references[0]:
        data['expected_'+name] = np.stack([row[name] for row in references])
    for index, (key, value) in enumerate(zip(cache_k, cache_v)):
        data[f'expected_cache_k_{index}'] = key
        data[f'expected_cache_v_{index}'] = value
    regions = geometry()
    def literal(name, words):
        return 'const '+name+f'=[{len(words)}]u32'+'{'+','.join(f'0x{int(word):08x}' for word in words)+'};\n'
    for name, rs in regions.items():
        if name == 'kv':
            continue
        tensor = f'{prefix}.mlp.{name}_proj.weight' if name in ('up', 'gate', 'down') else f'{prefix}.self_attn.{name}_proj.weight'
        weight = state[tensor].float().numpy()
        for group, region in enumerate(rs):
            for ordinal in range(region.participants):
                x, y = region.point(ordinal)
                shard = ordinal if name != 'down' or ordinal < 22 else 65-ordinal
                valid = min(112, weight.shape[1]-shard*112)
                tile = np.zeros((112, 128), np.uint16)
                tile[:valid] = (weight[group*128:(group+1)*128, shard*112:shard*112+valid].T.copy().view(np.uint32)>>16).astype(np.uint16)
                data[f'weights_{x}_{y}'] = tile
                (out / f'weights_{x}_{y}.csl').write_text(literal('values', pack_bf16_pairs(tile)))
            if name in ('q', 'k', 'v'):
                bias = state[f'{prefix}.self_attn.{name}_proj.bias'][group*128:(group+1)*128].float().numpy()
                (out / f'bias_{name}_{group}.csl').write_text(literal('bias', pack_bf16_pairs((bias.view(np.uint32)>>16).astype(np.uint16))))
    norms = torch.cat([state[f'{prefix}.{name}.weight'].float() for name in ['input_layernorm', 'post_attention_layernorm']]).numpy()
    auxiliary = literal('norms', pack_bf16_pairs((norms.view(np.uint32)>>16).astype(np.uint16)))
    auxiliary += 'const frequency=[32]f32{'+','.join(f'@bitcast(f32,@as(u32,0x{int(word):08x}))' for word in frequency.numpy().view(np.uint32))+'};\n'
    (out / 'controller_aux.csl').write_text(auxiliary)
    np.savez(out / 'inputs.npz', **data)
    files = [(Path(__file__), 'driver.py'), (source_root / 'tools/sdk_probe.py', 'executor.py'),
             (source_root / 'csl/programs/resident_decoder/pe.csl', 'pe.csl'),
             (source_root / 'csl/kernels/local_gemv_bf16_f32_colmajor.csl', 'kernel.csl'),
             (source_root / 'csl/kernels/vector_f32.csl', 'vector.csl'),
             (source_root / 'csl/kernels/qwen_position_f32.csl', 'rope.csl'),
             (source_root / 'csl/kernels/kv_block_f32.csl', 'kv.csl'),
             (source_root / 'csl/runtime/line_allreduce_f32.csl', 'line.csl'),
             (source_root / 'csl/runtime/filtered_input.csl', 'input.csl'),
             (source_root / 'csl/runtime/packets.csl', 'packets.csl'),
             (source_root / 'csl/runtime/activation_stream.csl', 'activation_stream.csl'),
             (source_root / 'configs/precision.json', 'precision.json')]
    files += [(source_root / 'src/csl_llm/decoder_layout.py', 'builder_decoder_layout.py'),
              (source_root / 'src/csl_llm/regions.py', 'builder_regions.py')]
    for source_file, target in files:
        shutil.copy2(source_file, out / target)
    (out / 'layout.csl').write_text(make_layout(regions))
    config = dict(cases=cases, layer=layer, canonical_input_key=input_key,
                  changed_token_ids=changed_ids, compile_parallelism=compile_parallelism,
                  model_sha256=digest(model_dir / 'model.safetensors'),
                  model_config_sha256=digest(model_dir / 'config.json'), trace_sha256=digest(trace), reference_report_sha256=digest(reference / 'report.json'), cpu_checks=cpu_checks,
                  roots={name: [list(r.point(0)) for r in rs] for name, rs in regions.items()},
                  scope=f'Complete native layer{layer} with1044 original weight tiles, biases/norms/RoPE,132 persistent KV PEs and device-only intermediate computation. Three canonical prefix positions, reset to two changed token IDs. Selected-layer fixture inputs are CPU reference activations; this is not24layer inference/full vocabulary/2048model trajectory.')
    (out / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',
                                                       files={p.name: digest(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform
    verify(root)
    os.chdir(root)
    config = json.loads((root / 'config.json').read_text())
    policy = json.loads((root / 'precision.json').read_text())['stage_gate']['absolute_or_normwise']
    data = np.load(root / 'inputs.npz')
    start = time.monotonic()
    def stage(name):
        event = dict(stage=name, elapsed_seconds=time.monotonic()-start)
        (root / 'stage.json').write_text(json.dumps(event)+'\n')
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(name, flush=True)
    command = ['cslc', 'layout.csl', '--arch=wse3', '--fabric-dims=71,22', '--fabric-offsets=4,1',
               '-o=out', '--memcpy', '--channels=1', '--dump-dsr-alloc-graph']
    # Public pinned cslc-driver option. This bounds compiler jobs, not simulator
    # threads; memory improvement must be measured on a new frozen run.
    parallelism = config.get('compile_parallelism')
    if parallelism is not None:
        if not isinstance(parallelism, int) or isinstance(parallelism, bool) or parallelism < 1:
            raise ValueError('Invalid frozen compiler parallelism')
        command.append(f'--max-parallelism={parallelism}')
    (root / 'compile-command.json').write_text(json.dumps(command)+'\n')
    stage('compile')
    subprocess.run(command, check=True)
    runner = SdkRuntime('out', get_platform(None, SimfabConfig(suppress_trace=True, num_threads=8, dump_core=True), SdkTarget.WSE3))
    ids = {name: runner.get_id(name) for name in ['input', 'result', 'context', 'activation', 'hidden', 'output',
                                                'progress', 'status', 'timing', 'cache_k', 'cache_v']}
    stage('sdk-load')
    runner.load()
    stage('sdk-run')
    runner.run()
    def read(name, x, y, width, height, count):
        values = np.zeros(width*height*count, np.uint32 if name in ('progress', 'status', 'timing') else np.float32)
        # Preserve precise API boundaries if a later diagnostic read times out.
        api_start = time.monotonic()
        def record_api(event):
            with (root / 'api-events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(event=event, operation='memcpy_d2h', symbol=name,
                                             rectangle=[x, y, width, height], count_per_pe=count,
                                             elapsed_seconds=time.monotonic()-start,
                                             call_seconds=time.monotonic()-api_start))+'\n')
        record_api('begin')
        runner.memcpy_d2h(values, ids[name], x, y, width, height, count, streaming=False,
                          data_type=MemcpyDataType.MEMCPY_16BIT if name == 'timing' else MemcpyDataType.MEMCPY_32BIT,
                          order=MemcpyOrder.ROW_MAJOR, nonblock=False)
        record_api('end')
        return values
    def check(actual, expected):
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        actual, expected = actual.astype(np.float64), expected.astype(np.float64)
        difference = actual-expected
        absolute = float(np.max(np.abs(difference)))
        norm = float(np.linalg.norm(difference)/max(np.linalg.norm(expected), 1e-30))
        peak = absolute/max(float(np.max(np.abs(expected))), 1e-30)
        assert absolute <= policy['max_abs'] or (norm <= policy['both']['relative_l2'] and peak <= policy['both']['relative_peak']), (absolute, norm, peak)
        return dict(max_abs=absolute, relative_l2=norm, relative_peak=peak)
    reports = []
    try:
        stage('initialize')
        runner.launch('initialize', nonblock=False)
        for index, case in enumerate(config['cases']):
            label = case['label']
            if case['reset']:
                stage(label+'-reset')
                runner.launch('reset_model', nonblock=False)
            stage(label+'-input')
            runner.memcpy_h2d(ids['input'], data['inputs'][index].copy(), 62, 0, 1, 1, 896,
                              streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
            stage(label+'-prepare')
            runner.launch('prepare', nonblock=False)
            stage(label+'-read-ready')
            ready = read('progress', 0, 0, 64, 20, 1)
            np.save(root / f'ready-{index}.npy', ready)
            np.testing.assert_array_equal(ready, np.full(1280, 2*index+1, np.uint32))
            stage(label+'-compute')
            runner.launch('compute', nonblock=False)
            stage(label+'-read-output')
            actual = {'output': read('output', 62, 0, 1, 1, 896), 'ready': ready}
            np.save(root / f'output-{index}.npy', actual['output'])
            stage(label+'-diagnostics')
            actual['progress'] = read('progress', 0, 0, 64, 20, 1)
            np.save(root / f'progress-{index}.npy', actual['progress'])
            actual['timing'] = read('timing', 62, 0, 1, 1, 6)
            actual['controller_status'] = read('status', 62, 0, 1, 1, 2)
            actual['activation_frame'] = read('activation', 62, 0, 1, 1, 4928)
            actual['activation'] = actual['activation_frame'][:4864]
            actual['attention_residual'] = read('hidden', 62, 0, 1, 1, 896)
            actual['post_norm'] = read('input', 62, 0, 1, 1, 896)
            actual['input_norm'] = read('input', 54, 0, 8, 1, 112)
            projection = read('result', 54, 0, 1, 16, 128).reshape(16, 128)
            actual['qkvo_column'] = projection
            actual['q'], actual['k'], actual['v'], actual['o'] = projection[:7].ravel(), projection[7], projection[8], projection[9:].ravel()
            for name in ['up', 'gate', 'down']:
                height = 14 if name == 'down' else 19
                columns = {}
                for x in sorted({p[0] for p in config['roots'][name]}):
                    columns[x] = read('result', x, 0, 1, height, 128).reshape(height, 128)
                    actual[f'{name}_column_{x}'] = columns[x]
                actual[name] = np.concatenate([columns[x][y] for x, y in config['roots'][name]])
            contexts = []
            for head, (x, y) in enumerate(config['roots']['kv']):
                contexts.append(read('context', x, y, 1, 1, 448))
                actual[f'kv_status_{head}'] = read('status', x, y, 11, 6, 2).reshape(6, 11, 2)
                expected_status = np.zeros((6, 11, 2), np.uint32)
                expected_status[:, :, 0] = case['position']+1
                for stripe in range(66):
                    expected_status[stripe//11, stripe % 11, 1] = min(32, max(0, case['position']+1-stripe*32))
                np.testing.assert_array_equal(actual[f'kv_status_{head}'], expected_status)
                if case['cache_read']:
                    actual[f'cache_k_{head}'] = read('cache_k', x, y, 1, 1, 2048).reshape(64, 32)
                    actual[f'cache_v_{head}'] = read('cache_v', x, y, 1, 1, 2048).reshape(32, 64)
            actual['context'] = np.concatenate(contexts)
            file = root / f'actual-{index}.npz'
            np.savez(file, **actual)
            np.testing.assert_array_equal(actual['progress'], np.full(1280, 2*index+2, np.uint32))
            assert actual['controller_status'][0] == case['position']+1
            assert np.all(actual['activation_frame'][4864:].view(np.uint32) == 0)
            names = ['input_norm', 'q', 'k', 'v', 'context', 'o', 'attention_residual', 'post_norm', 'up', 'gate', 'activation', 'down', 'output']
            checks = {name: check(actual[name], data['expected_'+name][index]) for name in names}
            checks['official_output'] = check(actual['output'], data['official_output'][index])
            if case['cache_read']:
                for head in range(2):
                    visible = case['position']+1
                    checks[f'cache_k_{head}'] = check(actual[f'cache_k_{head}'][:, :visible].T, data[f'expected_cache_k_{index}'][head])
                    checks[f'cache_v_{head}'] = check(actual[f'cache_v_{head}'][:visible], data[f'expected_cache_v_{index}'][head])
            t = actual['timing'].astype(np.uint64)
            assert np.all(t < 65536)
            ticks = lambda a: int(a[0])+(int(a[1]) << 16)+(int(a[2]) << 32)
            cycles = (ticks(t[3:])-ticks(t[:3])) & ((1 << 48)-1)
            assert 0 < cycles < (1 << 40)
            reports.append(dict(case=case, checks=checks, controller_cycles=cycles, actual_sha256=digest(file)))
            (root / 'results.json').write_text(json.dumps(dict(success=False, cases=reports), indent=2)+'\n')
    finally:
        runner.stop()
    (root / 'results.json').write_text(json.dumps(dict(success=True, cases=reports, scope=config['scope']), indent=2)+'\n')
    stage('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--execute', type=Path)
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--compile-parallelism', type=int, default=8)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--timeout', type=int, default=2700)
    args = parser.parse_args()
    if args.project_root:
        ROOT = args.project_root.resolve()
        sys.path.insert(0, str(ROOT / 'src'))
    if args.prepare:
        prepare(args.source_root.resolve() if args.source_root else None, args.compile_parallelism, args.layer)
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        if args.timeout <= 0:
            parser.error('--timeout must be positive')
        execute(args.execute.resolve(), args.timeout)
