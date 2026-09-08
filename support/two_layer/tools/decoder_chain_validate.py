"""Validate actual stopped-core two-layer arithmetic and drained transport.

Only position zero is accepted here. The primary layer-one oracle is frozen
independently from layer-zero reference output, never conditioned on SDK output.
"""
import json


def validate(core, expected_initial, reference, normal, policy, output):
    import numpy as np
    from csl_llm.decoder_layout import tiles, geometry
    from decoder_chain_symbols import requirements
    owners = {(tile.x, tile.y): tile for layer in (0, 1) for tile in tiles(layer)}
    controllers = {(62, 0), (126, 0)}
    actual, checks = {}, {}

    def read(point, name, dtype, count):
        dtype = np.dtype(dtype)
        return np.frombuffer(core.read(point[0]+4, point[1]+1, name, dtype.itemsize*count), dtype=dtype).copy()

    def compare(name, values, expected):
        expected = np.asarray(expected).reshape(-1)
        values = np.asarray(values).reshape(-1)
        if values.shape != expected.shape or not np.isfinite(values).all() or not np.isfinite(expected).all():
            raise ValueError('Nonfinite or invalid stage: '+name)
        error = values.astype(np.float64)-expected.astype(np.float64)
        absolute = float(np.max(np.abs(error)))
        norm = float(np.linalg.norm(error)/max(float(np.linalg.norm(expected.astype(np.float64))), 1e-30))
        peak = absolute/max(float(np.max(np.abs(expected))), 1e-30)
        metric = dict(max_abs=absolute, relative_l2=norm, relative_peak=peak)
        checks[name] = metric
        if not (absolute <= policy['max_abs'] or
                (norm <= policy['both']['relative_l2'] and peak <= policy['both']['relative_peak'])):
            raise ValueError(f'Stage exceeds frozen gate: {name}: {metric}')

    progress = np.empty((20, 129), np.uint32)
    protocols = []
    # Read and preserve diagnostics before evaluating arithmetic tolerances.
    for point, initial in sorted(expected_initial.items()):
        x, y = point
        progress[y, x] = read(point, 'progress', '<u4', 1)[0]
        for name in ('weights', 'norms', 'frequency', 'bias'):
            if name in initial and core.read(x+4, y+1, name, len(initial[name])) != initial[name]:
                raise ValueError(f'Immutable original state changed: {point} {name}')
        if point != (128, 0):
            tile = owners.get(point)
            expected_identity = [tile.packet_group, *tile.destination] if tile else [0, 0, 0]
            if read(point, 'identity', '<u4', 3).tolist() != expected_identity:
                raise ValueError(f'Actual initialized identity differs: {point}')
        fields = requirements(point, owners.get(point), point in controllers)
        protocol = {name: int(read(point, name, '<u4' if size == 4 else '<u2', 1)[0])
                    for name, (size, _) in fields.items() if name.startswith(('boundary.', 'packets.'))}
        if protocol:
            protocols.append(dict(coordinate=list(point), state=protocol))
    actual['progress'] = progress
    actual['endpoint_output'] = read((128, 0), 'output', '<f4', 896)
    actual['endpoint_input'] = read((128, 0), 'input', '<f4', 896)
    actual['endpoint_status'] = read((128, 0), 'status', '<u4', 2)
    actual['endpoint_timing'] = read((128, 0), 'timing', '<u2', 6)
    for layer in (0, 1):
        bx, control = layer*64, (layer*64+62, 0)
        prefix = f'layer_{layer}_'
        for name, symbol in [('output', 'output'), ('attention_residual', 'hidden'), ('post_norm', 'input')]:
            actual[prefix+name] = read(control, symbol, '<f4', 896)
        actual[prefix+'activation_frame'] = read(control, 'activation', '<f4', 4928)
        actual[prefix+'activation'] = actual[prefix+'activation_frame'][:4864]
        actual[prefix+'input_norm'] = np.concatenate([read((bx+54+x, 0), 'input', '<f4', 112) for x in range(8)])
        column = np.stack([read((bx+54, y), 'result', '<f4', 128) for y in range(16)])
        for name, values in [('q', column[:7].ravel()), ('k', column[7]), ('v', column[8]), ('o', column[9:].ravel())]:
            actual[prefix+name] = values
        regions = geometry((bx, 0))
        for name in ('up', 'gate', 'down'):
            actual[prefix+name] = np.concatenate([read(region.point(0), 'result', '<f4', 128) for region in regions[name]])
        actual[prefix+'context'] = np.concatenate([read(region.point(0), 'context', '<f4', 448) for region in regions['kv']])
        actual[prefix+'status'] = read(control, 'status', '<u4', 2)
        actual[prefix+'timing'] = read(control, 'timing', '<u2', 6)
        for head, region in enumerate(regions['kv']):
            # Physical stripe order, distinct from the collective snake order.
            points = [(region.x+s%11, region.y+s//11) for s in range(66)]
            for label, symbol, dtype, count in [('cache_k', 'kv.keys', '<f4', 2048),
                    ('cache_v', 'kv.values', '<f4', 2048), ('kv_status', 'status', '<u4', 2),
                    ('kv_seen', 'kv.seen', '<u2', 1), ('kv_valid', 'kv.valid', '<u2', 1)]:
                actual[f'{prefix}{label}_{head}'] = np.stack([read(point, symbol, dtype, count) for point in points])
    np.savez(output / 'actual.npz', **actual)
    (output / 'protocol.json').write_text(json.dumps(protocols, indent=2)+'\n')
    np.testing.assert_array_equal(progress, np.full((20, 129), 2, np.uint32))
    np.testing.assert_array_equal(normal['ready'], np.ones(2580, np.uint32))
    np.testing.assert_array_equal(normal['progress'].reshape(20, 129), progress)
    assert actual['endpoint_output'].tobytes() == normal['output'].tobytes()
    assert actual['endpoint_input'].tobytes() == reference['input'].tobytes()
    assert actual['endpoint_output'].tobytes() == actual['layer_1_output'].tobytes()
    np.testing.assert_array_equal(actual['endpoint_status'], [1, 1])
    for row in protocols:
        state = row['state']
        assert state['packets.sending'] == state['packets.receiving'] == 0, row
        if 'boundary.invocation' in state:
            assert state == {'packets.sending': 0, 'packets.receiving': 0,
                'boundary.invocation': 1, 'boundary.received_count': 896,
                'boundary.sent_count': 896, 'boundary.pending_count': 0,
                'boundary.command': 1, 'boundary.started': 1, 'boundary.sending': 0}, row
    stages = ('input_norm', 'q', 'k', 'v', 'context', 'o', 'attention_residual',
              'post_norm', 'up', 'gate', 'activation', 'down', 'output')
    for layer in (0, 1):
        prefix = f'layer_{layer}_'
        np.testing.assert_array_equal(actual[prefix+'status'], [1, 0])
        assert not actual[prefix+'activation_frame'][4864:].view(np.uint32).any()
        for name in stages:
            compare(prefix+name, actual[prefix+name], reference[prefix+name])
        for head in range(2):
            np.testing.assert_array_equal(actual[f'{prefix}kv_seen_{head}'], np.ones((66, 1), np.uint16))
            valid = np.zeros((66, 1), np.uint16); valid[0, 0] = 1
            np.testing.assert_array_equal(actual[f'{prefix}kv_valid_{head}'], valid)
            np.testing.assert_array_equal(actual[f'{prefix}kv_status_{head}'], np.column_stack([np.ones(66, np.uint32), valid[:, 0]]))
            keys = actual[f'{prefix}cache_k_{head}'].reshape(66, 64, 32)
            values = actual[f'{prefix}cache_v_{head}'].reshape(66, 32, 64)
            compare(f'{prefix}cache_k_{head}', keys[0, :, 0], reference[prefix+'cache_k'][head, 0])
            compare(f'{prefix}cache_v_{head}', values[0, 0], reference[prefix+'cache_v'][head, 0])
            assert not keys[0, :, 1:].view(np.uint32).any() and not keys[1:].view(np.uint32).any()
            assert not values[0, 1:].view(np.uint32).any() and not values[1:].view(np.uint32).any()
    cycles = {}
    for key in ('endpoint_timing', 'layer_0_timing', 'layer_1_timing'):
        timing = actual[key]
        ticks = [sum(int(timing[k+i]) << (16*i) for i in range(3)) for k in (0, 3)]
        cycles[key] = (ticks[1]-ticks[0]) & ((1 << 48)-1)
        assert 0 < cycles[key] < 1 << 40
    return dict(passed=True, checks=checks, cycles=cycles,
        cycle_scope='Endpoint interval covers both layers. Layer controller clocks include upstream waiting; do not add or interpret as isolated compute.',
        scope='Original single position-zero two-layer device chain, all initializers, completion, KV and protocol states. Not cached/reset or full-model inference.')
