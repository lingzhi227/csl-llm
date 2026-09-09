"""Actual full-model stopped-core checks, with explicit reset cache history.

The caller must bind the actual core and diagnostic symbol contract to the
complete original-weight assembly before entering this validator. Initializers
are expectations only and are never used to supply missing core bytes.
"""
import json
from pathlib import Path
import numpy as np
from model_step_validate import compare


def physical_cache(history, kind):
    """Construct expected physical SRAM after whole-prefix overwrites by resets.

    History contains independent original-input references in request order,
    ending in the current request. Visibility is checked separately; retained
    older slots are not allowed to affect attention merely because they exist.
    """
    if not history or kind not in ('key', 'value'):
        raise ValueError('Complete independent cache history is required')
    expected = np.zeros((24, 2, 2112, 64), np.float32)
    written = 0
    for reference in history:
        data = np.asarray(reference['cache_'+kind])
        if np.asarray(reference['cache_key']).shape != np.asarray(reference['cache_value']).shape:
            raise ValueError('K/V visibility must agree within each request')
        if data.dtype != np.float32 or data.ndim != 4 or data.shape[:2] != (24, 2) or data.shape[3] != 64 or not 1 <= data.shape[2] <= 2048 or not np.isfinite(data).all():
            raise ValueError('Invalid independent full-model KV history')
        expected[:, :, :data.shape[2]] = data
        written = max(written, data.shape[2])
    return expected, written


def validate(core, initializers, normal, history, *, invocation, precision, output):
    from csl_llm.decoder_layout import tiles, model_origin, geometry
    from csl_llm.coordinate_identity import decoder_identity, vocabulary_identity
    from decoder_chain_symbols import requirements
    from model_reference import DECODER_STAGES
    if type(invocation) is not int or not 1 <= invocation <= 65535:
        raise ValueError('Invalid final core epoch')
    output = Path(output)
    owners = {(t.x, t.y): t for layer in range(24) for t in tiles(layer)}
    controllers = {(model_origin(layer)[0]+62, model_origin(layer)[1]) for layer in range(24)}
    seen, protocols, checks, arrays = set(), [], {}, {}
    progress = np.empty((210, 193), np.uint32)
    universe = {(x, y) for y in range(210) for x in range(193)}

    def read(point, name, dtype, count):
        dtype = np.dtype(dtype)
        return np.frombuffer(core.read(point[0]+4, point[1]+1, name, count*dtype.itemsize), dtype=dtype).copy()

    for point, fields in initializers:
        if point in seen or point not in universe:
            raise ValueError('Repeated or unexpected original initializer coordinate')
        seen.add(point)
        x, y = point
        if 'weights' not in fields:
            raise ValueError('Every PE must bind its original or declared padding allocation')
        for name in ('weights', 'norms', 'frequency', 'bias'):
            if name in fields and core.read(x+4, y+1, name, len(fields[name])) != fields[name]:
                raise ValueError(f'Original immutable SRAM differs at {point}: {name}')
        progress[y, x] = read(point, 'progress', '<u4', 1)[0]
        tile = owners.get(point)
        protocol = {}
        if y < 160:
            identity = decoder_identity(tile.operation, x, y) if tile else (0, 0, 0)
            np.testing.assert_array_equal(read(point, 'identity', '<u4', 3), identity)
            spec = requirements(point, tile, point in controllers, endpoint=False)
            protocol = {name: int(read(point, name, '<u4' if size == 4 else '<u2', 1)[0])
                        for name, (size, _) in spec.items() if name.startswith(('packets.', 'boundary.'))}
        else:
            matrix = x < 192 and (y-160)*192+x < 9496
            np.testing.assert_array_equal(read(point, 'identity', '<u4', 1),
                                          [vocabulary_identity(x, y) if matrix else 0])
            if matrix or point == (192, 160):
                protocol = {name: int(read(point, name, '<u2', 1)[0])
                            for name in ('packets.sending', 'packets.receiving')}
            if point == (192, 160):
                protocol['boundary.invocation'] = int(read(point, 'boundary.invocation', '<u4', 1)[0])
                for name in ('received_count', 'sent_count', 'pending_count', 'command', 'started', 'sending'):
                    protocol['boundary.'+name] = int(read(point, 'boundary.'+name, '<u2', 1)[0])
        if protocol:
            protocols.append(dict(coordinate=point, state=protocol))
    if seen != universe:
        raise ValueError('Incomplete original model initializer coverage')
    arrays['progress'] = progress
    (output/'protocol.json').write_text(json.dumps(protocols, indent=2)+'\n')
    np.testing.assert_array_equal(progress, np.full((210, 193), invocation*2, np.uint32))
    np.testing.assert_array_equal(progress, normal['progress'])
    for row in protocols:
        state = row['state']
        if state['packets.sending'] or state['packets.receiving']:
            raise ValueError(f'Undrained packet state: {row}')
        if 'boundary.invocation' in state:
            wanted = {'packets.sending': 0, 'packets.receiving': 0,
                'boundary.invocation': invocation, 'boundary.received_count': 896,
                'boundary.sent_count': 896, 'boundary.pending_count': 0,
                'boundary.command': 1, 'boundary.started': 1, 'boundary.sending': 0}
            if state != wanted:
                raise ValueError(f'Incomplete layer boundary: {row}')
    reference = history[-1]
    consumed = reference['cache_key'].shape[2]
    if normal['summary'][0] != consumed or invocation < consumed:
        raise ValueError('Final reference visibility differs from the actual request')
    gate = precision['stage_gate']['absolute_or_normwise']
    for layer in range(24):
        x, y = model_origin(layer)
        value = read((x+62, y), 'output', '<f4', 896)
        arrays[f'layer_{layer}_output'] = value
        if value.tobytes() != normal['layer_outputs'][layer].tobytes():
            raise ValueError('Normal/core layer output bits differ')
        checks[f'layer_{layer}'] = compare(value, reference['layer_outputs'][layer], gate)
        np.testing.assert_array_equal(read((x+62, y), 'status', '<u4', 2), normal['controller_status'][layer])
        timing = read((x+62, y), 'timing', '<u2', 6)
        np.testing.assert_array_equal(timing.astype(np.uint32), normal['controller_timing'][layer])
        arrays[f'layer_{layer}_timing'] = timing
        if f'layer_{layer}_input_norm' in reference:
            stage_values = dict(output=value,
                attention_residual=read((x+62, y), 'hidden', '<f4', 896),
                post_norm=read((x+62, y), 'input', '<f4', 896))
            frame = read((x+62, y), 'activation', '<f4', 4928)
            if frame[4864:].view(np.uint32).any():
                raise ValueError('Nonzero decoder activation padding')
            stage_values['activation'] = frame[:4864]
            stage_values['input_norm'] = np.concatenate([read((x+54+shard, y), 'input', '<f4', 112) for shard in range(8)])
            column = np.stack([read((x+54, y+row), 'result', '<f4', 128) for row in range(16)])
            stage_values.update(q=column[:7].ravel(), k=column[7], v=column[8], o=column[9:].ravel())
            regions = geometry((x, y))
            for name in ('up', 'gate', 'down'):
                stage_values[name] = np.concatenate([read(region.point(0), 'result', '<f4', 128) for region in regions[name]])
            stage_values['context'] = np.concatenate([read(region.point(0), 'context', '<f4', 448) for region in regions['kv']])
            for name in DECODER_STAGES:
                key = f'layer_{layer}_{name}'
                arrays[key] = stage_values[name]
                checks[key] = compare(stage_values[name], reference[key], gate)
    for kind, symbol in [('key', 'kv.keys'), ('value', 'kv.values')]:
        wanted, written = physical_cache(history, kind)
        actual = np.empty_like(wanted)
        for layer in range(24):
            x, y = model_origin(layer)
            for head in range(2):
                for stripe in range(66):
                    point = (x+32+head*11+stripe % 11, y+14+stripe//11)
                    data = read(point, symbol, '<f4', 2048)
                    actual[layer, head, stripe*32:(stripe+1)*32] = data.reshape(64, 32).T if kind == 'key' else data.reshape(32, 64)
                    valid = min(32, max(0, consumed-stripe*32))
                    np.testing.assert_array_equal(read(point, 'status', '<u4', 2), [consumed, valid])
                    np.testing.assert_array_equal(read(point, 'kv.seen', '<u2', 1), [consumed])
                    np.testing.assert_array_equal(read(point, 'kv.valid', '<u2', 1), [valid])
        arrays['physical_cache_'+kind] = actual
        for layer in range(24):
            for head in range(2):
                # Check each token separately so a long cache cannot dilute a
                # damaged slot through a large aggregate norm.
                metrics = [compare(actual[layer, head, position], wanted[layer, head, position], gate)
                           for position in range(written)]
                checks[f'cache_{kind}_{layer}_{head}'] = dict(positions=written,
                    **{name: max(row[name] for row in metrics) for name in ('max_abs', 'relative_l2', 'relative_peak')})
        if actual[:, :, written:].view(np.uint32).any():
            raise ValueError('Unwritten KV capacity or padding differs from original zero bytes')
    point = (192, 160)
    for name, symbol, count, dtype in [('summary', 'summary', 7, '<u4'), ('control', 'control', 2, '<u4'),
            ('model_input', 'input', 896, '<f4'), ('prompt', 'sequence.prompt', 2048, '<u4'),
            ('generated', 'sequence.generated', 256, '<u4'), ('model_timing', 'timing', 6, '<u2')]:
        value = read(point, symbol, dtype, count)
        if name in ('prompt', 'generated'):
            # The reader requires the exact audited object size. Read the
            # whole physical allocation, then compare only its visible prefix.
            arrays['physical_'+name] = value
            if len(history) == 1 and value[len(normal[name]):].any():
                raise ValueError('Fresh request changed unused token storage')
            value = value[:len(normal[name])]
        arrays[name] = value
        np.testing.assert_array_equal(value, normal[name])
        if dtype == '<f4' and value.tobytes() != normal[name].tobytes():
            raise ValueError('Model input normal/core bits differ')
    if 'logits' in normal:
        copies = np.stack([read((tile % 192, 160+tile//192), 'result', '<f4', 128)
                           for tile in range(9496)]).reshape(1187, 8, 128)
        if not np.all(copies.view(np.uint32) == copies[:, :1].view(np.uint32)):
            raise ValueError('Actual core vocabulary allreduce participants differ')
        logits = copies[:, 0].reshape(151936).copy()
        arrays['logits'] = logits
        if logits.tobytes() != normal['logits'].tobytes():
            raise ValueError('Normal/core full vocabulary bits differ')
        checks['logits'] = compare(logits, reference['logits'], precision['logit_gate']['all_required'], all_required=True)
        for name, dtype in [('winner', '<u4'), ('best', '<f4')]:
            arrays[name] = read(point, name, dtype, 1)
            np.testing.assert_array_equal(arrays[name], normal[name])
    np.savez(output/'actual.npz', **arrays)
    return dict(passed=True, checks=checks, coordinates=len(seen),
        scope='Actual final core vs normal outputs, original initializers, independent layer/KV/logit references and drained transport. Scope/capacity is limited to the bound executed request history.')
