"""Bind frozen data declarations to typed full-model CSL initializer fields.

Artifact integrity and role matching only. Original-byte review and actual
ELF/SDK qualification remain separate prerequisites, never inferred here.
"""
import json
from pathlib import Path

from audit_model_weight_mapping import digest


def load_bindings(root):
    from csl_llm.decoder_layout import tiles, controller_parameters
    from csl_llm.static_data import StaticData
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if Path(name).name != name or digest(root / name) != expected:
            raise ValueError(f'Changed static input: {name}')
    config = json.loads((root / 'config.json').read_text())
    if not config['mapping_audit']['passed'] or config['mapping_audit']['matrix_tiles'] != 34552:
        raise ValueError('A complete original pack mapping audit is required')
    owners = {(tile.x, tile.y): tile for layer in range(24)
              for tile in tiles(layer) if tile.operation != 'kv'}
    auxiliaries = {}
    for row in config['auxiliaries']:
        point = tuple(row['coordinate'])
        if point in auxiliaries:
            raise ValueError('Repeated auxiliary owner')
        auxiliaries[point] = row
    expected_aux = {}
    for layer in range(24):
        controller = controller_parameters(layer)
        expected_aux[tuple(controller['coordinate'])] = dict(
            kind='decoder_norms_frequency', file=f'controller_aux_{layer}.csl',
            tensors=list(controller['norms']))
    for point, tile in owners.items():
        if tile.bias_tensor:
            expected_aux[point] = dict(kind='bias', file=f'bias_{point[0]}_{point[1]}.csl',
                tensor=tile.bias_tensor, row=tile.row)
    expected_aux[192, 160] = dict(kind='final_norm', file='final_norm.csl', tensor='model.norm.weight')
    if set(auxiliaries) != set(expected_aux) or len(auxiliaries) != 241:
        raise ValueError('All original auxiliary declarations are required')
    for point, row in auxiliaries.items():
        if any(row.get(key) != value for key, value in expected_aux[point].items()):
            raise ValueError(f'Auxiliary mapping differs at {point}')
    result = {point: StaticData(auxiliary_file=row['file']) for point, row in auxiliaries.items()
              if row['kind'] != 'bias'}
    matrices = config['matrices']
    if not 1 <= len(matrices) <= 1024:
        raise ValueError('A bounded selected matrix partition is required')
    selected = set()
    for row in matrices:
        point = tuple(row['coordinate'])
        if point in selected or row['file'] != f'weights_{point[0]}_{point[1]}.csl':
            raise ValueError('Repeated or misnamed matrix initializer')
        selected.add(point)
        original = row['original_record']
        keys = ['layer', 'tensor', 'row', 'column', 'valid_rows', 'valid_columns']
        if point in owners:
            tile = owners[point]
            expected = (tile.layer, tile.tensor, tile.row, tile.column, 128, tile.valid_columns)
            auxiliary = auxiliaries[point]['file'] if tile.bias_tensor else None
        else:
            x, y = point
            index = (y-160)*192+x
            if not (0 <= x < 192 and 160 <= y < 210 and 0 <= index < 9496):
                raise ValueError(f'Unknown original matrix owner {point}')
            expected = (-1, 'model.embed_tokens.weight', (index//8)*128, (index % 8)*112, 128, 112)
            auxiliary = None
        if (tuple(original.get(key) for key in keys) != expected or
                (original.get('x'), original.get('y')) != point):
            raise ValueError(f'Original matrix slice differs at {point}')
        result[point] = StaticData(row['file'], auxiliary)
    files = {row['file'] for row in matrices} | {row['file'] for row in auxiliaries.values()}
    if not files <= manifest['files'].keys():
        raise ValueError('Unbound data declaration file')
    return result, sorted(files), dict(
        input_bundle=root.name, input_manifest_sha256=digest(root / 'manifest.json'),
        selected_matrix_tiles=len(selected), initialized_controllers=25,
        initialized_bias_roots=sum(point in auxiliaries and auxiliaries[point]['kind'] == 'bias'
                                   for point in selected),
        scope='Typed binding of frozen input declarations only; no original byte re-audit or SDK acceptance.')
