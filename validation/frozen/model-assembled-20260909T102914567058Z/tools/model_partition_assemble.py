"""Assemble a complete original model from audited, intact compiler partitions.

Authoring entry point. Execution must use a fresh, self-contained guarded bundle.
This module never invokes a compiler or SdkRuntime. It deliberately rejects an
incomplete model before writing output; component passes are not model passes.
"""
import json
from pathlib import Path


def complete_coverage(configs, matrix_count=34552):
    """Validate independent pack ownership and invariant original data sources."""
    if not configs:
        raise ValueError('At least one completed partition is required')
    indices = set()
    coordinates = set()
    provenance = None
    for config in configs:
        mapping = config['mapping_audit']
        if mapping['passed'] is not True or mapping['matrix_tiles'] != matrix_count:
            raise ValueError('Complete verified original matrix mapping required')
        current = (config['original_pack'], mapping['pack_manifest_sha256'],
                   mapping['pack_verification_sha256'],
                   config['frequency_fixture_manifest_sha256'],
                   config['frequency_auxiliary_sha256'])
        if provenance is None:
            provenance = current
        elif provenance != current:
            raise ValueError('Partitions use different original data or frequencies')
        rows = config['matrices']
        if not 0 < len(rows) <= 1024:
            raise ValueError('Each audited partition must contain 1..1024 matrices')
        for row in rows:
            offset = row['original_record']['offset']
            point = tuple(row['coordinate'])
            original = row['original_record']
            if (type(offset) is not int or offset < 0 or offset % 28672 or
                    offset // 28672 >= matrix_count or point != (original['x'], original['y'])):
                raise ValueError('Invalid original matrix index or coordinate')
            index = offset // 28672
            if index in indices or point in coordinates:
                raise ValueError('Repeated original matrix or physical owner')
            indices.add(index)
            coordinates.add(point)
    if indices != set(range(matrix_count)):
        raise ValueError('Incomplete original model: every matrix block is required')
    return coordinates


def assemble(audits, output):
    from sdk_probe import sha
    from model_partition_contract import audit
    from elf_partition import compose
    output = Path(output)
    if output.exists():
        raise ValueError('Assembly output must be a new directory')
    roots = [Path(root).resolve() for root in audits]
    if len(set(roots)) != len(roots):
        raise ValueError('Repeated source audit')
    # These configs are untrusted until audited() verifies their frozen hashes.
    # Early coverage rejection avoids an expensive scan for a partial request.
    configs = [json.loads((root / 'compiled/static-input-config.json').read_text())
               for root in roots]
    matrix = complete_coverage(configs)
    universe = {(x, y) for y in range(210) for x in range(193)}
    if not matrix <= universe or len(matrix) != 34552:
        raise ValueError('Invalid complete-model placement')
    common = universe - matrix
    if len(common) != 5978:
        raise ValueError('Invalid common model ownership')
    parts = []
    contracts = []
    for root, config in zip(roots, configs):
        contract = audit(root, roots[0])
        compiled = root / 'compiled'
        selected = {tuple(row['coordinate']) for row in config['matrices']}
        actual = {tuple(point) for image in contract['classes']['selected']
                  for point in image['coordinates']}
        if actual != selected:
            raise ValueError('Audited ELF ownership differs from original matrix selection')
        if set(contract['original_pack_indices']) != {
                row['original_record']['offset'] // 28672 for row in config['matrices']}:
            raise ValueError('Contract and source original indices differ')
        parts.append((compiled / 'out', {(x+4, y+1) for x, y in selected}))
        contracts.append(dict(source_audit=root.name,
            source_execution_sha256=sha(root / 'execution.json'),
            initializer_report_sha256=sha(root / 'initializers.json'), contract=contract))
    # All common code is supplied by exactly one complete compiler partition.
    parts.append((roots[0] / 'compiled/out', {(x+4, y+1) for x, y in common}))
    result = compose(parts, output, {(x+4, y+1) for x, y in universe})
    return dict(passed=True, offline_only=True, original_matrix_tiles=34552,
        application_coordinates=40530, contracts=contracts, composition=result,
        scope='Complete original matrix coverage and intact ELF composition only. '
              'Original initializer acceptance belongs to the bound source audits. '
              'Runtime symbol qualification, actual SDK loading, full-model numerical '
              'inference, persistent KV and generation require separate execution.')


def iter_initializers(compiled_roots):
    """Stream complete original expectations for final core validation.

    Keep the existing per-partition parser bounded to 1,024 matrices. This
    iterator retains only one partition's original payloads and the common
    initializers instead of materializing another gigabyte of model weights.
    The runtime owner must additionally bind these compiled roots to the exact
    accepted assembly and its composition receipts.
    """
    from sdk_probe import verify, sha
    from model_initializer_audit import expected_symbols
    roots = [Path(root).resolve() for root in compiled_roots]
    if len(set(roots)) != len(roots):
        raise ValueError('Repeated compiled initializer source')
    configs = [json.loads((root/'static-input-config.json').read_text()) for root in roots]
    matrix = complete_coverage(configs)
    universe = {(x, y) for y in range(210) for x in range(193)}
    if not matrix <= universe or len(matrix) != 34552:
        raise ValueError('Incomplete original model placement')
    common = universe-matrix
    if len(common) != 5978:
        raise ValueError('Invalid common model coverage')
    common_expected = None
    for root, config in zip(roots, configs):
        verify(root)
        execution = json.loads((root/'execution.json').read_text())
        result = json.loads((root/'results.json').read_text())
        if (not execution['success'] or not result['success'] or not result['compile_only'] or
                execution['manifest_sha256'] != sha(root/'manifest.json') or
                execution['results_sha256'] != sha(root/'results.json')):
            raise ValueError('Original initializer compilation receipt differs')
        selected = {tuple(row['coordinate']) for row in config['matrices']}
        seen, current_common = set(), {}
        for point, fields in expected_symbols(root):
            if point in seen or point not in universe:
                raise ValueError('Invalid bounded initializer coverage')
            seen.add(point)
            if point in selected:
                yield point, fields
            elif point in common:
                current_common[point] = fields
        if seen != universe or set(current_common) != common:
            raise ValueError('Incomplete bounded initializer coverage')
        if common_expected is None:
            common_expected = current_common
        elif common_expected != current_common:
            raise ValueError('Original common initializers differ between partitions')
    yield from sorted(common_expected.items())
