"""Full two-layer coverage and common initializers are mandatory before assembly."""
import copy
import json
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'src')]
from decoder_chain_assemble import complete_coverage, complete_expected
from csl_llm.decoder_layout import tiles


def configs():
    records = [tile for layer in (0, 1) for tile in tiles(layer) if tile.operation != 'kv']
    rows = [dict(coordinate=[tile.x, tile.y], original_record=dict(
        offset=index*28672, layer=tile.layer, x=tile.x, y=tile.y)) for index, tile in enumerate(records)]
    return [dict(original_pack='original', mapping_audit=dict(passed=True, matrix_tiles=34552,
        pack_manifest_sha256='pack', pack_verification_sha256='verification'),
        frequency_fixture_manifest_sha256='fixture', frequency_auxiliary_sha256='frequency',
        matrices=rows[start:end]) for start, end in ((2048, 2088), (0, 1024), (1024, 2048))]


def test_complete_coverage_rejects_gaps_overlap_wrong_layer_and_provenance():
    source = configs()
    matrix, universe = complete_coverage(source)
    assert len(matrix) == 2088 and len(universe-matrix) == 492
    for change in (lambda c: c.pop(), lambda c: c.append(c[0]),
                   lambda c: c[0]['matrices'][0]['original_record'].update(layer=2),
                   lambda c: c[0].update(frequency_auxiliary_sha256='different'),
                   lambda c: c[0]['matrices'][0]['original_record'].update(offset=1)):
        bad = copy.deepcopy(source)
        change(bad)
        with pytest.raises(ValueError):
            complete_coverage(bad)


def test_merge_uses_original_owners_and_rejects_conflicting_common(tmp_path, monkeypatch):
    import decoder_chain_audit_probe
    source = configs()
    matrix, universe = complete_coverage(source)
    roots = []
    for index, config in enumerate(source):
        root = tmp_path / str(index)
        root.mkdir()
        (root / 'static-input-config.json').write_text(json.dumps(config))
        roots.append(root)
    conflict = False
    def expected(root):
        selected = {tuple(row['coordinate']) for row in source[int(root.name)]['matrices']}
        for point in universe:
            size = 28672 if point in matrix else 4
            fill = int(root.name)+1 if point in selected else 0
            if conflict and point == (128, 1) and root.name == '1':
                fill = 7
            yield point, dict(weights=bytes([fill])*size)
    monkeypatch.setattr(decoder_chain_audit_probe, 'expected_symbols', expected)
    merged = complete_expected(roots)
    for index, config in enumerate(source):
        for row in config['matrices']:
            assert merged[tuple(row['coordinate'])]['weights'] == bytes([index+1])*28672
    conflict = True
    with pytest.raises(ValueError, match='Common original initializers differ'):
        complete_expected(roots)
