"""Adversarial diagnostic checks; synthetic memory is not SDK evidence."""
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'src')]
from csl_llm.decoder_layout import tiles
from decoder_chain_validate import validate


class DiagnosticMemory:
    def __init__(self, mutation=None):
        self.owners = {(t.x, t.y): t for layer in (0, 1) for t in tiles(layer)}
        self.mutation = mutation

    def read(self, px, py, name, size):
        point = px-4, py-1
        tile = self.owners.get(point)
        raw = bytes(size)
        if name == 'progress':
            raw = np.array([2], '<u4').tobytes()
        elif name == 'identity' and tile:
            raw = np.array([tile.packet_group, *tile.destination], '<u4').tobytes()
        elif name == 'status':
            if point == (128, 0):
                raw = np.array([1, 1], '<u4').tobytes()
            elif tile and tile.operation == 'kv':
                raw = np.array([1, int(tile.stripe == 0)], '<u4').tobytes()
            elif point in ((62, 0), (126, 0)):
                raw = np.array([1, 0], '<u4').tobytes()
        elif name == 'timing':
            raw = np.array([1, 0, 0, 2, 0, 0], '<u2').tobytes()
        elif name.startswith('boundary.'):
            value = {'invocation': 1, 'sent_count': 896, 'received_count': 896,
                     'pending_count': 0, 'command': 1, 'started': 1, 'sending': 0}[name.split('.')[1]]
            raw = np.array([value], '<u4' if size == 4 else '<u2').tobytes()
        elif name == 'kv.seen':
            raw = np.array([1], '<u2').tobytes()
        elif name == 'kv.valid':
            raw = np.array([int(tile.stripe == 0)], '<u2').tobytes()
        if self.mutation:
            raw = self.mutation(point, name, raw)
        assert len(raw) == size
        return raw


def fixture():
    reference = dict(input=np.zeros((1, 896), np.float32))
    sizes = dict(input_norm=896, q=896, k=128, v=128, context=896, o=896,
                 attention_residual=896, post_norm=896, up=4864, gate=4864,
                 activation=4864, down=896, output=896)
    for layer in (0, 1):
        for name, size in sizes.items():
            reference[f'layer_{layer}_{name}'] = np.zeros((1, size), np.float32)
        for name in ('cache_k', 'cache_v'):
            reference[f'layer_{layer}_{name}'] = np.zeros((2, 1, 64), np.float32)
    initial = {(x, y): {} for y in range(20) for x in range(129)}
    normal = dict(ready=np.ones(2580, np.uint32), progress=np.full(2580, 2, np.uint32),
                  output=np.zeros(896, np.float32))
    policy = dict(max_abs=1e-5, both=dict(relative_l2=2e-4, relative_peak=5e-4))
    return initial, reference, normal, policy


def test_primary_downstream_reference_is_not_replaced_by_actual_upstream(tmp_path):
    initial, reference, normal, policy = fixture()
    assert validate(DiagnosticMemory(), initial, reference, normal, policy, tmp_path)['passed']
    reference['layer_1_output'][0, 0] = 1
    with pytest.raises(ValueError, match='layer_1_output'):
        validate(DiagnosticMemory(), initial, reference, normal, policy, tmp_path)
    assert (tmp_path / 'actual.npz').is_file()


@pytest.mark.parametrize('failure', ['inflight', 'stale_cache', 'not_ready'])
def test_endpoint_output_cannot_hide_inflight_or_stale_state(tmp_path, failure):
    initial, reference, normal, policy = fixture()
    def mutation(point, name, raw):
        if failure == 'inflight' and point == (62, 0) and name == 'packets.sending':
            return np.array([1], '<u2').tobytes()
        if failure == 'stale_cache' and point == (33, 14) and name == 'kv.values':
            return np.array([1], '<f4').tobytes()+raw[4:]
        return raw
    if failure == 'not_ready':
        normal['ready'][100] = 0
    with pytest.raises(AssertionError):
        validate(DiagnosticMemory(mutation), initial, reference, normal, policy, tmp_path)
    assert (tmp_path / 'actual.npz').is_file() and (tmp_path / 'protocol.json').is_file()
