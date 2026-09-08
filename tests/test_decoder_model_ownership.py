"""Cross-check layer-indexed builder against the complete checkpoint plan."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from csl_llm.decoder_layout import controller_parameters, geometry, model_origin, tiles
from csl_llm.layout import plan


class CompleteDecoderOwnership(unittest.TestCase):
    def test_all_24_layers_against_inventory_plan(self):
        full = plan(json.loads((ROOT / 'evidence/model-inventory.json').read_text()))
        matrices = {(t['tensor'], t['row'], t['column']): t for t in full['tiles'] if t['role'] == 'matrix'}
        caches = {(t['layer'], t['head'], t['stripe']): t for t in full['tiles'] if t['role'] in ('kv', 'kv_padding')}
        auxiliary = {t['tensor']: t for t in full['auxiliary_tensors']}
        occupied = set()
        for layer in range(24):
            records = tiles(layer)
            self.assertEqual(sum(t.operation != 'kv' for t in records), 1044)
            self.assertEqual(sum(t.operation == 'kv' for t in records), 132)
            for tile in records:
                self.assertNotIn((tile.x, tile.y), occupied)
                occupied.add((tile.x, tile.y))
                if tile.operation == 'kv':
                    original = caches[layer, tile.group, tile.stripe]
                else:
                    original = matrices[tile.tensor, tile.row, tile.column]
                    self.assertEqual(tile.valid_columns, original['valid_columns'])
                    self.assertEqual(original['valid_rows'], 128)
                    if tile.bias_tensor:
                        self.assertIn(dict(x=tile.x, y=tile.y, start=tile.row, end=tile.row+128),
                                      auxiliary[tile.bias_tensor]['owners'])
                self.assertEqual((tile.x, tile.y), (original['x'], original['y']))
            control = controller_parameters(layer)
            self.assertNotIn(control['coordinate'], occupied)
            for name in control['norms']:
                self.assertEqual(control['coordinate'], (auxiliary[name]['owner']['x'], auxiliary[name]['owner']['y']))
        self.assertEqual(len(occupied), 24*1176)

    def test_alternate_origin_preserves_relative_protocol(self):
        origin = (7, 11)
        base, shifted = tiles(12, (0, 0)), tiles(12, origin)
        for left, right in zip(base, shifted):
            self.assertEqual((right.x, right.y), (left.x+7, left.y+11))
            self.assertEqual(right.destination, (left.destination[0]+7, left.destination[1]+11))
            self.assertEqual((left.tensor, left.row, left.column, left.stripe, left.ordinal, left.packet_group),
                             (right.tensor, right.row, right.column, right.stripe, right.ordinal, right.packet_group))
        for bad in [-1, 24, True]:
            with self.assertRaises(ValueError):
                model_origin(bad)
        with self.assertRaises(ValueError):
            geometry((-1, 0))


if __name__ == '__main__':
    unittest.main()
