"""Unqualified full resident model layout assembled from shared region emitters.

Emits actual CSL placement/routes, not an execution or memory qualification.
"""
import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'src'))
from csl_llm.decoder_layout import model_origin
from csl_llm.static_data import bindings
import resident_decoder_probe as decoder
import resident_vocabulary_probe as vocabulary

WIDTH, HEIGHT = 193, 210
MODEL_CONTROLLER = (192, 160)


def generate(*, weight_storage='u32', static_data=None):
    if weight_storage not in ('u32', 'u8'):
        raise ValueError('Weight storage must be u32 or u8')
    data = bindings(static_data)
    if data and weight_storage != 'u8':
        raise ValueError('Explicit native static data requires u8 storage')
    remaining = set(data)
    lines = ['const memcpy=@import_module("<memcpy/get_params>",.{.width=193,.height=210});',
             'layout{@set_rectangle(193,210);']
    for layer in range(24):
        origin = model_origin(layer)
        next_origin = model_origin(layer+1) if layer < 23 else None
        destination = (next_origin[0]+62, next_origin[1]) if next_origin else MODEL_CONTROLLER
        local = {p: value for p, value in data.items()
                 if origin[0] <= p[0] < origin[0]+64 and origin[1] <= p[1] < origin[1]+20}
        remaining.difference_update(local)
        lines.append(decoder.make_layout(decoder.geometry(origin), origin=origin, layer=layer,
            fragment=True, next_destination=destination, initialization='sdk', code_file='decoder.csl',
            coordinate_identity=True, weight_storage=weight_storage, static_data=local))
    # Complete the extra router column above the vocabulary region. All these
    # PEs participate in global lifecycle barriers but own no weights or KV.
    for y in range(160):
        storage = ',.weight_type=u8' if weight_storage == 'u8' else ''
        lines.append(f'@set_tile_code(192,{y},"decoder.csl",.{{.memcpy_params=memcpy.get_params(192),.role=0,.group=0,.ordinal=0,.participants=2,.negative=WEST,.positive=EAST,.destination_x=0,.destination_y=0,.static_weights=false,.runtime_identity=true{storage}}});')
    local = {p: value for p, value in data.items() if p[1] >= 160}
    remaining.difference_update(local)
    lines.append(vocabulary.make_layout('sdk', origin=(0, 160), fragment=True,
        device_control=True, code_file='vocabulary.csl', first_layer=(62, 0), coordinate_identity=True, weight_storage=weight_storage, static_data=local))
    if remaining:
        raise ValueError('Static data targets a guard or an unknown full-model coordinate')
    exports = {}
    for source in [decoder.make_layout(decoder.geometry(), initialization='sdk', weight_storage=weight_storage),
                   vocabulary.make_layout('sdk', device_control=True, weight_storage=weight_storage)]:
        for line in source.splitlines():
            if not line.startswith('@export_name'):
                continue
            name = re.match(r'@export_name\("([^"]+)"', line)[1]
            if name in exports and exports[name] != line:
                raise ValueError(f'Incompatible shared export {name}')
            exports[name] = line
    lines.extend(exports.values())
    lines.append('}')
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with args.output.open('x') as stream:
        stream.write(generate())
    print('UNQUALIFIED FULL MODEL LAYOUT', args.output)
