"""Two original-sized streamed decoder regions and one diagnostic endpoint."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from csl_llm.static_data import bindings
from csl_llm.decoder_layout import model_origin
import resident_decoder_probe as decoder

WIDTH, HEIGHT = 129, 20
ENDPOINT = (128, 0)


def generate(static_data=None):
    data = bindings(static_data)
    remaining = set(data)
    lines = ['const memcpy=@import_module("<memcpy/get_params>",.{.width=129,.height=20});',
             'layout{@set_rectangle(129,20);']
    for layer in range(2):
        origin = model_origin(layer)
        destination = (126, 0) if layer == 0 else ENDPOINT
        local = {p: value for p, value in data.items()
                 if origin[0] <= p[0] < origin[0]+64 and 0 <= p[1] < 20}
        lines.append(decoder.make_layout(decoder.geometry(origin), origin=origin, layer=layer,
            fragment=True, next_destination=destination, initialization='sdk',
            code_file='decoder.csl', coordinate_identity=True, weight_storage='u8', static_data=local))
        remaining.difference_update(local)
    if remaining:
        raise ValueError('Static data lies outside the two native decoder regions')
    lines.append('@set_tile_code(128,0,"chain_endpoint.csl",.{.memcpy_params=memcpy.get_params(128),.first_x=62,.first_y=0});')
    for y in range(1, 20):
        lines.append(f'@set_tile_code(128,{y},"decoder.csl",.{{.memcpy_params=memcpy.get_params(128),.role=0,.group=0,.ordinal=0,.participants=2,.negative=WEST,.positive=EAST,.destination_x=0,.destination_y=0,.static_weights=false,.runtime_identity=true,.weight_type=u8}});')
    standalone = decoder.make_layout(decoder.geometry(), initialization='sdk', weight_storage='u8')
    lines.extend(line for line in standalone.splitlines() if line.startswith('@export_name'))
    return '\n'.join(lines+['}'])+'\n'
