"""Pack the complete checkpoint into resident column-major BF16 matrix tiles.

Packing changes layout only. Auxiliary norms/biases retain their original bits.
This is an input artifact, not a model execution or placement qualification.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pack(plan_path, model):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    plan = json.loads(plan_path.read_text())
    assert plan['tile_shape'] == {'outputs': 128, 'inner': 112}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT/'evidence'/('resident-weights-'+stamp)
    out.mkdir()
    shutil.copy2(__file__, out/'pack_driver.py')
    shutil.copy2(plan_path, out/'layout.json')
    tensors = load_file(model)
    bits = {name: value.view(torch.int16).numpy().view(np.uint16) for name, value in tensors.items()}
    records = []
    with (out/'matrices.bf16').open('wb') as stream:
        for tile in plan['tiles']:
            if tile['role'] != 'matrix':
                continue
            array = np.zeros((112,128), dtype='<u2')
            row, column = tile['row'], tile['column']
            rows, columns = tile['valid_rows'], tile['valid_columns']
            array[:columns,:rows] = bits[tile['tensor']][row:row+rows,column:column+columns].T
            offset = stream.tell()
            payload = array.tobytes()
            stream.write(payload)
            records.append(dict(**tile, offset=offset, bytes=len(payload)))
    auxiliary = {item['tensor']: bits[item['tensor']] for item in plan['auxiliary_tensors']}
    np.savez(out/'auxiliary.npz', **auxiliary)
    (out/'tile-index.json').write_text(json.dumps(records, separators=(',',':'))+'\n')
    assert len(records) == 34552 and len(auxiliary) == 121
    assert (out/'matrices.bf16').stat().st_size == plan['matrix_allocation_bytes']
    report = dict(kind='resident_weight_input_pack', full_model_sdk_pass=False,
                  model_sha256=digest(model), matrix_tiles=len(records),
                  auxiliary_tensors=len(auxiliary), byte_order='little',
                  matrix_storage='BF16 bits [inner112, outputs128], zero padding',
                  files={p.name:digest(p) for p in out.iterdir() if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=ROOT/'evidence/layout-candidate-v2.json')
    parser.add_argument('--model', type=Path, default=ROOT/'models/qwen2.5-0.5b-7ae5576/model.safetensors')
    args = parser.parse_args()
    pack(args.plan.resolve(), args.model.resolve())
