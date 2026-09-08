"""Verify every packed word directly against the safetensors byte layout.

No torch or safetensors implementation is used by this checker.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root, model):
    import numpy as np
    manifest = json.loads((root/'manifest.json').read_text())
    assert digest(model) == manifest['model_sha256']
    for name, expected in manifest['files'].items():
        assert digest(root/name) == expected, name
    with model.open('rb') as stream:
        header_length = struct.unpack('<Q',stream.read(8))[0]
        header = json.loads(stream.read(header_length))
    header.pop('__metadata__',None)
    assert len(header) == 290
    source = {}
    for name, record in header.items():
        assert record['dtype'] == 'BF16'
        begin, end = record['data_offsets']
        assert end-begin == math.prod(record['shape'])*2
        source[name] = np.memmap(model,dtype='<u2',mode='r',offset=8+header_length+begin,shape=tuple(record['shape']))
    matrices = {name for name, a in source.items() if a.ndim == 2}
    auxiliary = set(source)-matrices
    records = json.loads((root/'tile-index.json').read_text())
    plan = json.loads((root/'layout.json').read_text())
    planned = [t for t in plan['tiles'] if t['role']=='matrix']
    assert len(records) == len(planned) == 34552
    packed = np.memmap(root/'matrices.bf16',dtype='<u2',mode='r')
    seen = set()
    valid_words = 0
    for ordinal, (record, tile) in enumerate(zip(records,planned)):
        assert record == dict(**tile,offset=ordinal*28672,bytes=28672)
        name = record['tensor']; row=record['row']; col=record['column']
        rows=record['valid_rows'];cols=record['valid_columns']
        assert name in matrices and row%128==col%112==0
        assert rows==min(128,source[name].shape[0]-row) and cols==min(112,source[name].shape[1]-col)
        key=(name,row,col);assert key not in seen;seen.add(key)
        got=packed[ordinal*14336:(ordinal+1)*14336].reshape(112,128)
        assert np.array_equal(got[:cols,:rows].T,source[name][row:row+rows,col:col+cols]),key
        assert np.all(got[cols:]==0) and np.all(got[:cols,rows:]==0),('padding',key)
        valid_words += rows*cols
    expected={(name,row,col) for name in matrices for row in range(0,source[name].shape[0],128) for col in range(0,source[name].shape[1],112)}
    assert seen == expected and packed.size==len(records)*14336
    with np.load(root/'auxiliary.npz') as data:
        assert set(data.files) == auxiliary
        for name in auxiliary:
            assert np.array_equal(data[name],source[name]),name
            valid_words += source[name].size
    assert valid_words == 494032768
    report=dict(passed=True,checked_parameter_words=valid_words,matrix_tiles=len(records),
                manifest_sha256=digest(root/'manifest.json'),checker_sha256=digest(Path(__file__)),
                scope='Every BF16 matrix/auxiliary word matches the original checkpoint; exact coverage and zero padding. No device execution claim.')
    output=root/'verification.json';assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('WEIGHT PACK VERIFIED',valid_words,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('pack',type=Path);parser.add_argument('model',type=Path)
    args=parser.parse_args();verify(args.pack.resolve(),args.model.resolve())
