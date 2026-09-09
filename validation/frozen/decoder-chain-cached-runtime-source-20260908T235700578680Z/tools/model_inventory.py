"""Validate the complete safetensors inventory without importing model code."""
import argparse, hashlib, json, math, struct
from pathlib import Path


def expected():
    tensors={'model.embed_tokens.weight':[151936,896], 'model.norm.weight':[896]}
    for layer in range(24):
        prefix=f'model.layers.{layer}.'
        for name, shape in {
            'self_attn.q_proj.weight':[896,896], 'self_attn.q_proj.bias':[896],
            'self_attn.k_proj.weight':[128,896], 'self_attn.k_proj.bias':[128],
            'self_attn.v_proj.weight':[128,896], 'self_attn.v_proj.bias':[128],
            'self_attn.o_proj.weight':[896,896],
            'mlp.gate_proj.weight':[4864,896], 'mlp.up_proj.weight':[4864,896],
            'mlp.down_proj.weight':[896,4864],
            'input_layernorm.weight':[896], 'post_attention_layernorm.weight':[896],
        }.items(): tensors[prefix+name]=shape
    return tensors


def inspect(path):
    with path.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        header=json.loads(f.read(length))
        assert length < 1024*1024
    items={k:v for k,v in header.items() if k!='__metadata__'}
    assert set(items)==set(expected())
    offsets=[]; records=[]
    for name, shape in expected().items():
        value=items[name];assert value['dtype']=='BF16' and value['shape']==shape,name
        start,end=value['data_offsets'];assert end-start==2*math.prod(shape),name
        offsets.append((start,end));records.append(dict(name=name,shape=shape,dtype='BF16',parameters=math.prod(shape),bytes=end-start,offset=start))
    cursor=0
    for start,end in sorted(offsets):assert start==cursor;cursor=end
    assert 8+length+cursor==path.stat().st_size
    total=sum(x['parameters'] for x in records);assert total==494032768
    return dict(passed=True,tensors=records,tensor_count=len(records),parameters=total,payload_bytes=cursor,header_bytes=length,lm_head_alias='model.embed_tokens.weight')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('report',type=Path);a=p.parse_args();assert not a.report.exists()
    r=inspect(a.model)
    with a.model.open('rb') as f:r['model_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    r['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.report.write_text(json.dumps(r,indent=2)+'\n');print('INVENTORY PASS',r['tensor_count'],r['parameters'])
