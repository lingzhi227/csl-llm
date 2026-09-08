"""Auditable resident Qwen placement candidate; transport/code fit unqualified."""
from dataclasses import asdict,dataclass
import math


@dataclass(frozen=True)
class Tile:
    x:int
    y:int
    role:str
    layer:int
    tensor:str=''
    row:int=0
    column:int=0
    valid_rows:int=0
    valid_columns:int=0
    head:int=-1
    stripe:int=-1


def plan(inventory):
    shapes={v['name']:v['shape'] for v in inventory['tensors']}
    tiles=[];occupied=set();covered={};auxiliary=[]
    def add(tile):
        assert 0<=tile.x<192 and 0<=tile.y<210
        assert (tile.x,tile.y) not in occupied,(tile.x,tile.y,tile.role)
        occupied.add((tile.x,tile.y));tiles.append(tile)
    def matrix(name,layer,position):
        rows,columns=shapes[name];nr,nc=math.ceil(rows/128),math.ceil(columns/112)
        for i in range(nr):
            for j in range(nc):
                x,y=position(i,j,nc)
                add(Tile(x,y,'matrix',layer,name,i*128,j*112,min(128,rows-i*128),min(112,columns-j*112)))
        covered[name]=nr*nc
    for layer in range(24):
        bx,by=(layer%3)*64,(layer//3)*20;p=f'model.layers.{layer}.'
        for op,offset in [('mlp.up_proj',0),('mlp.gate_proj',16)]:
            matrix(p+op+'.weight',layer,lambda i,j,nc,offset=offset:(bx+offset+(i%2)*8+j,by+i//2))
        matrix(p+'mlp.down_proj.weight',layer,lambda i,j,nc:(bx+32+j%22,by+i*2+j//22))
        for op,dy in [('q',0),('k',7),('v',8),('o',9)]:
            matrix(p+f'self_attn.{op}_proj.weight',layer,lambda i,j,nc,dy=dy:(bx+54+j,by+dy+i))
            if op!='o':
                name=p+f'self_attn.{op}_proj.bias';length=shapes[name][0]
                auxiliary.append(dict(tensor=name,owners=[dict(x=bx+54,y=by+dy+i,start=i*128,end=min((i+1)*128,length)) for i in range(math.ceil(length/128))],bytes=length*2))
        for head in range(2):
            for ordinal in range(66):
                add(Tile(bx+32+head*11+ordinal%11,by+14+ordinal//11,'kv' if ordinal<64 else 'kv_padding',layer,head=head,stripe=ordinal))
        add(Tile(bx+62,by,'layer_control',layer))
        for suffix in ('input_layernorm.weight','post_attention_layernorm.weight'):auxiliary.append(dict(tensor=p+suffix,owner=dict(x=bx+62,y=by),bytes=math.prod(shapes[p+suffix])*2))
    matrix('model.embed_tokens.weight',-1,lambda i,j,nc:((i%24)*8+j,160+i//24))
    add(Tile(191,209,'model_control',-1));auxiliary.append(dict(tensor='model.norm.weight',owner=dict(x=191,y=209),bytes=1792))
    assert set(covered)|{a['tensor'] for a in auxiliary}==set(shapes)
    assert len(covered)==169 and len(auxiliary)==121
    assert sum(t.valid_rows*t.valid_columns for t in tiles if t.role=='matrix')+sum(math.prod(shapes[a['tensor']]) for a in auxiliary)==494032768
    counts={role:sum(t.role==role for t in tiles) for role in sorted({t.role for t in tiles})}
    assert counts['matrix']==34552 and counts['kv']==3072
    memory={
        'matrix_decode':dict(weights_bf16=128*112*2,input_f32=112*4,expanded_f32=128*4,output_f32=128*4,reduction_partial_f32=128*4,communication_buffers=2*128*4,code_allowance=12288,task_table_allowance=1024,stack_allowance=2048,metadata_allowance=256,bias_allowance=128*2),
        'kv_decode':dict(cache_f32=32*64*2*4,query_f32=7*64*4,scores_f32=7*32*4,context_f32=7*64*4,statistics_f32=2*7*4,communication_buffers=2*7*64*4,code_allowance=12288,task_table_allowance=1024,stack_allowance=2048,metadata_allowance=256),
    }
    for values in memory.values():assert sum(values.values())<=49152
    return dict(kind='unqualified_resident_layout_candidate',full_model_sdk_pass=False,fabric_application=dict(width=192,height=210,positions=40320),tile_shape=dict(outputs=128,inner=112),tiles=[asdict(t) for t in tiles],role_counts=counts,unassigned_transit_positions=40320-len(tiles),matrix_tiles_by_tensor=covered,auxiliary_tensors=auxiliary,estimated_memory_per_pe=memory,kv_bytes=counts['kv']*32*64*2*4,weights_payload_bytes=988065536,matrix_allocation_bytes=counts['matrix']*128*112*2,tied_embedding_lm_head_single_storage=True,context=2048,unproven=['communication routes/resources and global completion','actual composed code/stack/ELF fit per role','full fabric compiler/simulator cost','prefill batching scratch and phase lifetimes','complete-model numerical and generation acceptance'])
