"""Independent explicit Qwen equations, cached/full-prefix and precision study."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


def metric(a,b):
    a=a.detach().double();b=b.detach().double();d=a-b
    return dict(relative_l2=float(torch.linalg.vector_norm(d)/max(float(torch.linalg.vector_norm(b)),1e-30)),relative_peak=float(d.abs().max()/max(float(b.abs().max()),1e-30)),max_abs=float(d.abs().max()))


def forward(weights,ids,cache=None):
    x=F.embedding(ids,weights['model.embed_tokens.weight']);start=0 if cache is None else cache[0][0].shape[1];positions=torch.arange(start,start+len(ids))
    frequency=1.0/(1e6**(torch.arange(0,64,2).float()/64));phase=positions.float()[:,None]*frequency[None,:];phase=torch.cat([phase,phase],dim=-1);cos,sin=phase.cos(),phase.sin()
    def rotate(t):return t*cos+torch.cat([-t[...,32:],t[...,:32]],dim=-1)*sin
    def norm(v,w):return v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-6)*w
    nodes={'model.embed_tokens':x};next_cache=[]
    for layer in range(24):
        p=f'model.layers.{layer}';w=lambda name:weights[p+'.'+name]
        n=norm(x,w('input_layernorm.weight'));nodes[p+'.input_layernorm']=n
        values=[]
        for name,heads in [('q',14),('k',2),('v',2)]:
            value=F.linear(n,w(f'self_attn.{name}_proj.weight'),w(f'self_attn.{name}_proj.bias'));nodes[p+f'.self_attn.{name}_proj']=value;values.append(value.reshape(-1,heads,64).transpose(0,1))
        q,k,v=values;q,k=rotate(q),rotate(k)
        if cache is not None:k=torch.cat([cache[layer][0],k],dim=1);v=torch.cat([cache[layer][1],v],dim=1)
        next_cache.append((k,v));score=q@k.repeat_interleave(7,dim=0).transpose(-2,-1)*0.125
        score=score.masked_fill(torch.arange(k.shape[1])[None,:]>positions[:,None],float('-inf'));probability=score.softmax(-1);attention=probability@v.repeat_interleave(7,dim=0)
        delta=F.linear(attention.transpose(0,1).reshape(-1,896),w('self_attn.o_proj.weight'));nodes[p+'.self_attn.o_proj']=delta;x=x+delta
        n=norm(x,w('post_attention_layernorm.weight'));nodes[p+'.post_attention_layernorm']=n
        up=F.linear(n,w('mlp.up_proj.weight'));gate=F.linear(n,w('mlp.gate_proj.weight'));delta=F.linear(F.silu(gate)*up,w('mlp.down_proj.weight'))
        nodes.update({p+'.mlp.up_proj':up,p+'.mlp.gate_proj':gate,p+'.mlp.down_proj':delta,p+'.mlp':delta});x=x+delta;nodes[p]=x
    x=norm(x,weights['model.norm.weight']);nodes['model.norm']=x
    return F.linear(x[-1],weights['model.embed_tokens.weight']),next_cache,nodes


def main():
    p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('reference',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);(a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    source=json.loads((a.reference/'report.json').read_text());assert source['success'] and source['dtype']=='bfloat16'
    weights={k:v.float() for k,v in load_file(a.model/'model.safetensors').items()}
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True).eval();seen={}
    def capture(name):
        def hook(m,i,o):seen[name]=(o[0] if isinstance(o,tuple) else o).detach()[0]
        return hook
    names={'model.embed_tokens','model.norm'}
    for i in range(24):
        pfx=f'model.layers.{i}';names.add(pfx);names.update(pfx+'.'+n for n in ('input_layernorm','post_attention_layernorm','self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.up_proj','mlp.gate_proj','mlp.down_proj','mlp'))
    handles=[m.register_forward_hook(capture(n)) for n,m in model.named_modules() if n in names];rows=[];started=time.monotonic()
    with torch.inference_mode():
        for prompt in source['prompts']:
            sequence=list(prompt['prompt_tokens']);cache=None
            for step in range(len(prompt['generated_tokens'])):
                ids=torch.tensor(sequence);logits,whole,stages=forward(weights,ids);increment,cache,_=forward(weights,ids if cache is None else ids[-1:],cache)
                seen={};official=model(ids[None,:],use_cache=True,return_dict=True);checks={'logits':metric(logits,official.logits[0,-1]),'cached_logits':metric(increment,logits)}
                for name,value in stages.items():checks[name]=metric(value,seen[name])
                for layer,((key,value),(ck,cv)) in enumerate(zip(whole,cache)):
                    checks[f'cache.{layer}.key']=metric(key,official.past_key_values.key_cache[layer][0]);checks[f'cache.{layer}.value']=metric(value,official.past_key_values.value_cache[layer][0]);checks[f'increment.{layer}.key']=metric(ck,key);checks[f'increment.{layer}.value']=metric(cv,value)
                for name,m in checks.items():assert m['max_abs']<=1e-5 or (m['relative_l2']<=2e-5 and m['relative_peak']<=3e-5),(prompt['id'],step,name,m)
                baseline=np.load(a.reference/prompt['id']/f'step-{step:03d}.npz');precision={'logits':metric(torch.from_numpy(baseline['logits']),logits)}
                for name,value in stages.items():precision[name]=metric(torch.from_numpy(baseline[name][0,-1]),value[-1])
                top=torch.topk(logits,2);rows.append(dict(prompt=prompt['id'],step=step,context=len(sequence),independent_checks=checks,bf16_vs_f32=precision,f32_top1=int(logits.argmax()),bf16_top1=int(np.argmax(baseline['logits'])),f32_margin=float(top.values[0]-top.values[1])))
                sequence.append(prompt['generated_tokens'][step])
            print('SEMANTICS PASS',prompt['id'],flush=True)
    conversion={}
    for name,w in weights.items():
        converted=w.half().float();conversion[name]=dict(changed=int((converted!=w).sum()),elements=w.numel(),max_abs_error=float((converted-w).abs().max()),finite=bool(torch.isfinite(converted).all()),max_abs=float(w.abs().max()))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();report=dict(success=True,scope='CPU explicit-equation vs official FP32, cached vs whole prefix; BF16 precision characterization, not SDK acceptance.',reference_report_sha256=sha(a.reference/'report.json'),model_manifest_sha256=sha(a.model/'manifest.json'),driver_sha256=sha(Path(__file__)),gate=dict(max_abs=1e-5,or_both=dict(relative_l2=2e-5,relative_peak=3e-5)),steps=rows,weight_f16_conversion=conversion,elapsed_seconds=time.monotonic()-started)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('REFERENCE SEMANTICS COMPLETE',flush=True)
if __name__=='__main__':main()
