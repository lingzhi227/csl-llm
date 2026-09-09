"""Pinned official Transformers CPU reference; never used as device compute."""
import argparse, datetime, hashlib, importlib.metadata, inspect, json, platform, subprocess, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import modeling_qwen2


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('prompts',type=Path);p.add_argument('output',type=Path);p.add_argument('--dtype',choices=['bfloat16','float32'],required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'driver.py').write_bytes(Path(__file__).read_bytes());(a.output/'prompts.json').write_bytes(a.prompts.read_bytes())
    started=time.monotonic();manifest=json.loads((a.model/'manifest.json').read_text())
    for item in manifest['files']:assert sha(a.model/item['path'])==item['sha256'],item['path']
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,trust_remote_code=False,torch_dtype=getattr(torch,a.dtype),attn_implementation='eager').eval()
    assert model.model.embed_tokens.weight.data_ptr()==model.lm_head.weight.data_ptr()
    assert len(model.model.layers)==24 and model.config.vocab_size==151936
    source=Path(inspect.getfile(modeling_qwen2));(a.output/'modeling_qwen2.py').write_bytes(source.read_bytes())
    lock=subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True);(a.output/'requirements.lock').write_text(lock)
    (a.output/'torch-config.txt').write_text(torch.__config__.show())
    corpus=json.loads(a.prompts.read_text());stats={};captured={};active=False
    def save_tensor(name,value):
        tensor=value.detach().float().cpu();assert torch.isfinite(tensor).all(),name
        captured[name]=tensor.numpy().copy()
        low,high=float(tensor.min()),float(tensor.max());r=stats.setdefault(name,dict(min=low,max=high,max_abs=0.,observations=0))
        r.update(min=min(r['min'],low),max=max(r['max'],high),max_abs=max(r['max_abs'],abs(low),abs(high)),observations=r['observations']+1)
    def hook(name):
        def record(module,inputs,output):
            if active:save_tensor(name,output[0] if isinstance(output,tuple) else output)
        return record
    selected={'model.embed_tokens','model.norm'}
    for i in range(24):
        base=f'model.layers.{i}'
        selected.add(base)
        selected.update(base+'.'+n for n in ('input_layernorm','post_attention_layernorm','self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj','mlp'))
    handles=[module.register_forward_hook(hook(name)) for name,module in model.named_modules() if name in selected]
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    reports=[]
    with torch.inference_mode():
        for prompt in corpus['prompts']:
            path=a.output/prompt['id'];path.mkdir();ids=tokenizer.apply_chat_template(prompt['messages'],add_generation_prompt=True,return_tensors='pt')
            prompt_tokens=ids[0].tolist();sequence=ids.clone();cache=None;generated=[];steps=[];prior_kv=None
            (path/'prompt.txt').write_text(tokenizer.apply_chat_template(prompt['messages'],add_generation_prompt=True,tokenize=False))
            for step in range(corpus['max_new_tokens']):
                captured={};active=True
                current=sequence if cache is None else sequence[:,-1:]
                before=time.monotonic();output=model(input_ids=current,attention_mask=torch.ones_like(sequence),past_key_values=cache,use_cache=True,output_attentions=True,return_dict=True);active=False
                cache=output.past_key_values;assert cache.get_seq_length()==sequence.shape[1]
                kv=[]
                for layer,(key,value) in enumerate(zip(cache.key_cache,cache.value_cache)):
                    assert list(key.shape)==[1,2,sequence.shape[1],64]
                    if prior_kv is not None:
                        assert torch.equal(key[:,:,:-1],prior_kv[layer][0]) and torch.equal(value[:,:,:-1],prior_kv[layer][1])
                    kv.append((key.clone(),value.clone()));save_tensor(f'cache.{layer}.key',key);save_tensor(f'cache.{layer}.value',value)
                    save_tensor(f'attention.{layer}.probability',output.attentions[layer])
                prior_kv=kv
                logits=output.logits[0,-1].float();assert logits.numel()==151936 and torch.isfinite(logits).all()
                captured['logits']=logits.numpy().copy();top=torch.topk(logits,2);token=int(logits.argmax())
                np.savez_compressed(path/f'step-{step:03d}.npz',**captured)
                steps.append(dict(step=step,context_tokens=int(sequence.shape[1]),selected_token=token,top1=float(top.values[0]),top2=float(top.values[1]),margin=float(top.values[0]-top.values[1]),elapsed_seconds=time.monotonic()-before,trace=f'step-{step:03d}.npz'))
                generated.append(token);sequence=torch.cat([sequence,torch.tensor([[token]])],dim=1)
                if token in eos:break
            generated_official=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=corpus['max_new_tokens'],do_sample=False,repetition_penalty=1.0,temperature=1.0,top_p=1.0,top_k=0,pad_token_id=tokenizer.pad_token_id,use_cache=True)[0,ids.shape[1]:].tolist()
            (path/'generation-comparison.json').write_text(json.dumps(dict(manual=generated,official_generate=generated_official,repetition_penalty=1.0,do_sample=False),indent=2)+'\n')
            assert generated==generated_official,'Manual cached decode disagrees with official generate'
            record=dict(id=prompt['id'],prompt_tokens=prompt_tokens,generated_tokens=generated,text=tokenizer.decode(generated,skip_special_tokens=False),stop_reason='eos' if generated[-1] in eos else 'length',steps=steps,official_generate_exact=True)
            (path/'result.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n');reports.append(record)
            print('REFERENCE PROMPT PASS',a.dtype,prompt['id'],repr(record['text']),flush=True)
    for handle in handles:handle.remove()
    report=dict(success=True,scope='Official Transformers CPU reference and activation measurements only; no CSL/SDK acceptance.',dtype=a.dtype,generation_policy=dict(do_sample=False,repetition_penalty=1.0,raw_logits_argmax=True),model=manifest['model'],revision=manifest['revision'],model_manifest_sha256=sha(a.model/'manifest.json'),prompts_sha256=sha(a.prompts),driver_sha256=sha(Path(__file__)),versions={n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','safetensors','tokenizers')},python=platform.python_version(),threads=8,eager_attention=True,deterministic=True,tied_embedding=True,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,prompts=reports,activation_ranges=stats,files={str(path.relative_to(a.output)):sha(path) for path in a.output.rglob('*') if path.is_file()})
    (a.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print('REFERENCE COMPLETE',a.output,flush=True)


if __name__=='__main__':main()
