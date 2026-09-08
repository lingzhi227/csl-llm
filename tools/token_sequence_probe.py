"""Bounded SDK metadata lifecycle probe, not neural inference or KV capacity."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def expected(prompt, script):
    generated = []
    for token in script:
        generated.append(token)
        if token in (151643, 151645):
            break
    consumed_tokens = list(prompt)+generated[:-1]
    reason = 3 if not script else 2 if generated[-1] in (151643, 151645) else 1
    checksum = 17
    for position, token in enumerate(consumed_tokens):
        head = int(bool(script) and position >= len(prompt)-1)
        for value in (token, position, head):
            checksum = ((checksum*257) % 65521+value) % 65521
    return generated, [len(consumed_tokens), len(generated), reason, 1, 0, checksum,
                       len(generated), len(prompt)-int(bool(script)), consumed_tokens[0],
                       consumed_tokens[-1], len(prompt)-1 if script else 0xffffffff]


def prepare(source_root):
    import numpy as np
    cases = [('one-prefill', [7], []),
             ('prompt-eos-is-data', [151645, 151643, 19], [7, 8, 9]),
             ('first-generated-eos', [21], [151643, 8, 9]),
             ('second-generated-eos', [22, 23], [17, 151645, 19]),
             ('eos-at-limit', [9, 10], [3, 151645]),
             ('capacity-maxgen', list(range(5, 1797)), list(range(500, 756))),
             ('capacity-prefill', list(range(2048)), []),
             ('reset-small', [33, 34], [44]),
             ('repeat-small', [33, 34], [44])]
    model_config = ROOT / 'models/qwen2.5-0.5b-7ae5576/generation_config.json'
    assert json.loads(model_config.read_text())['eos_token_id'] == [151645, 151643]
    out = ROOT / 'evidence' / ('token-sequence-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    arrays, descriptions = {}, []
    for i, (name, prompt, script) in enumerate(cases):
        assert len(prompt)+len(script) <= 2048 and len(script) <= 256
        generated, summary = expected(prompt, script)
        arrays[f'prompt_{i}'] = np.array(prompt, np.uint32)
        arrays[f'script_{i}'] = np.array(script, np.uint32)
        arrays[f'generated_{i}'] = np.array(generated, np.uint32)
        arrays[f'summary_{i}'] = np.array(summary, np.uint32)
        descriptions.append(dict(name=name, prompt_length=len(prompt), generation_limit=len(script)))
    np.savez(out / 'inputs.npz', **arrays)
    for source, target in [(Path(__file__), 'driver.py'), (source_root / 'tools/sdk_probe.py', 'executor.py'),
                           (source_root / 'csl/programs/token_sequence/sequence.csl', 'sequence.csl')]:
        shutil.copy2(source, out / target)
    (out / 'pe.csl').write_text('''param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const seq=@import_module("sequence.csl");
var script=@zeros([256]u32);var control=@zeros([2]u32);var summary=@zeros([11]u32);
var progress=@zeros([1]u32);
fn mix(value:u32) void {summary[5]=((summary[5]*257)%65521+value)%65521;}
fn compute() void {
 seq.reset(@as(u16,control[0]),@as(u16,control[1]));
 for(@range(u16,11))|i|{summary[i]=0;}
 summary[5]=17;summary[10]=@as(u32,0xffffffff);
 while(!seq.finished){
  const position=seq.consumed;const token=seq.start();
  if(position==0){summary[8]=token;}summary[9]=token;
  mix(token);mix(@as(u32,position));mix(if(seq.head_expected) @as(u32,1) else @as(u32,0));
  if(seq.head_expected){
   if(summary[6]==0){summary[10]=@as(u32,position);}
   summary[6]+=1;seq.complete_head(script[seq.generated_count]);
  }else{summary[7]+=1;seq.complete_prefill();}
 }
 summary[0]=@as(u32,seq.consumed);summary[1]=@as(u32,seq.generated_count);summary[2]=@as(u32,seq.reason);
 summary[3]=if(seq.finished) @as(u32,1) else @as(u32,0);summary[4]=if(seq.busy) @as(u32,1) else @as(u32,0);
 progress[0]+=1;sys.unblock_cmd_stream();
}
const pp:[*]u32=&seq.prompt;const gp:[*]u32=&seq.generated;const sp:[*]u32=&script;
const cp:[*]u32=&control;const rp:[*]u32=&summary;const ep:[*]u32=&progress;
comptime{@export_symbol(pp,"prompt");@export_symbol(gp,"generated");@export_symbol(sp,"script");@export_symbol(cp,"control");@export_symbol(rp,"summary");@export_symbol(ep,"progress");@export_symbol(compute);}
''')
    (out / 'layout.csl').write_text('const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});layout{@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});'+''.join(f'@export_name("{n}",[*]u32,false);' for n in ['prompt','generated','script','control','summary','progress'])+'@export_name("compute",fn()void);}\n')
    (out / 'config.json').write_text(json.dumps(dict(cases=descriptions, generation_config_sha256=digest(model_config),
                                                  scope='1PE/1thread scripted token metadata only: prompt/generated selection, both EOS, reset,2048total/256generation counters. No actual logits, KV cache or LLM inference claim.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',
                                                      files={p.name:digest(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform
    verify(root);os.chdir(root)
    config=json.loads((root/'config.json').read_text());data=np.load(root/'inputs.npz')
    command=['cslc','layout.csl','--arch=wse3','--fabric-dims=8,3','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--max-parallelism=1','--dump-dsr-alloc-graph']
    (root/'compile-command.json').write_text(json.dumps(command)+'\n');subprocess.run(command,check=True)
    runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=1,dump_core=True),SdkTarget.WSE3))
    ids={name:runner.get_id(name) for name in ['prompt','generated','script','control','summary','progress']}
    runner.load();runner.run();reports=[]
    kwargs=dict(streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
    try:
        for index,case in enumerate(config['cases']):
            for name,value in [('prompt',data[f'prompt_{index}']),('script',data[f'script_{index}']),
                               ('control',np.array([case['prompt_length'],case['generation_limit']],np.uint32))]:
                if len(value):runner.memcpy_h2d(ids[name],value.copy(),0,0,1,1,len(value),**kwargs)
            runner.launch('compute',nonblock=False);actual={}
            for name,count in [('summary',11),('generated',256),('progress',1)]:
                value=np.zeros(count,np.uint32);runner.memcpy_d2h(value,ids[name],0,0,1,1,count,**kwargs);actual[name]=value
            path=root/f'actual-{index}.npz';np.savez(path,**actual)
            np.testing.assert_array_equal(actual['summary'],data[f'summary_{index}'])
            wanted=data[f'generated_{index}'];np.testing.assert_array_equal(actual['generated'][:len(wanted)],wanted)
            assert actual['progress'][0]==index+1
            reports.append(dict(case=case,exact=True,actual_sha256=digest(path)))
            (root/'results.json').write_text(json.dumps(dict(success=False,cases=reports),indent=2)+'\n');print(case['name'],flush=True)
    finally:runner.stop()
    (root/'results.json').write_text(json.dumps(dict(success=True,cases=reports,scope=config['scope']),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',type=Path);parser.add_argument('--execute',type=Path)
    parser.add_argument('--project-root',type=Path);parser.add_argument('--source-root',type=Path);args=parser.parse_args()
    if args.project_root:ROOT=args.project_root.resolve()
    if args.prepare:prepare(args.source_root.resolve() if args.source_root else ROOT)
    elif args.worker:worker(args.worker.resolve())
    else:
        sys.path.insert(0,str(args.execute.resolve()));from executor import execute
        execute(args.execute.resolve(),240)
