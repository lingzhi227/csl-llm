"""Frozen SDK probe boundary, actual outputs, per-process-tree resource sampling."""
import argparse,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
SIF=Path(os.environ.get('CSL_LLM_SDK_IMAGE', 'sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif')).expanduser().resolve()


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify(root):
    m=json.loads((root/'manifest.json').read_text())
    for name,digest in m['files'].items():assert sha(root/name)==digest,name
    return m


def worker(root):
    import numpy as np
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
    verify(root);os.chdir(root);c=json.loads((root/'config.json').read_text());w,h=c['width'],c['height']
    def stage(name):
        (root/'stage.json').write_text(json.dumps(dict(stage=name,time=time.time()))+'\n');print(name,flush=True)
    stage('compile');cmd=['cslc','layout.csl','--arch=wse3',f'--fabric-dims={w+7},{h+2}','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1'];(root/'compile-command.json').write_text(json.dumps(cmd)+'\n');subprocess.run(cmd,check=True)
    runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=c['threads'],dump_core=True),SdkTarget.WSE3));ids={n:runner.get_id(n) for n in ('weights','input','output','progress','timing')}
    stage('load');runner.load();stage('run');runner.run();report=dict(success=False,cases=[],runtime_instances=1)
    try:
        values=np.load(root/'inputs.npz')
        if c['real_weights']:
            half=c.get('storage')=='bf16';matrix=values['weights_u16'] if half else values['weights']
            if c.get('colmajor'):matrix=matrix.T.copy()
            data=matrix.astype(np.uint32 if half else np.float32).ravel()
            runner.memcpy_h2d(ids['weights'],data,0,0,w,h,len(data),streaming=False,data_type=MemcpyDataType.MEMCPY_16BIT if half else MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
        for epoch in range(c['calls']):
            if c['real_weights']:
                data=values['input_batches'][epoch].astype(np.float32);runner.memcpy_h2d(ids['input'],data,0,0,w,h,len(data),streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
            stage('compute-'+str(epoch));runner.launch('compute',nonblock=False);raw={}
            points=sorted(set([(0,0),(w-1,0),(0,h-1),(w-1,h-1)])) if c.get('sampled') else None
            for name,n,bits in [('output',c['rows'],32),('progress',1,32),('timing',6,16)]:
                stage('read-'+name+'-'+str(epoch));parts=[]
                regions=[(x,y,1,1) for x,y in points] if points else [(0,0,w,h)]
                for x,y,rw,rh in regions:
                    data=np.zeros(rw*rh*n,np.float32 if name=='output' else np.uint32)
                    runner.memcpy_d2h(data,ids[name],x,y,rw,rh,n,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT if bits==32 else MemcpyDataType.MEMCPY_16BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);parts.append(data.reshape(rh,rw,n))
                raw[name]=np.stack([v[0,0] for v in parts]) if points else parts[0]
            np.savez(root/f'actual-{epoch}.npz',**raw)
            actual=raw['output'].astype(np.float64);expected=values['expected_batches'][epoch];assert np.isfinite(actual).all();difference=actual-expected
            l2=float(np.linalg.norm(difference,axis=-1).max()/max(np.linalg.norm(expected),1e-30));peak=float(np.max(np.abs(difference))/max(np.max(np.abs(expected)),1e-30))
            assert l2<=c['gate']['relative_l2'] and peak<=c['gate']['relative_peak'],(l2,peak)
            assert np.all(raw['progress']==epoch+1)
            t=raw['timing'].astype(np.uint64);assert np.all(t<65536)
            ticks=lambda v:v[...,0]+(v[...,1]<<16)+(v[...,2]<<32)
            cycles=(ticks(t[...,3:])-ticks(t[...,:3])) & np.uint64((1<<48)-1);assert np.all((cycles>0)&(cycles<1<<40))
            report['cases'].append(dict(epoch=epoch,relative_l2=l2,relative_peak=peak,min_cycles=int(cycles.min()),max_cycles=int(cycles.max()),observed_points=points,cycle_scope='sampled PE extrema only' if points else 'all application PEs',actual_sha256=sha(root/f'actual-{epoch}.npz')))
            (root/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:stage('stop');runner.stop()
    report['success']=True;(root/'results.json').write_text(json.dumps(report,indent=2)+'\n');stage('complete')


def process_identity(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text();return text[text.rfind(')')+2:].split()[19]
    except (FileNotFoundError,ProcessLookupError):return None


def cleanup_tree(known):
    signals=[]
    for sig in (signal.SIGTERM,signal.SIGKILL):
        live=[pid for pid,start in known.items() if start is not None and process_identity(pid)==start]
        if not live:break
        for pid in reversed(live):
            try:os.kill(pid,sig);signals.append(dict(pid=pid,starttime=known[pid],signal=int(sig)))
            except ProcessLookupError:pass
        time.sleep(2)
    return signals


def finalize_tree(process,known):
    signals=cleanup_tree(known)
    # Reap the Popen child before recording identities. A terminated zombie
    # still has /proc/stat until wait(), and must not look like a live leak.
    process.wait(timeout=5)
    return signals,{str(pid):process_identity(pid) for pid in known}


def execute(root,timeout,rss_limit_kib=24*1024*1024):
    if not isinstance(rss_limit_kib,int) or isinstance(rss_limit_kib,bool) or not 0<rss_limit_kib<=24*1024*1024:
        raise ValueError('Owned process-tree RSS limit must be positive and at most24GiB')
    m=verify(root);assert not (root/'sdk.log').exists();assert sha(SIF)==m['sdk_sha256']
    cmd=[os.environ.get('CSL_LLM_CS_PYTHON', 'cs_python'),str(root/'driver.py'),'--worker',str(root)]
    result=dict(success=False,sdk_sha256=m['sdk_sha256'],manifest_sha256=sha(root/'manifest.json'),command=cmd,timeout_seconds=timeout,rss_limit_kib=rss_limit_kib,rss_kind='Sum of process-tree resident pages, may double-count shared mappings',peak_rss_kib=0)
    start=time.monotonic();samples=[];known={}
    with (root/'sdk.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,SINGULARITYENV_CS_TARGET='SDR',SINGULARITYENV_PYTHONUNBUFFERED='1'))
        try:
            while process.poll() is None:
                rows=[]
                for line in subprocess.check_output(['ps','-eo','pid=,ppid=,rss='],text=True).splitlines():
                    try:rows.append(tuple(map(int,line.split())))
                    except ValueError:pass
                family={process.pid};prior=None
                while prior!=family:
                    prior=set(family);family.update(pid for pid,ppid,rss in rows if ppid in family)
                known.update({pid:process_identity(pid) for pid in family if pid not in known});rss=sum(rss for pid,ppid,rss in rows if pid in family);result['peak_rss_kib']=max(result['peak_rss_kib'],rss);samples.append(dict(elapsed=time.monotonic()-start,rss_kib=rss,processes=len(family)))
                if time.monotonic()-start>timeout or rss>rss_limit_kib:raise TimeoutError('Probe resource/time budget reached')
                time.sleep(1)
            assert process.returncode==0,process.returncode
            actual=json.loads((root/'results.json').read_text());assert actual['success'] is True
            result.update(success=True,results_sha256=sha(root/'results.json'))
        except BaseException:
            result['error']=traceback.format_exc()
            raise
        finally:
            result['observed_process_identities']={str(pid):start for pid,start in known.items()}
            result['cleanup_signals'],result['after_cleanup_identities']=finalize_tree(process,known)
            result['elapsed_seconds']=time.monotonic()-start;(root/'resource-samples.json').write_text(json.dumps(samples)+'\n');result['samples_sha256']=sha(root/'resource-samples.json');(root/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SDK PROBE PASS',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path);p.add_argument('--execute',type=Path);p.add_argument('--timeout',type=int,default=1800);a=p.parse_args()
    if a.worker:worker(a.worker.resolve())
    else:execute(a.execute.resolve(),a.timeout)
