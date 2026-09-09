"""One read-only process-tree snapshot for an identified active SDK executor."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def identity(pid):
    text = (Path('/proc') / str(pid) / 'stat').read_text()
    return text[text.rfind(')')+2:].split()[19]


def observe(bundle):
    roots = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            args = (path / 'cmdline').read_bytes().split(b'\0')
            if b'--execute' in args and any(arg.endswith((bundle.name+'/driver.py').encode()) for arg in args):
                roots.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if len(roots) != 1:
        raise ValueError(f'Expected one owned active executor, found {roots}')
    root = roots[0]
    started = identity(root)
    rows = [tuple(map(int, line.split())) for line in subprocess.check_output(['ps', '-eo', 'pid=,ppid=,rss='], text=True).splitlines()]
    family, old = {root}, None
    while family != old:
        old = set(family)
        family.update(pid for pid, parent, _ in rows if parent in family)
    members = []
    for pid, parent, rss in rows:
        if pid not in family:
            continue
        path = Path('/proc') / str(pid)
        try:
            starttime = identity(pid)
            name = (path / 'comm').read_text().strip()
            pss = None
            try:
                for line in (path / 'smaps_rollup').read_text().splitlines():
                    if line.startswith('Pss:'):
                        pss = int(line.split()[1])
            except PermissionError:
                pass
            if identity(pid) != starttime:
                raise ValueError('Process identity changed during snapshot')
            members.append(dict(pid=pid, parent=parent, starttime=starttime, name=name, rss_kib=rss, pss_kib=pss))
        except (FileNotFoundError, ProcessLookupError):
            members.append(dict(pid=pid, parent=parent, exited_during_snapshot=True, rss_kib=rss))
    if identity(root) != started:
        raise ValueError('Executor changed during snapshot')
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    return dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), root_pid=root,
                root_starttime=started, members=members, sum_rss_kib=sum(r['rss_kib'] for r in members),
                mem_available_kib=available,
                stage=json.loads((bundle / 'stage.json').read_text()) if (bundle / 'stage.json').exists() else None,
                application_elf_count=len(list((bundle / 'out/bin').glob('*.elf'))),
                manifest_sha256=hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest(),
                tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Read-only sampled process tree, not an execution budget change. RSS may count shared pages repeatedly; per-process PSS is reported when accessible. Snapshot is not a peak or exact compiler pass attribution.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    value = observe(args.bundle.resolve())
    with args.report.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print('SDK RESOURCE SNAPSHOT', value['sum_rss_kib'], 'KiB', len(value['members']), 'processes')
