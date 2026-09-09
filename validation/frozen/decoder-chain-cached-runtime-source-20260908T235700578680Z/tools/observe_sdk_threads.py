"""Read-only Linux CPU deltas for one explicitly identified SDK worker."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time


def snapshot(pid):
    root = Path('/proc') / str(pid)
    stat = (root / 'stat').read_text()
    identity = stat[stat.rfind(')')+2:].split()[19]
    threads = {}
    for path in (root / 'task').iterdir():
        try:
            text = (path / 'stat').read_text()
            values = text[text.rfind(')')+2:].split()
            threads[path.name] = dict(name=text[text.find('(')+1:text.rfind(')')],
                starttime=values[19], user_ticks=int(values[11]), system_ticks=int(values[12]),
                state=values[0], processor=int(values[36]))
        except FileNotFoundError:
            continue
    return identity, threads


def observe(pid, expected_identity, duration):
    if not 1 <= duration <= 30:
        raise ValueError('Read-only sample duration must be1..30seconds')
    identity, before = snapshot(pid)
    if identity != expected_identity:
        raise ValueError('PID no longer belongs to the recorded SDK worker')
    start = time.monotonic()
    time.sleep(duration)
    identity, after = snapshot(pid)
    elapsed = time.monotonic()-start
    if identity != expected_identity:
        raise ValueError('SDK worker identity changed during sampling')
    ticks = os.sysconf('SC_CLK_TCK')
    rows = []
    for tid, end in after.items():
        begin = before.get(tid)
        if begin is None or begin['starttime'] != end['starttime']:
            continue
        user = (end['user_ticks']-begin['user_ticks'])/ticks
        system = (end['system_ticks']-begin['system_ticks'])/ticks
        rows.append(dict(tid=int(tid), **end, user_seconds=user, system_seconds=system,
                         cpu_fraction=(user+system)/elapsed))
    return dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        pid=pid, process_starttime=identity, elapsed_seconds=elapsed,
        observed_before=len(before), observed_after=len(after),
        threads=sorted(rows, key=lambda row: -row['cpu_fraction']),
        aggregate_cpu_cores=sum(row['cpu_fraction'] for row in rows),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Read-only CPU deltas for stable thread identities. CPU activity is not useful simulator progress or evidence against a protocol deadlock; excludes threads created/exited during the sample.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pid', type=int)
    parser.add_argument('starttime')
    parser.add_argument('report', type=Path)
    parser.add_argument('--seconds', type=int, default=10)
    args = parser.parse_args()
    result = observe(args.pid, args.starttime, args.seconds)
    with args.report.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('SAMPLED CPU CORES', result['aggregate_cpu_cores'])
