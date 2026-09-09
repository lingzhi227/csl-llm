"""Summarize observed SDK calls; call latency is not device compute latency."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(events):
    pending, completed, previous = None, [], -1.0
    fields = ('operation', 'symbol', 'rectangle', 'count_per_pe')
    for event in events:
        timestamp = event['elapsed_seconds']
        if timestamp < previous:
            raise ValueError('Non-monotonic API event timestamps')
        previous = timestamp
        if event['event'] == 'begin':
            if pending is not None:
                raise ValueError('Overlapping calls in synchronous API log')
            pending = event
        elif event['event'] == 'end':
            if pending is None or any(event[k] != pending[k] for k in fields):
                raise ValueError('API completion does not match its begin record')
            duration = event['call_seconds']
            if duration < 0 or abs(duration-(timestamp-pending['elapsed_seconds'])) > .01:
                raise ValueError('API elapsed time does not match event interval')
            completed.append(event)
            pending = None
        else:
            raise ValueError('Unknown API event kind')
    groups = {}
    for event in completed:
        key = (event['operation'], event['symbol'], tuple(event['rectangle']), event['count_per_pe'])
        row = groups.setdefault(key, dict(operation=key[0], symbol=key[1], rectangle=key[2], count_per_pe=key[3],
                                         calls=0, total_seconds=0.0, max_seconds=0.0))
        row['calls'] += 1
        row['total_seconds'] += event['call_seconds']
        row['max_seconds'] = max(row['max_seconds'], event['call_seconds'])
    return dict(completed_calls=len(completed), pending_call=pending,
                groups=sorted(groups.values(), key=lambda r: r['total_seconds'], reverse=True),
                scope='Observed synchronous SDK API wall time, including pending device work, command/transfer latency and simulator scheduling. Not device cycles or standalone kernel latency. An unmatched final begin is retained, not reported as a completed call.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('events', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    raw = args.events.read_bytes()
    value = summarize([json.loads(line) for line in raw.splitlines() if line])
    value['events_sha256'] = hashlib.sha256(raw).hexdigest()
    value['tool_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.report.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print('COMPLETED API CALLS', value['completed_calls'], 'PENDING', value['pending_call'] is not None)
