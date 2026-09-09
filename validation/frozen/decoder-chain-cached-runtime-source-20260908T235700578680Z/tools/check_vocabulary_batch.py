"""Read-only dispatch gate for the independently reviewed serial batch queue."""
import argparse
import json
from pathlib import Path
import subprocess
import re

from sdk_probe import verify, sha, process_identity


def check(completed, candidate):
    verify(completed)
    execution = json.loads((completed / 'execution.json').read_text())
    result = json.loads((completed / 'results.json').read_text())
    assert execution['success'] and result['success'] and result['compile_only']
    assert result['all_matrix_coordinates_exact']
    assert execution['manifest_sha256'] == sha(completed / 'manifest.json')
    assert execution['results_sha256'] == sha(completed / 'results.json')
    assert all(value is None for value in execution['after_cleanup_identities'].values())
    assert all(process_identity(int(pid)) != start
               for pid, start in execution['observed_process_identities'].items())
    resources = json.loads((completed / 'elf-memory.json').read_text())
    dsr = json.loads((completed / 'dsr-graphs.json').read_text())
    for report in [resources, dsr]:
        assert report['passed'] and report['manifest_sha256'] == sha(completed / 'manifest.json')
    images = {p.name: sha(p) for p in (completed / 'out/bin').glob('*.elf')}
    assert images == {row['elf']: row['sha256'] for row in resources['classes']} == dsr['elf_sha256']
    assert all(images[row['file']] == row['sha256'] for row in result['images'])
    assert resources['max_static_high_water_bytes'] <= 49152
    verify(candidate)
    assert not (candidate / 'sdk.log').exists() and not (candidate / 'execution.json').exists()
    config = json.loads((candidate / 'config.json').read_text())
    assert config['timeout_seconds'] == 900 and config['rss_limit_kib'] == 12*1024*1024
    available = int(next(line for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')).split()[1])
    assert available > 14*1024*1024
    # Only this project's actual frozen driver command lines count. Other
    # projects and unrelated jobs are never signalled or modified.
    pattern = re.compile(re.escape(str(completed.parent)) + r'/[^/\s]+/driver\.py\s+--(?:execute|worker|case-worker|variant-worker)\b')
    processes = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True)
    assert not any(pattern.search(line) for line in processes.splitlines()), 'Another owned execution is active'
    return dict(passed=True, completed=completed.name, candidate=candidate.name,
        completed_execution_sha256=sha(completed / 'execution.json'),
        candidate_manifest_sha256=sha(candidate / 'manifest.json'), available_kib=available,
        scope='Actual completion, frozen hashes, resource reports, cleanup and current host memory only. Candidate source review is a separate prerequisite; this does not qualify full model inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('completed', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.completed.resolve(), args.candidate.resolve()), indent=2))
