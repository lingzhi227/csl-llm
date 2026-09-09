"""Fetch a pinned official checkpoint and verify Git/LFS content identities."""
import argparse, datetime, hashlib, json, os, urllib.request
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(target, destination):
    config = json.loads(target.read_text())
    destination.mkdir(parents=True, exist_ok=False)
    base = 'https://huggingface.co/' + config['model']
    api = 'https://huggingface.co/api/models/' + config['model'] + '/tree/' + config['revision'] + '?recursive=false&expand=true'
    with urllib.request.urlopen(api, timeout=60) as response:
        tree_bytes = response.read()
    (destination / 'upstream-tree.json').write_bytes(tree_bytes)
    tree = json.loads(tree_bytes)
    records = []
    for item in tree:
        if item['type'] != 'file':
            raise ValueError('Unexpected directory in checkpoint')
        name = item['path']
        assert Path(name).name == name
        path = destination / name
        url = base + '/resolve/' + config['revision'] + '/' + name
        partial = path.with_name(path.name + '.partial')
        print('Downloading', name, item['size'], flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, partial.open('xb') as out:
            while chunk := response.read(4 * 1024 * 1024):
                out.write(chunk)
        assert partial.stat().st_size == item['size'], name
        sha = digest(partial)
        if 'lfs' in item:
            assert sha == item['lfs']['oid'], name
        else:
            blob = partial.read_bytes()
            git_sha = hashlib.sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest()
            assert git_sha == item['oid'], name
        if name in config['files_sha256']:
            assert sha == config['files_sha256'][name], name
        os.rename(partial, path)
        records.append(dict(path=name, bytes=item['size'], sha256=sha, upstream_identity=item.get('lfs', {}).get('oid', item['oid'])))
    manifest = dict(success=True, model=config['model'], revision=config['revision'], acquired_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), files=records, tree_sha256=digest(destination/'upstream-tree.json'), acquisition_driver_sha256=digest(Path(__file__)), target_sha256=digest(target), scope='Official checkpoint bytes only; not numerical or SDK acceptance.')
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('CHECKPOINT VERIFIED', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('target',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();fetch(a.target,a.destination)
