"""Verify the published source payload without modifying it."""
import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'release/MANIFEST.json').read_text())
for item in manifest['files']:
    path = root / item['path']
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['path']
print('Verified', len(manifest['files']), 'published files')
