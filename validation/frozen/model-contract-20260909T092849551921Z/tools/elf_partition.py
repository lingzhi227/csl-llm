"""Experimental composition of intact compiler-produced coordinate partitions.

Never patches ELF contents or crops their coordinate metadata. Compatibility
must be established by an actual SDK run; this is not an official SDK linker.
"""
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def image(path):
    from compare_elf_images import image as inspect
    from cerebras.elf.cself import ELFMemory
    value = inspect(path)
    value['coordinates'] = sorted(value['coordinates'])
    value['sections'] = sorted(value['sections'], key=lambda row: (row['address'], row['name']))
    memory = ELFMemory(str(path))
    value['fabric_dimensions'] = memory.get_fabric_dimensions()
    value['architecture_flags'] = memory.get_flags()
    # The SDK loader uses ordered block/scatter segments, not just sections.
    # Preserve segment order, extent and delta even when section bytes agree.
    value['loader_segments'] = [dict(address=s.addr, extent=s.extent,
        memory_size=s.mem_size, data_size=s.data_size, delta=s.delta,
        flags=s.p_flags, bytes_sha256=hashlib.sha256(s.data).hexdigest())
        for s in memory.iter_segments()]
    return value


def rpc(path):
    value = json.loads(path.read_text())
    symbols = value['rpc_symbols']
    if len({s['id'] for s in symbols}) != len(symbols) or len({s['name'] for s in symbols}) != len(symbols):
        raise ValueError('RPC IDs and names must be unique')
    return dict(value, rpc_symbols=sorted(symbols, key=lambda s: s['id']))


def io_signature(path):
    files = {}
    for name in ['west', 'east']:
        root = path / name
        if not root.is_dir():
            raise ValueError('Both compiled SDK I/O directories are required')
        files[name] = dict(
            metadata={str(p.relative_to(root)): json.loads(p.read_text()) for p in root.rglob('*.json')},
            images=sorted((image(p) for p in root.rglob('*.elf')),
                          key=lambda value: json.dumps(value['coordinates'])))
        if not files[name]['images']:
            raise ValueError('Missing actual I/O ELF images')
    return files


def compose(partitions, output, expected_coordinates):
    """Each (compiled directory, physical coordinate set) supplies whole ELFs."""
    if output.exists() or not partitions:
        raise ValueError('Composition needs new output and nonempty partitions')
    expected_coordinates = set(expected_coordinates)
    owner = set()
    selected = []
    reference = Path(partitions[0][0])
    metadata = json.loads((reference / 'out.json').read_text())
    rpc_metadata = rpc(reference / 'bin/out_rpc.json')
    io = io_signature(reference)
    io_coordinates = {tuple(point) for bank in io.values()
                      for img in bank['images'] for point in img['coordinates']}
    if io_coordinates & expected_coordinates:
        raise ValueError('Application ownership overlaps compiled SDK I/O coordinates')
    for part, (directory, wanted) in enumerate(partitions):
        directory = Path(directory)
        wanted = set(wanted)
        if not wanted or owner & wanted or not wanted <= expected_coordinates:
            raise ValueError('Partition ownership must be nonempty, disjoint and in bounds')
        if json.loads((directory / 'out.json').read_text()) != metadata:
            raise ValueError('Compilation parameter/color metadata differs')
        if rpc(directory / 'bin/out_rpc.json') != rpc_metadata:
            raise ValueError('RPC symbol IDs/types/exports differ')
        if io_signature(directory) != io:
            raise ValueError('SDK I/O allocated bytes, coordinates or metadata differ')
        covered = set()
        for path in sorted((directory / 'bin').glob('*.elf')):
            compiled_image = image(path)
            coordinates = {tuple(point) for point in compiled_image['coordinates']}
            if coordinates & wanted:
                if not coordinates <= wanted or covered & coordinates:
                    raise ValueError('ELF straddles partitions or repeats an owned coordinate')
                covered |= coordinates
                selected.append((part, path, compiled_image))
        if covered != wanted:
            raise ValueError('Compiler ELFs do not cover the declared partition')
        owner |= wanted
    if owner != expected_coordinates:
        raise ValueError('Incomplete application coordinate coverage')
    output.mkdir()
    (output / 'bin').mkdir()
    shutil.copy2(reference / 'out.json', output / 'out.json')
    shutil.copy2(reference / 'bin/out_rpc.json', output / 'bin/out_rpc.json')
    for name in ['west', 'east']:
        shutil.copytree(reference / name, output / name)
    records = []
    for index, (part, path, compiled_image) in enumerate(selected):
        destination = output / 'bin' / f'out_{index}_0.elf'
        shutil.copy2(path, destination)
        source_sha = digest(path)
        if digest(destination) != source_sha:
            raise ValueError('ELF copy changed bytes')
        family = path.stem.rsplit('_', 1)[0]
        graphs = list(path.parent.glob(family+'.*.dot'))
        if not graphs:
            raise ValueError('Missing source allocation graphs')
        for graph in graphs:
            shutil.copy2(graph, output / 'bin' / f'out_{index}.{graph.name.rsplit(".", 2)[1]}.dot')
        records.append(dict(partition=part, source=str(path), output=destination.name,
                            sha256=source_sha, allocated_image=compiled_image))
    return dict(passed=True, elf_count=len(records), coordinate_count=len(owner),
                images=records, io_equal=True, rpc_equal=True,
                artifact_sha256={str(p.relative_to(output)): digest(p)
                                 for p in output.rglob('*') if p.is_file()},
                scope='Intact compiler ELF file selection/copy with unique full declared coordinate ownership and equal SDK I/O/RPC. No ELF edits. Experimental artifact composition; actual SDK compatibility and neural correctness require separate execution.')
