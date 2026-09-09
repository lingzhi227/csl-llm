"""Actual two-layer ELF symbols for numerical and transport diagnostics.

The pinned compiler emitted two-byte CSL bool objects in actual compile224126.
Require that observed ABI explicitly; do not infer bool storage from Python/C.
"""
import hashlib
from pathlib import Path


def requirements(point, tile, controller=False, *, endpoint=None):
    """Shared decoder ABI; full-model callers explicitly disable the probe endpoint.

    Coordinate128,0 is the diagnostic endpoint only in the two-layer layout.
    In the complete model it is an ordinary layer2 matrix PE.
    """
    if endpoint is None:
        endpoint = point == (128, 0)
    if type(endpoint) is not bool:
        raise ValueError('Endpoint role must be explicit boolean or inferred for the chain')
    fields = dict(progress=(4, 4), status=(8, 4), timing=(12, 2))
    if endpoint:
        fields.update(input=(3584, 4), output=(3584, 4))
    else:
        matrix = tile is not None and tile.operation != 'kv'
        fields.update(identity=(12, 4), weights=(28672 if matrix else 4, 4),
                      norms=(3584 if controller else 4, 4),
                      frequency=(128 if controller else 4, 4),
                      bias=(256 if tile is not None and tile.bias_tensor else 4, 4))
        if controller:
            fields.update(input=(3584, 4), hidden=(3584, 4), output=(3584, 4), activation=(19712, 4))
        if matrix:
            fields.update(input=(448, 4), result=(512, 4))
            if tile.operation == 'gate' and tile.ordinal == 0:
                fields['partial'] = (512, 4)
        if tile is not None and tile.operation == 'kv':
            fields.update(input=(2304, 4), result=(1820, 4))
            fields.update({'kv.keys': (8192, 4), 'kv.values': (8192, 4)})
            fields.update({'kv.seen': (2, 2), 'kv.valid': (2, 2)})
            if tile.ordinal == 0:
                fields['context'] = (1792, 4)
    if endpoint or controller:
        fields['boundary.invocation'] = (4, 4)
        for name in ('received_count', 'sent_count', 'pending_count', 'command', 'started', 'sending'):
            fields['boundary.'+name] = (2, 2)
    # Non-root PEs do not send packets and the compiler can remove that flag.
    if endpoint or controller or (tile is not None and tile.ordinal == 0):
        fields.update({'packets.sending': (2, 2), 'packets.receiving': (2, 2)})
    return fields


def audit(directory):
    from cerebras.elf.cself import ELFMemory
    from elftools.elf.elffile import ELFFile
    from csl_llm.decoder_layout import tiles
    from model_symbols import resolve
    owners = {(tile.x, tile.y): tile for layer in (0, 1) for tile in tiles(layer)}
    controllers = {(62, 0), (126, 0)}
    universe = {(x+4, y+1) for y in range(20) for x in range(129)}
    seen, records = set(), []
    for path in sorted(Path(directory).glob('*.elf')):
        memory = ELFMemory(str(path))
        assert tuple(memory.get_fabric_dimensions()) == (136, 22)
        points = list(map(tuple, memory.iter_coordinates()))
        if not points or len(set(points)) != len(points) or seen & set(points) or not set(points) <= universe:
            raise ValueError('Invalid or overlapping chain symbol ownership')
        seen.update(points)
        with path.open('rb') as stream:
            elf = ELFFile(stream)
            assert elf.little_endian and elf.elfclass == 64
            table = elf.get_section_by_name('.symtab')
            assert table is not None
            objects = [dict(name=s.name, address=int(s['st_value']), size=int(s['st_size']))
                       for s in table.iter_symbols() if s['st_info']['type'] == 'STT_OBJECT' and s['st_size']]
        symbols, aliases = {}, {}
        for px, py in points:
            point = (px-4, py-1)
            for logical, (size, alignment) in requirements(point, owners.get(point), point in controllers).items():
                row = resolve(objects, logical, size, alignment=alignment)
                symbols[row['name']] = dict(address=row['address'], size=size)
                aliases[logical] = row['name']
        records.append(dict(elf=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            coordinates=[list(point) for point in points],
                            required_symbols=symbols, logical_symbols=aliases))
    if seen != universe:
        raise ValueError('Incomplete chain symbol coverage')
    return dict(passed=True, application_coordinates=2580, images=records,
                tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Actual ELF ABI for two-layer diagnostic reading; no core or neural acceptance.')
