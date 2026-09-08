"""Actual stopped simulator memory; compiled ELF supplies addresses only.

This is a validation/debugging path, not deployed device I/O. Never overlay
initial ELF contents on the runtime core or silently fill missing core bytes.
The caller must verify the frozen compiler-file hashes before construction;
this low-level reader validates memory structure, not artifact provenance.
"""
from pathlib import Path


def contract_symbols(contract, composition):
    """Validate the relation between two already hash-verified static reports.

    This does not re-audit ELF symbol tables. The caller must bind both reports
    and the actual runtime compiler files to the accepted assembly receipts.
    """
    if contract.get('passed') is not True or composition.get('passed') is not True:
        raise ValueError('Completed symbol and composition reports are required')
    for count in [contract['application_coordinates'], composition['coordinate_count']]:
        if type(count) is not int or count <= 0:
            raise ValueError('A nonempty application is required')
    images = composition['images']
    compiled = {row['output']: row for row in images}
    if len(compiled) != len(images) or not compiled or composition['elf_count'] != len(images):
        raise ValueError('Incomplete or repeated composition images')
    metadata = None
    symbols = {}
    used = set()
    for row in contract['images']:
        name = row['elf']
        if name in used or name not in compiled:
            raise ValueError('Unexpected symbol-contract image')
        used.add(name)
        original = compiled[name]
        if (row['sha256'] != original['sha256'] or
                composition['artifact_sha256'].get('bin/'+name) != row['sha256']):
            raise ValueError('Symbol contract and compiled artifact hashes differ')
        image = original['allocated_image']
        current = (tuple(image['fabric_dimensions']), image['architecture_flags'])
        if (len(current[0]) != 2 or any(type(x) is not int or x <= 0 for x in current[0])
                or type(current[1]) is not int):
            raise ValueError('Invalid compiled fabric metadata')
        if metadata is None:
            metadata = current
        if current != metadata:
            raise ValueError('Mixed compiled fabric or architecture')
        coordinates = [tuple(point) for point in row['coordinates']]
        expected = [tuple(point) for point in image['coordinates']]
        if (not coordinates or len(set(coordinates)) != len(coordinates) or
                len(set(expected)) != len(expected) or set(coordinates) != set(expected)):
            raise ValueError('Symbol and composition coordinate ownership differs')
        entries = {}
        for symbol, value in row['required_symbols'].items():
            address, size = value['address'], value['size']
            if (not isinstance(symbol, str) or not symbol or type(address) is not int or
                    type(size) is not int or address < 0 or size <= 0 or address+size > 49152):
                raise ValueError('Invalid contracted symbol range')
            entries[symbol] = (address, size)
        if not entries:
            raise ValueError('Empty symbol contract')
        for point in coordinates:
            if (len(point) != 2 or any(type(x) is not int for x in point) or
                    not (0 <= point[0] < current[0][0] and 0 <= point[1] < current[0][1]) or
                    point in symbols):
                raise ValueError('Invalid or repeated application ownership')
            symbols[point] = entries
    if (used != set(compiled) or len(symbols) != composition['coordinate_count'] or
            len(symbols) != contract['application_coordinates']):
        raise ValueError('Incomplete full symbol contract')
    return symbols, metadata


class StoppedCore:
    def _open_core(self, core_path):
        from cerebras.elf.cself import ELFMemory
        from elftools.elf.elffile import ELFFile
        self.path = Path(core_path)
        with self.path.open('rb') as stream:
            if ELFFile(stream)['e_type'] != 'ET_CORE':
                raise ValueError('Actual SDK ET_CORE required')
        self.memory = ELFMemory(str(self.path))

    def __init__(self, core_path, binary_directory):
        from cerebras.elf.cself import ELFMemory
        from elftools.elf.elffile import ELFFile
        self._open_core(core_path)
        self.symbols = {}
        for path in sorted(Path(binary_directory).glob('*.elf')):
            compiled = ELFMemory(str(path))
            if (compiled.get_flags() != self.memory.get_flags() or
                    compiled.get_fabric_dimensions() != self.memory.get_fabric_dimensions()):
                raise ValueError(f'Core and compiled fabric differ: {path}')
            with path.open('rb') as stream:
                table = ELFFile(stream).get_section_by_name('.symtab')
                if table is None:
                    raise ValueError(f'ELF symbols required: {path}')
                symbols = {}
                for symbol in table.iter_symbols():
                    if symbol['st_info']['type'] == 'STT_OBJECT' and symbol['st_size']:
                        name = symbol.name
                        entry = (int(symbol['st_value']), int(symbol['st_size']))
                        if name in symbols and symbols[name] != entry:
                            raise ValueError(f'Ambiguous symbol: {path}:{name}')
                        symbols[name] = entry
            for coordinate in compiled.iter_coordinates():
                coordinate = tuple(coordinate)
                if coordinate in self.symbols:
                    raise ValueError(f'Overlapping compiler images: {coordinate}')
                self.symbols[coordinate] = symbols
        if not self.symbols:
            raise ValueError('No compiled application images')

    @classmethod
    def from_verified_contract(cls, core_path, contract, composition):
        """Read an actual core using previously audited, hash-bound addresses.

        The caller verifies report hashes and actual runtime compiled-file
        receipts first. No initial ELF bytes or fallback symbol names are used.
        """
        symbols, metadata = contract_symbols(contract, composition)
        instance = cls.__new__(cls)
        instance._open_core(core_path)
        if (tuple(instance.memory.get_fabric_dimensions()) != metadata[0] or
                instance.memory.get_flags() != metadata[1]):
            raise ValueError('Actual core differs from the contracted fabric')
        instance.symbols = symbols
        return instance

    def read(self, x, y, name, expected_bytes):
        address, size = self.symbols[(x, y)][name]
        if size != expected_bytes:
            raise ValueError(f'Unexpected symbol size: {(x,y,name,size,expected_bytes)}')
        raw = self.memory.read_data(x, y, address, size)
        if len(raw) != size or any(value is None for value in raw):
            raise ValueError(f'Core does not back the complete symbol: {(x,y,name)}')
        return bytes(raw)
