"""Actual stopped simulator memory; compiled ELF supplies addresses only.

This is a validation/debugging path, not deployed device I/O. Never overlay
initial ELF contents on the runtime core or silently fill missing core bytes.
The caller must verify the frozen compiler-file hashes before construction;
this low-level reader validates memory structure, not artifact provenance.
"""
from pathlib import Path


class StoppedCore:
    def __init__(self, core_path, binary_directory):
        from cerebras.elf.cself import ELFMemory
        from elftools.elf.elffile import ELFFile
        self.path = Path(core_path)
        with self.path.open('rb') as stream:
            if ELFFile(stream)['e_type'] != 'ET_CORE':
                raise ValueError('Actual SDK ET_CORE required')
        self.memory = ELFMemory(str(self.path))
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

    def read(self, x, y, name, expected_bytes):
        address, size = self.symbols[(x, y)][name]
        if size != expected_bytes:
            raise ValueError(f'Unexpected symbol size: {(x,y,name,size,expected_bytes)}')
        raw = self.memory.read_data(x, y, address, size)
        if len(raw) != size or any(value is None for value in raw):
            raise ValueError(f'Core does not back the complete symbol: {(x,y,name)}')
        return bytes(raw)
