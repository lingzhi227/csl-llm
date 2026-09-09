"""Explicit local data-module bindings for generated CSL placement.

These names describe declarations, not proof of original data or SDK loading.
The freezing tool must verify the source artifact and its complete file hashes.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class StaticData:
    weight_file: str | None = None
    auxiliary_file: str | None = None

    def __post_init__(self):
        if self.weight_file is None and self.auxiliary_file is None:
            raise ValueError('A static binding requires a data module')
        for name in [self.weight_file, self.auxiliary_file]:
            if name is not None and (not isinstance(name, str) or
                    re.fullmatch(r'[A-Za-z0-9_]+\.csl', name) is None):
                raise ValueError('Static data modules must be simple local CSL filenames')


def bindings(value):
    result = {} if value is None else dict(value)
    for point, data in result.items():
        if (not isinstance(point, tuple) or len(point) != 2 or
                any(type(x) is not int or x < 0 for x in point) or
                not isinstance(data, StaticData)):
            raise ValueError('Static data requires integer coordinates and typed bindings')
    return result
