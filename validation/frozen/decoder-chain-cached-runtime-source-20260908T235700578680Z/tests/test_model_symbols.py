"""Private compiler names must be observed uniquely, never guessed."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from model_symbols import resolve


def obj(name, address=16, size=3584):
    return dict(name=name, address=address, size=size)


def test_exact_and_observed_compiler_prefix():
    for name in ('input', '$$csl_base_address$$3$$input'):
        assert resolve([obj(name)], 'input', 3584)['name'] == name
    assert resolve([obj('sequence.prompt', size=8192)], 'sequence.prompt', 8192)['size'] == 8192


def test_ambiguity_and_invalid_ranges_fail_closed():
    for objects in ([], [obj('other.input')], [obj('input'), obj('$$csl_base_address$$0$$input')],
                    [obj('input', address=17)], [obj('input', address=49000)], [obj('input', size=4)]):
        with pytest.raises(ValueError):
            resolve(objects, 'input', 3584)


def test_explicit_protocol_alignment_preserves_default_strictness():
    row = obj('boundary.command', address=18, size=2)
    assert resolve([row], 'boundary.command', 2, alignment=2) == row
    with pytest.raises(ValueError):
        resolve([row], 'boundary.command', 2)
    for alignment in (True, 0, 3, 8):
        with pytest.raises(ValueError):
            resolve([row], 'boundary.command', 2, alignment=alignment)
