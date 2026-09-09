"""Initial resident decoder transfers in SDK row-major raw-u32 format.

Input tensors are original BF16 *bits*, never rounded numerical substitutes.
This module does no inference and does not issue runtime calls. Its caller must
retain each buffer through SDK completion and verify raw readback independently.
"""
from dataclasses import dataclass
import numpy as np
from .decoder_layout import controller_parameters, geometry, model_origin, tiles


@dataclass(frozen=True)
class Transfer:
    """Application coordinates; SDK fabric offsets are applied by the runtime.

    The contiguous array is borrowed, not made immutable by this dataclass.
    Its owner must not change it before the corresponding SDK transfer ends.
    """
    symbol: str
    x: int
    y: int
    width: int
    height: int
    words_per_pe: int
    values: np.ndarray

    def __post_init__(self):
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError('SDK transfer requires an exported symbol name')
        for name in ['x', 'y', 'width', 'height', 'words_per_pe']:
            value = getattr(self, name)
            minimum = 0 if name in ['x', 'y'] else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f'SDK transfer {name} must be an integer at least {minimum}')
        if (not isinstance(self.values, np.ndarray) or self.values.dtype != np.uint32
                or not self.values.flags.c_contiguous):
            raise ValueError('SDK initialization requires contiguous raw-u32 storage')
        if self.values.shape != (self.height, self.width, self.words_per_pe):
            raise ValueError('Transfer shape must exactly match its application rectangle')


def pairs(bits):
    """First BF16 half occupies the low16 bits, matching the validated kernel."""
    if bits.dtype != np.uint16 or bits.size % 2:
        raise ValueError('Original BF16 storage must be uint16 with an even extent')
    halves = np.ascontiguousarray(bits).reshape(-1, 2).astype(np.uint32)
    return halves[:, 0] | (halves[:, 1] << 16)


def layer_transfers(tensors, frequency, layer, origin=None):
    default_origin = model_origin(layer)
    origin = default_origin if origin is None else origin
    geometry(origin)
    bx, by = origin
    records = tiles(layer, origin)
    owners = {(t.x, t.y): t for t in records if t.operation != 'kv'}
    control = controller_parameters(layer, origin)

    def tensor(name, shape):
        value = tensors[name]
        if value.dtype != np.uint16 or value.shape != shape:
            raise ValueError(f'Original BF16 tensor {name} must have shape {shape}')
        return value

    identity = np.zeros((20, 64, 3), np.uint32)
    identity[:, :, 1:] = control['coordinate']
    for t in records:
        identity[t.y-by, t.x-bx] = [t.packet_group, *t.destination]
    yield Transfer('identity', bx, by, 64, 20, 3, identity)

    # These three dense rectangles cover every matrix PE and no KV/guard PE.
    # The latter have only a one-word placeholder, so padding across them would
    # corrupt SRAM. Each local tile is column-major128x112, including DOWN tail.
    covered = set()
    for x, y, width, height in [(bx, by, 32, 19), (bx+32, by, 22, 14), (bx+54, by, 8, 16)]:
        values = np.zeros((height, width, 7168), np.uint32)
        for row in range(height):
            for column in range(width):
                point = x+column, y+row
                t = owners[point]
                if point in covered:
                    raise ValueError('Duplicate matrix initialization')
                covered.add(point)
                shape = (4864, 896) if t.operation in ('up', 'gate') else (896, 4864) if t.operation == 'down' else (128, 896) if t.operation in ('k', 'v') else (896, 896)
                original = tensor(t.tensor, shape)
                tile = np.zeros((112, 128), np.uint16)
                tile[:t.valid_columns] = original[t.row:t.row+128, t.column:t.column+t.valid_columns].T
                values[row, column] = pairs(tile)
        yield Transfer('weights', x, y, width, height, 7168, values)
    if covered != set(owners):
        raise ValueError('Missing matrix initialization')

    bias = np.zeros((9, 1, 64), np.uint32)
    for t in records:
        if t.bias_tensor:
            shape = (896,) if t.operation == 'q' else (128,)
            original = tensor(t.bias_tensor, shape)
            bias[t.y-by, 0] = pairs(original[t.row:t.row+128])
    yield Transfer('bias', bx+54, by, 1, 9, 64, bias)
    norms = np.concatenate([tensor(name, (896,)) for name in control['norms']])
    cx, cy = control['coordinate']
    yield Transfer('norms', cx, cy, 1, 1, 896, pairs(norms).reshape(1, 1, 896))
    if frequency.dtype != np.float32 or frequency.shape != (32,) or not np.isfinite(frequency).all():
        raise ValueError('Pinned RoPE frequency must contain32 finite FP32 values')
    yield Transfer('frequency', cx, cy, 1, 1, 32, np.ascontiguousarray(frequency).view(np.uint32).reshape(1, 1, 32))
