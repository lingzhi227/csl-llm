"""Initial raw-bit transfers for the full resident model, without SDK calls.

This is an optional SDK loader plan, not a claim that full-fabric transfers are
fast enough in simulation. Static initialization can bypass matrix transfers.
The caller owns buffer lifetime and must establish real device completion.
"""
import numpy as np
from .decoder_storage import Transfer, layer_transfers, pairs

MODEL_CONTROLLER = (192, 160)


def vocabulary_transfers(weights, norm, rows_per_chunk=1):
    if weights.dtype != np.uint16 or weights.shape != (151936, 896):
        raise ValueError('Tied vocabulary must contain original151936x896 BF16 bits')
    if norm.dtype != np.uint16 or norm.shape != (896,):
        raise ValueError('Final norm must contain original896 BF16 bits')
    if (not isinstance(rows_per_chunk, int) or isinstance(rows_per_chunk, bool)
            or not 1 <= rows_per_chunk <= 49):
        raise ValueError('Vocabulary transfer chunk must contain1..49 rows')
    identity = np.zeros((50, 193, 1), np.uint32)
    for index in range(9496):
        y, x = divmod(index, 192)
        identity[y, x, 0] = index // 8
    yield Transfer('identity', 0, 160, 193, 50, 1, identity)

    rectangles = [(y, 192, min(rows_per_chunk, 49-y))
                  for y in range(0, 49, rows_per_chunk)] + [(49, 88, 1)]
    for y, width, height in rectangles:
        values = np.empty((height, width, 7168), np.uint32)
        for row in range(height):
            for x in range(width):
                group = (y+row)*24+x//8
                column = (x % 8)*112
                values[row, x] = pairs(weights[group*128:(group+1)*128,
                                              column:column+112].T)
        # The final row stops atx87. Its other PEs have only placeholder storage.
        yield Transfer('weights', 0, 160+y, width, height, 7168, values)
    yield Transfer('norms', *MODEL_CONTROLLER, 1, 1, 448,
                   pairs(norm).reshape(1, 1, 448))


def model_transfers(tensors, frequency, rows_per_chunk=1):
    """One-time initialization only; no host activation or token computation."""
    for layer in range(24):
        yield from layer_transfers(tensors, frequency, layer)
    yield from vocabulary_transfers(tensors['model.embed_tokens.weight'],
                                    tensors['model.norm.weight'], rows_per_chunk)


def request_transfers(prompt, generation_limit):
    """Load prompt IDs and limits before a globally drained model reset."""
    if (not isinstance(prompt, np.ndarray) or prompt.dtype != np.uint32
            or prompt.ndim != 1 or not 1 <= prompt.size <= 2048
            or np.any(prompt >= 151936)):
        raise ValueError('Prompt must contain1..2048 valid raw-u32 token IDs')
    if (not isinstance(generation_limit, int) or isinstance(generation_limit, bool)
            or not 0 <= generation_limit <= 256
            or prompt.size+generation_limit > 2048):
        raise ValueError('Generation limit must fit the fixed2048/256 capacity')
    yield Transfer('prompt', *MODEL_CONTROLLER, 1, 1, prompt.size,
                   np.ascontiguousarray(prompt).reshape(1, 1, prompt.size))
    yield Transfer('control', *MODEL_CONTROLLER, 1, 1, 2,
                   np.array([prompt.size, generation_limit], np.uint32).reshape(1, 1, 2))
