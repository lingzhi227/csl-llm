"""Lossless SDK transport of two raw BF16 storage words per u32 wavelet."""
import numpy as np


def pack_bf16_pairs(raw):
    raw = np.asarray(raw)
    if raw.dtype.kind != 'u' or raw.dtype.itemsize != 2 or raw.size % 2:
        raise ValueError('BF16 transport requires an even number of unsigned 16-bit words')
    words = np.ascontiguousarray(raw, dtype='<u2').reshape(-1)
    pairs = words.view('<u4')
    # Explicit value/byte checks: do not interpret or numerically convert BF16.
    expected = words[0::2].astype(np.uint32) | (words[1::2].astype(np.uint32) << 16)
    assert np.array_equal(pairs, expected)
    assert pairs.tobytes() == words.tobytes()
    assert np.array_equal(pairs.view('<u2'), words)
    return pairs
