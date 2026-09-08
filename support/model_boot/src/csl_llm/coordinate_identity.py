"""Specification for opt-in model metadata derived from application coordinates.

This does not execute CSL or replace numerical work. The device implementation
must be checked against these values after actual SDK initialization.
"""


def decoder_identity(operation, x, y):
    if type(x) is not int or type(y) is not int or not (0 <= x < 192 and 0 <= y < 160):
        raise ValueError('Coordinate must be inside the fixed24-layer model region')
    lx, ly = x % 64, y % 20
    bx, by = x-lx, y-ly
    bounds = {
        'up': (0, 15, 0, 18), 'gate': (16, 31, 0, 18),
        'down': (32, 53, 0, 13), 'q': (54, 61, 0, 6),
        'k': (54, 61, 7, 7), 'v': (54, 61, 8, 8),
        'o': (54, 61, 9, 15), 'kv': (32, 53, 14, 19),
    }
    if operation not in bounds:
        raise ValueError('Unknown matrix/KV operation')
    left, right, top, bottom = bounds[operation]
    if not (left <= lx <= right and top <= ly <= bottom):
        raise ValueError('Operation does not own this application coordinate')
    if operation in ['up', 'gate']:
        group = ly*2+(lx-(16 if operation == 'gate' else 0))//8
    elif operation == 'down':
        group = ly//2
    elif operation in ['q', 'k', 'v']:
        group = ly
    elif operation == 'o':
        group = ly-9
    else:
        group = (lx-32)//11
    destination = (bx+16+8*(group % 2), by+group//2) if operation == 'up' else (bx+62, by)
    return (group, *destination)


def vocabulary_identity(x, y):
    if (type(x) is not int or type(y) is not int or
            not (0 <= x < 192 and 160 <= y < 210) or (y == 209 and x >= 88)):
        raise ValueError('Coordinate must own an original vocabulary matrix tile')
    return (y-160)*24+x//8
