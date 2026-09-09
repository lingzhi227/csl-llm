"""Static argmax paths using this model's application color palette0..7.

The palette is a project allocation policy, not the hardware's color count.
"""


def forwarding_routes(participants, stride, origin=(0, 0), colors=(5, 6)):
    if any(type(value) is not int or value < 2 for value in [participants, stride]):
        raise ValueError('A strided reduction needs at least two participants and stride2')
    if (len(origin) != 2 or any(type(value) is not int or value < 0 for value in origin)
            or len(colors) != 2 or len(set(colors)) != 2
            or any(type(value) is not int or not 0 <= value < 8 for value in colors)):
        raise ValueError('Invalid origin or distinct application colors')
    bx, by = origin
    return {(bx+ordinal*stride+offset, by, color): ('EAST', 'WEST')
            for ordinal in range(participants-1) for offset in range(1, stride)
            for color in colors}


def emit(participants, stride, origin=(0, 0), colors=(5, 6)):
    return '\n'.join(
        '@set_color_config(%d,%d,@get_color(%d),.{.routes=.{.rx=.{%s},.tx=.{%s}}});' % (x,y,color,rx,tx)
        for (x, y, color), (rx, tx) in forwarding_routes(participants, stride, origin, colors).items())
