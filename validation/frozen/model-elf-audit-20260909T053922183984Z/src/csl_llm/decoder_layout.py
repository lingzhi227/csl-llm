"""Layer-indexed resident ownership; transport integration is a separate step."""
from dataclasses import dataclass
from .regions import Region


def geometry(origin=(0, 0)):
    bx, by = origin
    if (not isinstance(bx, int) or isinstance(bx, bool) or
            not isinstance(by, int) or isinstance(by, bool) or bx < 0 or by < 0):
        raise ValueError('Layer origin must contain nonnegative integer coordinates')
    return {
        'up': [Region(bx+8*(g % 2), by+g//2, 8, 1) for g in range(38)],
        'gate': [Region(bx+16+8*(g % 2), by+g//2, 8, 1) for g in range(38)],
        'down': [Region(bx+32, by+2*g, 22, 2) for g in range(7)],
        'q': [Region(bx+54, by+g, 8, 1) for g in range(7)],
        'k': [Region(bx+54, by+7, 8, 1)],
        'v': [Region(bx+54, by+8, 8, 1)],
        'o': [Region(bx+54, by+9+g, 8, 1) for g in range(7)],
        'kv': [Region(bx+32, by+14, 11, 6), Region(bx+43, by+14, 11, 6)],
    }


def model_origin(layer):
    if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < 24:
        raise ValueError('Pinned Qwen layer index must be in [0,24)')
    return (layer % 3)*64, (layer//3)*20


@dataclass(frozen=True)
class LayerTile:
    layer: int
    operation: str
    x: int
    y: int
    group: int
    ordinal: int
    participants: int
    negative: str
    positive: str
    destination: tuple[int, int]
    tensor: str = ''
    row: int = 0
    column: int = 0
    valid_columns: int = 0
    bias_tensor: str = ''
    stripe: int = -1

    @property
    def packet_group(self):
        return 7 if self.operation == 'k' else 8 if self.operation == 'v' else self.group


def tiles(layer, origin=None):
    default_origin = model_origin(layer)
    origin = default_origin if origin is None else origin
    regions = geometry(origin)
    controller = (origin[0]+62, origin[1])
    records, occupied = [], set()
    for operation, groups in regions.items():
        for group, region in enumerate(groups):
            for ordinal in range(region.participants):
                x, y = region.point(ordinal)
                if (x, y) in occupied or (x, y) == controller:
                    raise ValueError('Overlapping decoder ownership')
                occupied.add((x, y))
                negative, positive = region.neighbors(ordinal)
                destination = regions['gate'][group].point(0) if operation == 'up' else controller
                if operation == 'kv':
                    # Stripe identity is physical row order, not the collective
                    # serpentine ordinal. Padding stripes remain participants.
                    record = LayerTile(layer, operation, x, y, group, ordinal, region.participants,
                                       negative, positive, destination,
                                       stripe=(y-region.y)*11+x-region.x)
                else:
                    category = 'mlp' if operation in ('up', 'gate', 'down') else 'self_attn'
                    prefix = f'model.layers.{layer}.{category}.{operation}_proj'
                    shard = ordinal if operation != 'down' or ordinal < 22 else 65-ordinal
                    inner = 4864 if operation == 'down' else 896
                    record = LayerTile(layer, operation, x, y, group, ordinal, region.participants,
                                       negative, positive, destination, tensor=prefix+'.weight',
                                       row=group*128, column=shard*112, valid_columns=min(112, inner-shard*112),
                                       bias_tensor=prefix+'.bias' if operation in ('q', 'k', 'v') and ordinal == 0 else '')
                records.append(record)
    assert len(records) == 1176
    return records


def controller_parameters(layer, origin=None):
    default_origin = model_origin(layer)
    origin = default_origin if origin is None else origin
    geometry(origin)  # Validate alternate origin with the same contract.
    return dict(layer=layer, coordinate=(origin[0]+62, origin[1]),
                norms=[f'model.layers.{layer}.{name}.weight'
                       for name in ('input_layernorm', 'post_attention_layernorm')],
                incoming_values=896, outgoing_values=896,
                incoming_requires=['matching invocation', 'all896 values', 'compute command'],
                outgoing_requires=['final residual', 'buffer held through send completion'],
                scope='Ownership/interface declaration only; no generated24layer program or runtime qualification')
