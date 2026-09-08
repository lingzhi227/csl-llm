"""Geometry for disjoint region-relative serpentine collectives."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int

    @property
    def participants(self):
        return self.width * self.height

    def point(self, ordinal):
        if not 0 <= ordinal < self.participants:
            raise ValueError('Ordinal outside region')
        row, col = divmod(ordinal, self.width)
        return self.x + (col if row % 2 == 0 else self.width - 1 - col), self.y + row

    def neighbors(self, ordinal):
        def direction(target):
            x, y = self.point(ordinal)
            a, b = self.point(target)
            return {(1, 0): 'EAST', (-1, 0): 'WEST', (0, 1): 'SOUTH', (0, -1): 'NORTH'}[(a-x, b-y)]
        return (direction(ordinal-1) if ordinal else 'WEST',
                direction(ordinal+1) if ordinal+1 < self.participants else 'EAST')


def layout(width, height, regions, exports):
    occupied = {}
    for group, region in enumerate(regions):
        if region.width < 1 or region.height < 1 or region.participants < 2:
            raise ValueError('Region must contain at least two PEs')
        for ordinal in range(region.participants):
            point = region.point(ordinal)
            if point in occupied or not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError('Overlapping or out-of-bounds region')
            occupied[point] = (group, ordinal, *region.neighbors(ordinal), region.participants)
    lines = [f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height={height}}});layout{{@set_rectangle({width},{height});']
    for y in range(height):
        for x in range(width):
            group, ordinal, negative, positive, participants = occupied.get((x,y), (-1,0,'WEST','EAST',2))
            lines.append(f'@set_tile_code({x},{y},"pe.csl",.{{.memcpy_params=memcpy.get_params({x}),.group={group},.ordinal={ordinal},.participants={participants},.negative={negative},.positive={positive}}});')
    lines.extend(f'@export_name("{name}",[*]{kind},false);' for name, kind in exports)
    lines.extend(['@export_name("reset",fn()void);','@export_name("compute",fn()void);','}'])
    return '\n'.join(lines)+'\n'
