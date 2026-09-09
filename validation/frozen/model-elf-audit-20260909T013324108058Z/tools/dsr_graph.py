"""Validate pinned CSL compiler allocation graphs, not runtime task ordering."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def inspect(text):
    nodes, edges = {}, []
    for line in text.splitlines():
        line = line.strip()
        if line in ('graph G {', '}', ''):
            continue
        edge = re.fullmatch(r'(\d+) -- (\d+);?', line)
        if edge:
            edges.append(tuple(map(int, edge.groups())))
            continue
        node = re.match(r'(\d+)\[', line)
        if not node:
            raise ValueError(f'Unsupported graph syntax: {line[:100]}')
        number = int(node[1])
        if number in nodes:
            raise ValueError(f'Duplicate node {number}')
        assigned = re.search(r'Assigned color = (-?\d+)', line)
        if assigned is None or int(assigned[1]) < 0:
            raise ValueError(f'Unassigned node {number}')
        color = int(assigned[1])
        explicit = re.search(r'Explicit DSR = (\d+)', line)
        bounds = re.search(r'Range = \[(\d+),(\d+)\)', line)
        if explicit:
            if color != int(explicit[1]):
                raise ValueError(f'Explicit DSR mismatch for node {number}')
        elif bounds:
            if not int(bounds[1]) <= color < int(bounds[2]):
                raise ValueError(f'DSR outside range for node {number}')
        else:
            raise ValueError(f'Missing allocation constraint for node {number}')
        nodes[number] = color
    if not nodes:
        raise ValueError('Empty allocation graph')
    for left, right in edges:
        if left not in nodes or right not in nodes:
            raise ValueError(f'Edge has undefined endpoint: {(left, right)}')
        if nodes[left] == nodes[right]:
            raise ValueError(f'Interfering nodes share DSR: {(left, right)}')
    return dict(nodes=len(nodes), edges=len(edges), assigned_indices=sorted(set(nodes.values())))


def report(bundle):
    directory = bundle / 'out/bin'
    groups = {}
    for path in directory.glob('*.dot'):
        key, algorithm, extension = path.name.rsplit('.', 2)
        if algorithm not in ('greedy', 'dsatur') or extension != 'dot':
            raise ValueError(f'Unrecognized allocation artifact: {path.name}')
        groups.setdefault(key, {})[algorithm] = path
    if not groups:
        raise ValueError('No compiler allocation graphs')
    elfs = sorted(directory.glob('*.elf'))
    families = {path.stem.rsplit('_', 1)[0] for path in elfs}
    if families != set(groups):
        raise ValueError(f'ELF/graph family coverage mismatch: {families ^ set(groups)}')
    results = []
    for key, paths in sorted(groups.items()):
        # A greedy failure may be followed by a valid DSATUR allocation.
        path = paths.get('dsatur', paths.get('greedy'))
        row = inspect(path.read_text())
        row.update(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        results.append(row)
    return dict(passed=True, graphs=results, elf_count=len(elfs), family_count=len(families),
                elf_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in elfs},
                manifest_sha256=hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest(),
                tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Validate every final application allocation graph: node assignment, declared bounds, explicit DSR equality and every interference edge. Graph colors are DSR indices, not router colors. Does not prove runtime lifetimes, task ordering, dynamic stack or whole-model fit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    value = report(args.bundle)
    with args.report.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print('DSR FINAL GRAPHS', len(value['graphs']))
