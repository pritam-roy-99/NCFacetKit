"""Reproducible NCFacetKit microbenchmark suite.

Run from the repository root:
    python benchmarks/run.py
    python benchmarks/run.py --full   # may be very expensive

Prints timings; no assertive claims about universally achievable speedups.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from ncfacetkit import Scenario


def cases():
    return {
        'hazra4': Scenario(4, 2, 2).prepare('(P0+P1)/2=(P2+P3)/2'),
        'linked4': Scenario(4, 2, 2).prepare('(P0+P1)/2=(P2+P3)/2').measure('M[0|0]+M[0|1]=I'),
        'hazra5': Scenario(5, 2, 2).prepare('(P0+P1)/2=(P2+P3+P4)/3'),
        'six': Scenario(6, 3, 2).prepare('(P0+P1)/2=(P2+P3)/2').prepare('(P0+P1)/2=(P4+P5)/2'),
    }


def pivot_case():
    """Large n-choose-r, small feasible-basis graph (old scanner failed)."""
    s=Scenario(24,1,2,vertex_backend='pivot')
    for j in range(5):
        s.prepare(f'P{2*j}=P{2*j+1}')
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pivot',action='store_true',help='benchmark large sparse feasible-basis graph')
    parser.add_argument('--full',action='store_true',help='attempt potentially slow higher-dimensional hulls')
    parser.add_argument('--repeats',type=int,default=200,help='number of bound evaluations')
    parser.add_argument('--output',type=Path,help='optional JSON results file')
    args=parser.parse_args()
    collected={}
    for name,s in cases().items():
        start=perf_counter()
        verts=s.vertices()
        elapsed_vertices=perf_counter()-start
        row={'behavior_vertices':len(verts[3]),'vertices_seconds':elapsed_vertices}
        if name in ('hazra4','linked4') or args.full:
            start=perf_counter()
            facets=s.facets()
            row['facets_seconds']=perf_counter()-start
            row['facets_count']=len(facets.facets)
        start=perf_counter()
        for _ in range(args.repeats):
            s.noncontextual_bound('p(0|0,0)+2*p(0|1,1)-p(1|2,0)')
        row['bounds_seconds']=perf_counter()-start
        row['bounds_repeats']=args.repeats
        print(name, json.dumps(row),flush=True)
        collected[name]=row
    if args.pivot:
        sparse=pivot_case()
        start=perf_counter()
        prep=sparse._component_vertices(max_bases=1000)[0]
        collected['sparse_24x6']={'prep_vertices':len(prep),
            'candidate_bases':134596,
            'pivot_seconds':perf_counter()-start}
        print('sparse_24x6',json.dumps(collected['sparse_24x6']),flush=True)
    if args.output:
        args.output.write_text(json.dumps(collected,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':main()
