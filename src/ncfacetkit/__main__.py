import argparse
import json
from pathlib import Path
from . import Scenario


def main():
    parser=argparse.ArgumentParser(
        description='Generalized-noncontextuality facets and exact NC score bounds')
    parser.add_argument('scenario',help='JSON scenario definition')
    parser.add_argument('--output','-o',help='Save facets as .json, .csv, or .tex')
    parser.add_argument('--bound',help='Maximize a linear score such as p(0|1,0)-p(1|2,1)')
    parser.add_argument('--analyze',action='store_true',help='Show polytope complexity first')
    parser.add_argument('--backend',choices=['auto','qhull','cdd'],default='auto')
    parser.add_argument('--max-bases',type=int,default=200_000)
    parser.add_argument('--max-products',type=int,default=400_000)
    parser.add_argument('--max-hull-vertices',type=int,default=10_000)
    parser.add_argument('--show',type=int,default=30)
    parser.add_argument('--orbits',action='store_true',
                        help='Group facets under exhaustively tested relabelings')
    opts=parser.parse_args()
    config=json.loads(Path(opts.scenario).read_text(encoding='utf8'))
    scenario=Scenario.from_dict(config)
    if opts.analyze:
        print(json.dumps(scenario.analyze(max_bases=opts.max_bases),indent=2))
    if opts.bound is not None:
        bound=scenario.noncontextual_bound(opts.bound,max_bases=opts.max_bases)
        print(bound)
        if opts.output:
            if Path(opts.output).suffix!='.json':
                parser.error('--bound output requires a .json filename')
            Path(opts.output).write_text(json.dumps(bound.to_dict(),indent=2)+'\n')
        return
    result=scenario.facets(max_bases=opts.max_bases,
        max_products=opts.max_products,max_hull_vertices=opts.max_hull_vertices,
        backend=opts.backend)
    result.show(opts.show)
    print('Backend:',result.backend)
    print('Completeness:',result.completeness)
    if opts.orbits:
        classes=scenario.orbits(result)
        print(f'\n{len(classes)} inequivalent classes:')
        for f,n in classes:print(f'  orbit {n:>4}: {f}')
    if opts.output:
        result.save(opts.output)
        print('Saved',opts.output)


if __name__=='__main__':main()
