"""Orbit grouping under all feasible preparation, setting, outcome relabelings.

Brute-force enumeration is intentionally limited and optional: it is intended
for small scenarios where a *complete* inequivalent-class decomposition can be
verified, never a silently partial symmetry classification.
"""
import itertools
import math
from fractions import Fraction as F

from .core import ComplexityLimitExceeded


def facet_orbits(scenario, result, *, max_relabelings=5000):
    """Return orbit representatives and orbit sizes for a previously computed result.

    Returns list[(Facet, int)]. Performs an exhaustive symmetry search only
    if the total relabeling search space fits the supplied budget.
    """
    n=scenario.preparations
    outs=scenario.outcomes
    ny=len(outs)
    # Maximum candidate count counts permutations preserving outcome cardinality.
    groups={o:outs.count(o) for o in set(outs)}
    settings=math.prod(math.factorial(k) for k in groups.values())
    outcome_perm=math.prod(math.factorial(o) for o in outs)
    total=math.factorial(n)*settings*outcome_perm
    if total>max_relabelings:
        raise ComplexityLimitExceeded(f'Complete orbit search requires testing up to '
                                      f'{total:,} relabelings (max_relabelings={max_relabelings:,}).')
    _,_,_,vertices,observables=scenario.vertices()
    loc={(x,y,z):i for i,(x,y,z) in enumerate(observables)}
    lookup={tuple(b):i for i,b in enumerate(vertices)}
    # Identify each facet by the subset of normalized vertices it saturates.
    local_indices=[loc[(int(p.split('|')[1].split(',')[0]),
                        int(p.split(',')[1].split(')')[0]),
                        int(p.split('(')[1].split('|')[0]))]
                   for p in result.coordinates]
    # Parsing independent-coordinate name here is safe: all names are internal.
    active=[]
    for f in result.facets:
        active.append(frozenset(i for i,b in enumerate(vertices)
                                if sum(c*b[j] for c,j in zip(f.coefficients,local_indices))==f.bound))
    by_active={a:i for i,a in enumerate(active)}
    parent=list(range(len(result.facets)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def union(i,j):
        a,b=root(i),root(j)
        if a!=b:parent[b]=a

    setting_perms=[p for p in itertools.permutations(range(ny))
                   if all(outs[p[y]]==outs[y] for y in range(ny))]
    # Precompute probabilities including omitted last outcomes.
    def get(b,x,y,z):
        if z<outs[y]-1:return b[loc[x,y,z]]
        return F(1)-sum(b[loc[x,y,j]] for j in range(outs[y]-1))

    for xp in itertools.permutations(range(n)):
        for yp in setting_perms:
            outcome_generators=[itertools.permutations(range(outs[yp[y]])) for y in range(ny)]
            for zp in itertools.product(*outcome_generators):
                transformed=[]
                for b in vertices:
                    v=tuple(get(b,xp[x],yp[y],zp[y][z]) for x,y,z in observables)
                    j=lookup.get(v)
                    if j is None:break
                    transformed.append(j)
                if len(transformed)!=len(vertices):continue
                for i,a in enumerate(active):
                    j=by_active.get(frozenset(transformed[k] for k in a))
                    if j is not None:union(i,j)
    classes={}
    for i in range(len(result.facets)):
        classes.setdefault(root(i),[]).append(i)
    return [(result.facets[min(inds)],len(inds))
            for inds in sorted(classes.values(),key=lambda inds:(-len(inds),min(inds)))]
