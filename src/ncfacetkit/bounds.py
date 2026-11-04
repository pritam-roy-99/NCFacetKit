"""Exact noncontextual bounds on arbitrary linear probability scores.

Uses a *small* normalization-weight polytope, avoiding enumeration of every
behaviour vertex and every facet.  The state-independent response optimum
is attained at a response-polytope vertex for each epistemic vertex.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction as F
from typing import Optional

from .core import Linear, _parse_linear, ComplexityLimitExceeded


@dataclass(frozen=True)
class BoundResult:
    value: F
    score: str
    preparation_weights: tuple[F,...]
    measurement_vertex_choices: tuple[int|None,...]
    probabilities: dict[str,F]
    witness: str
    certification: str = 'exact over exhaustively enumerated rational vertices'

    def __str__(self):
        return f'{self.score} <= {self.value} (exact noncontextual bound)'

    def to_dict(self):
        return {'score':self.score,'bound':str(self.value),'bound_float':float(self.value),
                'preparation_weights':[str(x) for x in self.preparation_weights],
                'measurement_vertex_choices':list(self.measurement_vertex_choices),
                'witness_probabilities':{k:str(v) for k,v in self.probabilities.items()},
                'certification':self.certification}


def parse_score(statement:str, scenario) -> Linear:
    """Parse a linear score, e.g. 'p(0|1,0) - 2*p(1|2,1)'."""
    n=scenario.preparations
    outs=scenario.outcomes
    def sub(m):
        z,x,y=map(int,m.groups())
        if not (0<=x<n and 0<=y<len(outs) and 0<=z<outs[y]):
            raise ValueError(f'Probability p({z}|{x},{y}) is outside the scenario')
        return f'prob_{z}_{x}_{y}'
    expression=re.sub(r'p\(\s*(\d+)\s*\|\s*(\d+)\s*,\s*(\d+)\s*\)',sub,statement)
    result=_parse_linear(expression, allowed='probability')
    return result


def noncontextual_bound(scenario,expression:str,*,max_bases:int=200_000) -> BoundResult:
    """Exact maximum of a score without enumerating the behaviour hull.

    The result includes a concrete rational, normalized noncontextual model.
    Requires exhaustive prep/response/weight vertex enumeration; if a budget
    is exceeded this *raises*, never invents a bound.
    """
    q,xi,t=scenario._component_vertices(max_bases)
    score=parse_score(expression,scenario)
    terms=[]
    for label,c in score.coefficients.items():
        _,z,x,y=label.split('_')
        terms.append((int(x),int(y),int(z),c))
    effect_index={(z,y):i for i,(z,y) in enumerate(scenario._measurement_matrix()[0])}
    contrib=[]
    choices=[]
    for u in q:
        scores=[]
        for response in xi:
            scores.append(sum(c*u[x]*response[effect_index[z,y]] for x,y,z,c in terms))
        best=max(scores)
        contrib.append(best)
        choices.append(next(j for j,s in enumerate(scores) if s==best))
    scores=[sum(wi*gi for wi,gi in zip(w,contrib)) for w in t]
    optimum=max(scores)
    position=scores.index(optimum)
    weights=t[position]
    probabilities={}
    for y,k in enumerate(scenario.outcomes):
        for z in range(k):
            index=effect_index[z,y]
            for x in range(scenario.preparations):
                val=sum(weights[i]*u[x]*xi[choices[i]][index] for i,u in enumerate(q) if weights[i])
                probabilities[f'p({z}|{x},{y})']=val
    selected=tuple(choices[i] if weights[i] else None for i in range(len(q)))
    status=('exact rational optimum over exhaustively enumerated component vertices'
            if scenario._cert_cache[max_bases] else
            'rational optimum over enumerated vertices; global completeness NOT certified')
    return BoundResult(optimum+score.constant,expression,weights,selected,probabilities,
                       'convex mixture of q-vertices and response-vertices',status)
