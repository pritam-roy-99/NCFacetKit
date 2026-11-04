"""Exact noncontextual bounds on arbitrary linear probability scores.

Uses a *small* normalization-weight polytope, avoiding enumeration of every
behaviour vertex and every facet.  The state-independent response optimum
is attained at a response-polytope vertex for each epistemic vertex.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np

from .core import Linear, _parse_linear, _integer_matrix, ComplexityLimitExceeded


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
    # Build the score tensor in exact common-denominator integer arithmetic.
    # For modest coefficients, NumPy int64 matrix multiplication provides the
    # fast path; for large numerators the object backend avoids overflow.
    p,dp=_integer_matrix(q)
    m,dm=_integer_matrix(xi)
    w,dw=_integer_matrix(t)
    dc=__import__('math').lcm(*(c.denominator for _,_,_,c in terms)) if terms else 1
    score_matrix=np.zeros((scenario.preparations,len(effect_index)),dtype=object)
    for x,y,z,c in terms:
        score_matrix[x,effect_index[z,y]]+=int(c*dc)
    pa=max((abs(int(v)) for v in p.flat),default=0)
    ma=max((abs(int(v)) for v in m.flat),default=0)
    wa=max((abs(int(v)) for v in w.flat),default=0)
    ca=max((abs(int(v)) for v in score_matrix.flat),default=0)
    if pa*ma*ca*wa*max(1,scenario.preparations)*max(1,len(effect_index))*max(1,len(q))<(1<<62):
        score_matrix=np.asarray(score_matrix,dtype=np.int64)
    else:
        p=np.asarray(p,dtype=object)
        m=np.asarray(m,dtype=object)
        w=np.asarray(w,dtype=object)
    pair_scores=(p @ score_matrix) @ m.T
    choices=[int(j) for j in np.argmax(pair_scores,axis=1)]
    contrib=[int(pair_scores[i,j]) for i,j in enumerate(choices)]
    totals=w @ np.asarray(contrib,dtype=w.dtype)
    position=int(np.argmax(totals))
    optimum=F(int(totals[position]),dp*dm*dc*dw)
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
            if scenario._cert_cache[max_bases,scenario.vertex_backend] else
            'rational optimum over enumerated vertices; global completeness NOT certified')
    return BoundResult(optimum+score.constant,expression,weights,selected,probabilities,
                       'convex mixture of q-vertices and response-vertices',status)
