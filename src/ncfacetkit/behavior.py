"""Numerical membership and generalized-noncontextual fraction LPs.

These tools accept an observed *normalized* operational probability table.
They are distinct from exhaustive facet enumeration and may still be useful
when the latter is prohibitively expensive.  Optimization is numerical (HiGHS).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class MembershipResult:
    noncontextual: bool
    feasible: bool
    status: str
    maximum_residual: float|None
    decomposition: tuple[tuple[int,int,float], ...]


@dataclass(frozen=True)
class FractionResult:
    noncontextual_fraction: float
    contextual_fraction: float
    status: str
    decomposition: tuple[tuple[int,int,float], ...]
    probability_residual: float


def _probability_table(scenario,data, tol=1e-8):
    n=scenario.preparations
    ks=scenario.outcomes
    if isinstance(data,Mapping):
        if not all(f'p({z}|{x},{y})' in data
                   for x in range(n) for y,k in enumerate(ks) for z in range(k)):
            raise ValueError('Supply all probabilities p(z|x,y), including final outcomes.')
        table=[[[float(data[f'p({z}|{x},{y})']) for z in range(k)]
                for y,k in enumerate(ks)] for x in range(n)]
    else:
        table=[[[float(v) for v in p] for p in x] for x in data]
    if len(table)!=n or any(len(t)!=len(ks) for t in table):
        raise ValueError('Probability table must be indexed [preparation][measurement][outcome].')
    if any(len(table[x][y])!=ks[y] for x in range(n) for y in range(len(ks))):
        raise ValueError('Incorrect outcome count in probability table.')
    if any(not np.isfinite(v) or v < -tol for row in table for p in row for v in p):
        raise ValueError('Probabilities must be finite and nonnegative.')
    for x in range(n):
        for y in range(len(ks)):
            if abs(sum(table[x][y])-1)>tol:
                raise ValueError(f'Probabilities for preparation {x}, setting {y} do not sum to 1.')
    for lhs,rhs,_ in scenario._peqs:
        diff=lhs-rhs
        for y,k in enumerate(ks):
            for z in range(k):
                val=sum(float(v)*table[int(sym[1:])][y][z]
                        for sym,v in diff.coefficients.items())
                if abs(val)>tol:
                    raise ValueError('Data violate a preparation operational equivalence.')
    for lhs,rhs,_ in scenario._meqs:
        diff=lhs-rhs
        for x in range(n):
            val=float(diff.constant)
            for sym,c in diff.coefficients.items():
                z,y=map(int,sym[1:].split('_'))
                val+=float(c)*table[x][y][z]
            if abs(val)>tol:
                raise ValueError('Data violate a measurement operational equivalence.')
    flat=np.array([table[x][y][z]
                  for x in range(n) for y,k in enumerate(ks) for z in range(k)])
    return flat


def _model(scenario,max_bases=200_000,max_variables=50_000):
    prep,meas,_=scenario._component_vertices(max_bases)
    n=scenario.preparations
    ks=scenario.outcomes
    indices=[(x,y,z) for x in range(n) for y,k in enumerate(ks) for z in range(k)]
    effects=scenario._measurement_matrix()[0]
    ei={ef:i for i,ef in enumerate(effects)}
    count=len(prep)*len(meas)
    if count>max_variables:
        from .core import ComplexityLimitExceeded
        raise ComplexityLimitExceeded(f'{count:,} LP columns > max_variables={max_variables:,}')
    pmat=np.array([[float(q[x]*m[ei[z,y]]) for q in prep for m in meas]
                   for x,y,z in indices])
    qmat=np.array([[float(q[x]) for q in prep for _ in meas] for x in range(n)])
    return pmat,qmat,prep,meas


def is_noncontextual(scenario,data,*,tol=1e-8,max_bases=200_000):
    target=_probability_table(scenario,data,tol)
    V,Q,prep,meas=_model(scenario,max_bases)
    eq=np.vstack([V,Q,np.ones((1,V.shape[1]))])
    rhs=np.r_[target, np.ones(scenario.preparations+1)]
    res=linprog(np.zeros(V.shape[1]),A_eq=eq,b_eq=rhs,bounds=(0,None),method='highs')
    if not res.success:
        return MembershipResult(False,False,res.message,None,())
    residual=float(np.max(np.abs(eq@res.x-rhs)))
    weights=tuple((i,j,float(res.x[i*len(meas)+j]))
                  for i in range(len(prep)) for j in range(len(meas))
                  if res.x[i*len(meas)+j]>tol)
    return MembershipResult(residual<=max(tol,1e-7),True,res.message,residual,weights)


def contextual_fraction(scenario,data,*,tol=1e-8,max_bases=200_000):
    target=_probability_table(scenario,data,tol)
    V,Q,prep,meas=_model(scenario,max_bases)
    num=V.shape[1]
    # lambda is the sum of subnormalized NC ontic weights.  The same lambda
    # must normalize every preparation so that the remainder is operational.
    eq=Q-np.ones((scenario.preparations,num))
    res=linprog(-np.ones(num),A_ub=np.vstack([V,np.ones((1,num))]),
                b_ub=np.r_[target,1],A_eq=eq,b_eq=np.zeros(scenario.preparations),
                bounds=(0,None),method='highs')
    if not res.success:
        raise RuntimeError('Contextual-fraction linear program failed: '+res.message)
    fraction=float(np.clip(sum(res.x),0,1))
    residual=float(max(np.max(V@res.x-target),np.max(np.abs(eq@res.x)),0))
    weights=tuple((i,j,float(res.x[i*len(meas)+j]))
                  for i in range(len(prep)) for j in range(len(meas))
                  if res.x[i*len(meas)+j]>tol)
    return FractionResult(fraction,1-fraction,res.message,weights,residual)
