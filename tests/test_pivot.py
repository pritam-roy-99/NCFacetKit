"""Exact pivoting engine: graph completeness, degeneracy and budget safety."""
import itertools
import math
import random
from fractions import Fraction as F

import pytest

from ncfacetkit import ComplexityLimitExceeded, Scenario
from ncfacetkit.core import (_enumerate_standard_vertices, _fraction_solve,
                             _independent_rows)


def _enumerate_reference(rows, rhs):
    """Small-only independent oracle: exhaustively scan all n-choose-r bases."""
    rows, rhs = _independent_rows(rows, rhs)
    n, r = len(rows[0]), len(rows)
    found = set()
    for basis in itertools.combinations(range(n), r):
        matrix = [[row[j] for j in basis] for row in rows]
        values = _fraction_solve(matrix, rhs)
        if values is None or any(v < 0 for v in values):
            continue
        vertex = [F(0)]*n
        for j, value in zip(basis, values):
            vertex[j] = value
        found.add(tuple(vertex))
    return found


def test_pivot_matches_exhaustive_basis_reference_randomized():
    rng = random.Random(20261105)
    for n in range(3, 10):
        for trial in range(8):
            r = rng.randint(1, min(4, n-1))
            rows = [[F(1,n)]*n]
            for _ in range(r-1):
                rows.append([F(rng.randint(-4,4)) for _ in range(n)])
            x0 = [F(rng.randint(0,5),7) for _ in range(n)]
            rhs = [sum(a*v for a,v in zip(row,x0)) for row in rows]
            expected = _enumerate_reference(rows,rhs)
            result = set(_enumerate_standard_vertices(rows,rhs,5000,
                         label='randomized',backend='pivot'))
            assert result == expected, (n,trial)


def test_degenerate_zero_length_basis_pivots_are_not_lost():
    rows = [[F(1,6)]*6,
            [F(1),F(-1),F(0),F(0),F(0),F(0)]]
    rhs = [F(1),F(0)]
    result = set(_enumerate_standard_vertices(rows,rhs,100,
                 label='degenerate',backend='pivot'))
    assert result == _enumerate_reference(rows,rhs)
    assert len(result) == 5


def test_many_infeasible_bases_no_longer_block_vertex_enumeration():
    scenario=Scenario(24,1,2,vertex_backend='pivot')
    for j in range(5):
        scenario.prepare(f'P{2*j}=P{2*j+1}')
    assert math.comb(24,6) == 134596
    prep,_,_=scenario._component_vertices(max_bases=1000)
    assert len(prep)==19
    assert scenario.analyze(max_bases=1000)['component_vertex_enumeration_certified']
    with pytest.raises(ComplexityLimitExceeded,match='feasible-basis'):
        scenario._component_vertices(max_bases=10)


def test_explicit_backend_and_json_roundtrip():
    s=Scenario(4,2,2,vertex_backend='pivot').prepare('(P0+P1)/2=(P2+P3)/2')
    assert Scenario.from_dict(s.to_dict()).vertex_backend=='pivot'
    assert s.analyze()['vertex_backend']=='pivot'
    assert len(s.facets().facets)==24
    with pytest.raises(ValueError,match='vertex_backend'):
        Scenario(4,2,2,vertex_backend='not-a-backend')


def test_gmp_cdd_matches_pivot_if_installed():
    pytest.importorskip('cdd.gmp')
    p=Scenario(4,2,2,vertex_backend='pivot').prepare('(P0+P1)/2=(P2+P3)/2')
    c=Scenario(4,2,2,vertex_backend='cdd').prepare('(P0+P1)/2=(P2+P3)/2')
    pp,pm,pw=p._component_vertices()
    cp,cm,cw=c._component_vertices()
    assert set(pp)==set(cp)
    assert set(pm)==set(cm)
    assert set(pw)==set(cw)
    pf=p.facets(backend='qhull')
    cf=c.facets(backend='cdd')
    assert {(f.coefficients,f.bound) for f in pf.facets}=={
        (f.coefficients,f.bound) for f in cf.facets}
    assert 'exact' in cf.completeness
