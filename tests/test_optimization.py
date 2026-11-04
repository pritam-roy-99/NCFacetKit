"""Regression tests for exact arithmetic, integer vectorization and pivoting."""
from fractions import Fraction as F
from itertools import product

import numpy as np
from ncfacetkit import Scenario
from ncfacetkit.core import _rank, _rref, _independent_rows


def test_fraction_row_reduction_and_degenerate_rows():
    assert _rank([[1, 0, 3], [2, 0, 6], [0, 1, 4]]) == 2
    rows, rhs = _independent_rows([[1, 2, 0], [2, 4, 0], [0, 0, 1]], [F(3), F(6), F(5)])
    assert rows == [[1, 2, 0], [0, 0, 1]]
    assert rhs == [F(3), F(5)]
    from pytest import raises
    with raises(ValueError, match='Inconsistent'):
        _independent_rows([[1, 2], [2, 4]], [F(3), F(5)])
    assert _rref([[1, F(1, 2), 0], [0, 1, 1]])[1] == [0, 1]


def test_integer_vertex_multiplication_agrees_with_fraction_reference():
    s = Scenario(4, 2, 2).prepare('(P0+P1)/2=(P2+P3)/2')
    q, xi, weights, behaviors, observable = s.vertices()
    effects = s._measurement_matrix()[0]
    lookup = {ef:i for i,ef in enumerate(effects)}
    direct = set()
    for weight in weights:
        active = [i for i,wi in enumerate(weight) if wi>0]
        for assignment in product(range(len(xi)), repeat=len(active)):
            direct.add(tuple(sum(weight[i]*q[i][x]*xi[j][lookup[z,y]]
                                 for i,j in zip(active,assignment))
                             for x,y,z in observable))
    assert set(behaviors) == direct
    assert s.vertices()[3] is behaviors  # cache preserves exact behavior data


def test_cache_invalidated_by_equivalences():
    s = Scenario(4, 2, 2)
    prior = s.vertices()
    s.prepare('(P0+P1)/2=(P2+P3)/2')
    current = s.vertices()
    assert current[3] is not prior[3]
    assert len(current[3]) == 28


def test_overflow_safe_exact_scores_and_large_denominators():
    s = Scenario(2, 1, 2)
    large = 10**22
    bound = s.noncontextual_bound(f'{large}*p(0|0,0)+{large}*p(0|1,0)')
    assert bound.value == 2*large
    tiny = s.noncontextual_bound('p(0|0,0)/999999937+p(1|1,0)/999999929')
    assert tiny.value == F(1,999999937)+F(1,999999929)


def test_mixed_outcome_linked_exact_optimization():
    s=(Scenario(6,outcomes=[4,4,2])
       .prepare('(P0+P1)/2=(P2+P3)/2')
       .prepare('(P0+P1)/2=(P4+P5)/2')
       .measure('M[0|2]=M[0|0]+M[1|0]')
       .measure('M[0|2]=M[2|1]+M[3|1]'))
    assert s.noncontextual_bound('p(0|0,0)+p(0|1,1)-p(1|2,0)').value == 2
    assert len(s.vertices(max_products=300000)[3])>0
