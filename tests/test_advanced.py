"""Full-stack regression tests for the advanced finite scenario engine."""
from fractions import Fraction
from math import sqrt

import pytest
from ncfacetkit import Scenario, ComplexityLimitExceeded


def test_exact_score_bound_without_facets():
    s=Scenario(4,2,2).prepare('(P0 + P1)/2 = (P2 + P3)/2')
    b=s.noncontextual_bound('p(0|1,0)-p(0|2,0)-p(0|0,1)+p(0|2,1)')
    assert b.value==1
    assert b.certification.startswith('exact')
    assert len(b.probabilities)==16
    assert b.to_dict()['bound']=='1'


def test_score_exact_fraction_and_constant():
    s=Scenario(2,1,2)
    assert s.noncontextual_bound('p(1|0,0)/3 + p(0|1,0)/5 + 1/7').value == Fraction(71,105)
    with pytest.raises(ValueError):s.noncontextual_bound('p(3|0,0)')
    with pytest.raises(ValueError):s.noncontextual_bound('p(0|0,0)*p(1|1,0)')


def test_100_preparations_fast_path_without_large_behavior_hull():
    s=Scenario(100,outcomes=[4,4,2])
    result=s.facets(max_products=2)
    assert result.affine_dimension==700
    assert len(result.facets)==1000
    assert result.backend=='exact-product-factorization'
    assert result.completeness.startswith('exact')


def test_named_binary_linked_effects():
    s=Scenario(4,2,2).prepare('(P0+P1)/2=(P2+P3)/2')
    s.effect_relation({(0,0):1,(0,1):1},rhs=1)
    r=s.facets()
    assert len(r.facets)==8
    assert r.affine_dimension==3


def test_structured_preparation_relation_roundtrip():
    s=Scenario(4,2,2).preparation_relation([1,1,0,0],[0,0,1,1])
    t=Scenario.from_dict(s.to_dict())
    assert len(s.facets().facets)==len(t.facets().facets)==24


def test_asymmetric_4_4_2_effects_and_six_preparations():
    s=(Scenario(6,outcomes=[4,4,2])
       .prepare('(P0+P1)/2=(P2+P3)/2')
       .prepare('(P0+P1)/2=(P4+P5)/2')
       .measure('E[0|2]=E[0|0]+E[1|0]')
       .measure('M[0|2]=M[2|1]+M[3|1]'))
    summary=s.analyze()
    assert (summary['preparation_vertices'],summary['measurement_vertices'],
            summary['normalization_weight_vertices'])==(8,8,6)
    bound=s.noncontextual_bound('p(0|0,0)+p(0|1,1)-p(1|2,0)')
    assert bound.value==2
    assert Scenario.from_dict(s.to_dict()).analyze()==summary


def test_fixed_measurement_response():
    s=Scenario(5,outcomes=[2]).measure('M[0|0]=I/2')
    r=s.facets()
    assert r.affine_dimension==0
    assert len(r.facets)==0
    assert r.behavior_vertices==1


def test_product_measurement_equivalence_intersection():
    s=Scenario(5,outcomes=[2,2]).measure('M[0|0]+M[0|1]=I')
    r=s.facets()
    assert r.affine_dimension==5
    assert len(r.facets)==10
    assert r.behavior_vertices==2**5


def test_empirical_membership_and_contextual_fraction():
    import numpy as np
    s=Scenario(4,2,2).prepare('(P0+P1)/2=(P2+P3)/2')
    Q=[np.array(v) for v in [(1,0,0),(-1,0,0),(0,0,1),(0,0,-1)]]
    M=[np.array(v)/sqrt(2) for v in [(-1,0,-1),(-1,0,1)]]
    p=[[[float((1+r@m)/2),float((1-r@m)/2)] for m in M] for r in Q]
    assert not s.is_noncontextual(p).noncontextual
    f=s.contextual_fraction(p)
    assert abs(f.contextual_fraction-(sqrt(2)-1))<1e-6
    # The barycenter of quantum preparations is noncontextual.
    nc=[[[.5,.5] for y in range(2)] for x in range(4)]
    assert s.is_noncontextual(nc).noncontextual
    assert abs(s.contextual_fraction(nc).noncontextual_fraction-1)<1e-8


def test_invalid_data_and_degenerate_constraints():
    s=Scenario(2,1,2)
    with pytest.raises(ValueError):s.is_noncontextual([[[.5,.5]]])
    with pytest.raises(ValueError):s.contextual_fraction([[[1,1]],[[0,1]]])
    s.measure('M[0|0] = 2*I')
    with pytest.raises(ValueError):s.facets()


def test_guard_on_generalized_case():
    s=(Scenario(6,outcomes=[4,4,2])
       .prepare('(P0+P1)/2=(P2+P3)/2')
       .prepare('(P0+P1)/2=(P4+P5)/2'))
    with pytest.raises(ComplexityLimitExceeded):s.facets(max_products=2)
