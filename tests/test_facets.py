from fractions import Fraction

import pytest

from ncfacetkit import ComplexityLimitExceeded, Scenario


def simplest():
    return Scenario(4,2,2).prepare('(P0 + P1)/2 = (P2 + P3)/2')


def test_hazra_table_1_facet_count_and_symmetries():
    s=simplest()
    r=s.facets()
    assert r.affine_dimension==6
    assert r.preparation_vertices==4
    assert r.measurement_vertices==4
    assert r.behavior_vertices==28
    assert len(r.facets)==24
    orbits=s.orbits(r)
    assert len(orbits)==2
    assert sorted(size for facet,size in orbits)==[8,16]


def test_hazra_nontrivial_inequality():
    s=simplest()
    _,_,_,behaviors,observable=s.vertices()
    ix={tuple(o):i for i,o in enumerate(observable)}
    values=[b[ix[1,0,0]]-b[ix[2,0,0]]-b[ix[0,1,0]]+b[ix[2,1,0]]
            for b in behaviors]
    assert max(values)==1
    assert any(f.bound==1 and
        dict(zip(f.coordinates,f.coefficients)).get('p(0|1,0)',0)==1 and
        dict(zip(f.coordinates,f.coefficients)).get('p(0|2,0)',0)==-1 and
        dict(zip(f.coordinates,f.coefficients)).get('p(0|0,1)',0)==-1 and
        dict(zip(f.coordinates,f.coefficients)).get('p(0|2,1)',0)==1
        for f in s.facets().facets)


def test_measurement_equivalence_facet_enumeration():
    s=simplest().measure('M[0|0] + M[0|1] = I')
    r=s.facets()
    assert r.measurement_vertices==2
    assert r.affine_dimension==3
    assert len(r.facets)==8


def test_no_preparation_equivalences():
    r=Scenario(2,1,2).facets()
    assert r.affine_dimension==2
    assert len(r.facets)==4


def test_partial_equivalence_is_supported():
    r=Scenario(4,2,2).prepare('P0=P1').facets()
    assert r.affine_dimension==6
    assert len(r.facets)==12


def test_three_outcomes():
    r=Scenario(2,1,3).facets()
    assert r.affine_dimension==4
    assert len(r.facets)==6


def test_hazra_scenario_2_bound():
    s=Scenario(4,3,2).prepare('(P0 + P1 + P2)/3 = (P0 + P3)/2')
    _,_,_,behaviors,observable=s.vertices()
    ix={tuple(o):i for i,o in enumerate(observable)}
    values=[-b[ix[0,0,0]]+2*b[ix[1,0,0]]+b[ix[0,1,0]]-2*b[ix[2,1,0]]
            for b in behaviors]
    assert max(values)==2


def test_accept_implicit_multiplication_and_output():
    s=Scenario(4,2,2).prepare('1/2(P0 + P1) = 1/2(P2 + P3)')
    assert len(s.facets().facets)==24


def test_reject_invalid_and_nonlinear_expressions():
    s=Scenario(4,2,2)
    with pytest.raises(ValueError):s.prepare('P0*P1 = P2')
    with pytest.raises(ValueError):s.prepare('P0 = P4')
    with pytest.raises(ValueError):s.measure('M[2|0] = M[0|1]')
    with pytest.raises(ValueError):s.measure('__import__("os").system("true") = I')


def test_complexity_guard_is_explicit():
    with pytest.raises(ComplexityLimitExceeded):
        simplest().facets(max_bases=1)
