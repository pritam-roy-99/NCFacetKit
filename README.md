# NCFacetKit

A scenario-first Python toolkit for finite generalized-noncontextuality inequalities.

Specify preparations, POVM outcomes, and operational equivalences; obtain noncontextual facets, exact classical bounds, noncontextual behavior membership, and contextual fractions.

## Quick start

```python
from ncfacetkit import Scenario

s = Scenario(preparations=4, measurements=2, outcomes=2)
s.prepare("(P0 + P1)/2 = (P2 + P3)/2")
r = s.facets()
r.show()
print(s.noncontextual_bound("p(0|1,0)-p(0|2,0)-p(0|0,1)+p(0|2,1)"))
```

## Installation

```bash
pip install -e .
pip install -e ".[exact]"
pytest -q
```

## Scope and guarantees

Finite prepare-and-measure scenarios, heterogeneous finite outcomes, and rational linear preparation/measurement operational equivalences. Not a general solver for arbitrary quantum instruments or nonlinear constraints. Small component polytopes use rational arithmetic; default Qhull candidate facet enumeration has exact candidate validation but does **not** generally certify completeness. Install optional `pycddlib` GMP support for stronger exact polyhedral enumeration. Computation can be expensive for large scenarios.

Reference: S. Hazra et al., *Efficient Computation of Generalized Noncontextual Polytopes and Quantum Violation of Their Facet Inequalities*, Quantum (2026), arXiv:2406.09111.

See the full release bundle for mathematical derivations, examples, tests, and additional documentation.
