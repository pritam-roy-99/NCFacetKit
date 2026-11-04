# NCFacetKit

**A scenario-first Python toolkit for finite generalized-noncontextuality inequalities.**

Describe arbitrary finite preparation procedures, heterogeneous finite-outcome POVMs and rational linear operational equivalences. Compute noncontextual facets when feasible, the exact noncontextual bound of a user-entered linear inequality, numerical membership tests and contextual fractions.

## Install

```bash
git clone https://github.com/pritam-roy-99/NCFacetKit.git
cd NCFacetKit
python -m pip install -e .
python -m pytest -q
```

For exact GMP rational polyhedral enumeration (recommended for publishable complete facet lists):

```bash
python -m pip install -e ".[exact]"
```

## Quick start

```python
from ncfacetkit import Scenario

s = Scenario(preparations=4, measurements=2, outcomes=2)
s.prepare("(P0 + P1)/2 = (P2 + P3)/2")
result = s.facets()
result.show()
print(result.backend, result.completeness)
for inequality, orbit_size in s.orbits(result):
    print(inequality, orbit_size)

score = "p(0|1,0)-p(0|2,0)-p(0|0,1)+p(0|2,1)"
print(s.noncontextual_bound(score).value)  # 1 (exact Fraction)
result.save("facets.json")
```

The Hazra et al. four-preparation example gives **24 facets, two symmetry classes**, with orbit sizes 8 and 16.

## Heterogeneous POVM outcomes and effect equivalences

```python
s = Scenario(preparations=6, outcomes=[4, 4, 2])
s.prepare("(P0+P1)/2=(P2+P3)/2")
s.prepare("(P0+P1)/2=(P4+P5)/2")
s.measure("M[0|2]=M[0|0]+M[1|0]")
s.measure("M[0|2]=M[2|1]+M[3|1]")
print(s.analyze())
print(s.noncontextual_bound("p(0|0,0)+p(0|1,1)-p(1|2,0)"))
```

## Version 0.3.0 optimization

- Fraction-based exact row reduction and facet reconstruction without a required SymPy dependency.
- Integer NumPy matrix products for exact behavior vertices, with overflow-safe object arithmetic for large coefficients.
- Vectorized Qhull support checks and numerical-plane deduplication *before* exact rational verification.
- Cached behavior vertices and faster integer matrix operations for repeated inequality bounds.
- Five new optimization tests: **26 passing tests in GitHub Actions**.
- Reproducible timings with `python benchmarks/run.py`.

Representative local benchmarks: the 7,904-behavior-vertex six-preparation example improved from 1.12 seconds to 0.117 seconds (~9.6x), while the unrestricted 100-preparation facet case improved from 4.43 seconds to 1.50 seconds (~3.0x). Timings are machine- and scenario-dependent.

## Scientific guarantees and limitations

All finite numbers of preparations, measurement settings, outcomes, and rational **linear** operational equivalences are supported as inputs; the full facet output may still be computationally infeasible. The standard backend uses Qhull to *discover* candidate hyperplanes and verifies each candidate by rational arithmetic. **This does not independently certify completeness.** Use `backend="cdd"` with GMP pycddlib for complete exact facet enumeration when tractable. A complexity budget raises an explicit error rather than silently claiming exhaustiveness.

Fixed-dimensional quantum see-saw and prepare-and-measure NPA are **not** included yet. The core package currently computes classical generalized-noncontextual structures.

Documentation: [Mathematics](docs/mathematics.md) · [Limitations](docs/limitations.md) · [Roadmap](docs/roadmap.md). Example notebooks and JSON scenarios are in [examples](examples/).

Reference: S. Hazra *et al.*, *Efficient Computation of Generalized Noncontextual Polytopes and Quantum Violation of Their Facet Inequalities*, **Quantum** (2026), [arXiv:2406.09111](https://arxiv.org/abs/2406.09111).

**License:** MIT.
