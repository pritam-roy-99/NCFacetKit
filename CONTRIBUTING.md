# Contributing to NCFacetKit

Please add a small reproducible scenario and a regression test for every new polyhedral feature. Verify all vertices and inequalities with exact rational arithmetic where possible; do not label numerical completeness as an exact certificate. New backends should identify whether their V- and H-representations are exact and return a controlled exception if they cannot finish.

Run `python -m pytest -q` before submitting changes. Public research use should identify the NCFacetKit version, scenario equivalences, solver backend, certification status, and any symmetry assumptions. An issue should include the JSON scenario and the full error message.
