# Scope and numerical limitations

- **Generality means expressivity, not finite runtime.** Finite preparations and finite-outcome measurements may be arbitrarily many, and linear operational equivalences may be arbitrary. The number of vertices and facets can nevertheless be intractably large.
- **No continuum of outcomes.** CV homodyne, heterodyne and continuously parametrized measurements require prior discretization. The software does not claim to provide facets for uncountable scenarios.
- **No transformations or instruments.** The ontic model is prepare-and-measure. Sequential processes, disturbance constraints and temporal noncontextuality require an additional structure.
- **The operational equivalences are exact.** Experimental finite-statistics violations require a tolerance model; the current membership function checks consistency up to a specified numerical tolerance and does not estimate confidence intervals.
- **Classical generalized noncontextuality, not quantum optimization.** Fixed-Hilbert-dimension see-saw and dimension-independent prepare-and-measure hierarchies must include the operator-level preparation and measurement identities. Bell NPA alone does not suffice.
- **Certification grades are surfaced.** Default high-dimensional facets use numerical discovery with rational validation. For a complete exact polyhedral conversion, use `cdd.gmp` and ensure that component vertices were also enumerated exactly. In large cases the optional exact backend can still run out of resources.
- **Contextual fraction is a numerical LP output.** Inspect the returned status and residual. It is defined relative to the full operational behavior polytope satisfying declared linear identities, not relative to an arbitrarily selected noise model.
- **Relabeling orbits are exhaustive only within the finite search budget.** The method raises rather than returning partial classes under the label of complete symmetry reduction.
