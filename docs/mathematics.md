# Why the normalization-weight construction is exact

Consider `n` operational preparations \(P_x\), effect labels \((y,z)\), convex equivalences \(\sum_x \alpha_{x|s}P_x\simeq\sum_x\alpha_{x|s'}P_x\), and arbitrary linear measurement-effect equivalences.

Let \(\nu(\lambda)=\frac1n\sum_x\mu(\lambda|x)\). Ignore ontic points for which \(\nu(\lambda)=0\), since all \(\mu(\lambda|x)=0\) there. Put \(q_x(\lambda)=\mu(\lambda|x)/\nu(\lambda)\). Then \(q_x\ge 0\), \(n^{-1}\sum_x q_x=1\), and every preparation equivalence becomes a homogeneous linear constraint on \(q\). The resulting polytope \(\mathcal Q\) is compact. Measurement response vectors \(\xi\) lie in another compact polytope \(\mathcal M\), cut out by the response normalization, positivity, and the measurement-effect equivalences.

Let \(q^{(i)}\) and \(\xi^{(j)}\) be vertices. Expanding each ontic \(q\) and \(\xi\) in these vertices, then integrating over \(\nu(\lambda)\), gives

\[
p(z|x,y)=\sum_{i,j}w_{ij}q_x^{(i)}\xi^{(j)}(z|y),\quad
w_{ij}\ge0,\quad \sum_{i,j}w_{ij}=1.
\]

**Normalization is essential:** \(\sum_z p(z|x,y)=\sum_{ij}w_{ij}q_x^{(i)}=1\) for every \(x\). Conversely, any such nonnegative weights define a noncontextual ontic model on the finite space \(\lambda=(i,j)\) by \(\mu((i,j)|x)=w_{ij}q_x^{(i)}\) and \(\xi(z|(i,j),y)=\xi^{(j)}(z|y)\). It satisfies the equivalences and the prescribed probabilities. This proves both inclusions in the finite noncontextual polytope characterization.

## Why weights on response vertices may be selected deterministically

Write \(t_i=\sum_j w_{ij}\). Then the normalized preparation-weight polytope is

\[
\mathcal T=\{t\ge0:\sum_i t_i q_x^{(i)}=1\ \forall x\}.
\]

The fiber above any \(t\) is a product of simplices, one for each \(i\) with \(t_i>0\): \(w_{ij}=t_i\pi(j|i)\). Every extreme \(w\) must have \(t\) an extreme point of \(\mathcal T\) and every \(\pi(j|i)\) supported at a single \(j\). Otherwise either a perturbation within one simplex or a feasible perturbation of \(t\) decomposes \(w\) into two distinct points. The converse follows because if \(t\) is extreme and each row is supported on one vertex, no convex decomposition is possible. Accordingly, it suffices to enumerate vertices \(t\) and assign a response vertex independently to each active \(q^{(i)}\).

The map from \(w\) to \(p\) is linear, so its images include every vertex of the physical behavior polytope, but some images may be duplicated or nonextreme. The software deduplicates images and computes their convex hull.

## Linear-score bound without a behavior hull

For \(S(p)=c_0+\sum_{x,y,z}c_{xyz}p(z|x,y)\), define

\[
g_i=\max_j\sum_{x,y,z}c_{xyz}q_x^{(i)}\xi^{(j)}(z|y).
\]

Then

\[
S_{\mathrm{NC}}^{\max}=c_0+\max_{t\in\mathcal T}\sum_i t_i g_i.
\]

Because this last expression is linear on the compact polytope \(\mathcal T\), it suffices to evaluate the vertices of \(\mathcal T\). This method can be practical even if there are too many behaviors or too many facets to list. Every bound comes with an explicit saturating model.

## No preparation equivalences

In the absence of preparation operational equivalences, each preparation may be assigned its own disjoint ontic support. Thus the allowed behavior set is the direct Cartesian product of one measurement response polytope \(\mathcal M\) per preparation. Its facets are lifted facets of the factors. The factor polytopes themselves are intersections of an affine subspace with a nonnegative orthant, so their facets are exposed by (some of) the probability nonnegativity conditions. This directly yields the facets without forming \(|\mathrm{vert}(\mathcal M)|^n\) behavior vertices.

## Computational certification

Exact component-vertex enumeration + exact V→H conversion (`cdd.gmp`) gives a globally complete rational facet description for a tractable finite input. Otherwise numerical candidate discovery may miss inequalities. Returning inequalities that are each proven to be supporting facets is **not** equivalent to certifying that no facet is missing.
