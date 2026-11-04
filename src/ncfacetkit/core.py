"""Generalized-noncontextual facet construction for finite prepare-and-measure scenarios.

Normalized-epistemic-state construction from Hazra et al., Quantum (2026),
"Efficient Computation of Generalized Noncontextual Polytopes and Quantum
violation of their Facet Inequalities". No assumptions about Hilbert dimension.
"""
from __future__ import annotations

import ast
import itertools
import json
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction as F
from pathlib import Path
from typing import Iterable

import numpy as np
import sympy as sp
from scipy.spatial import ConvexHull, QhullError


class ComplexityLimitExceeded(RuntimeError):
    """Exact complete enumeration exceeded the user-selected complexity budget."""


@dataclass
class Linear:
    coefficients: dict[str, F] = field(default_factory=dict)
    constant: F = F(0)

    @staticmethod
    def scalar(x):
        return Linear({}, F(x))

    def __add__(self, other):
        if not isinstance(other, Linear): other = Linear.scalar(other)
        c = self.coefficients.copy()
        for k,v in other.coefficients.items(): c[k] = c.get(k,F(0)) + v
        return Linear({k:v for k,v in c.items() if v}, self.constant + other.constant)

    __radd__ = __add__

    def __neg__(self):
        return Linear({k:-v for k,v in self.coefficients.items()}, -self.constant)

    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)

    def __mul__(self, other):
        if not isinstance(other,Linear): other=Linear.scalar(other)
        if self.coefficients and other.coefficients:
            raise ValueError('Operational equivalences must be linear; products of effects are unsupported.')
        if other.coefficients:
            return other * self.constant
        return Linear({k:v*other.constant for k,v in self.coefficients.items()},self.constant*other.constant)
    __rmul__=__mul__

    def __truediv__(self,other):
        if not isinstance(other,Linear): other=Linear.scalar(other)
        if other.coefficients or other.constant==0: raise ValueError('Only division by a nonzero scalar is supported.')
        return self*(F(1)/other.constant)


def _parse_linear(expression: str, allowed: str = 'operational') -> Linear:
    """Safe arithmetic parser: no eval, Python calls, attributes, or powers."""
    s=expression.replace('−','-').replace('×','*')
    s=re.sub(r'[ME]\[\s*(\d+)\s*\|\s*(\d+)\s*\]',r'M\1_\2',s)
    s=re.sub(r'M(\d+)\|(\d+)',r'M\1_\2',s)
    s=re.sub(r'P\[(\d+)\]',r'P\1',s)
    s=re.sub(r'(?<=\d)\s*(?=[PMI])','*',s)
    s=re.sub(r'(?<=\d)\s*(?=\()','*',s)
    s=re.sub(r'(?<=\))\s*(?=[PMI\d])','*',s)
    # Fix common 1/2(P0+P1) via preceding digit rule.
    tree=ast.parse(s,mode='eval')
    def read(t):
        if isinstance(t,ast.Expression):return read(t.body)
        if isinstance(t,ast.Constant) and type(t.value) in (int,float):
            return Linear.scalar(F(str(t.value)))
        if isinstance(t,ast.Name):
            if t.id=='I':return Linear.scalar(1)
            if not re.fullmatch(r'P\d+|M\d+_\d+' + (r'|prob_\d+_\d+_\d+' if allowed=='probability' else ''),t.id):
                raise ValueError('Unknown symbol '+t.id)
            return Linear({t.id:F(1)},F(0))
        if isinstance(t,ast.UnaryOp):
            if isinstance(t.op,ast.USub): return -read(t.operand)
            if isinstance(t.op,ast.UAdd): return read(t.operand)
        if isinstance(t,ast.BinOp):
            a,b=read(t.left),read(t.right)
            if isinstance(t.op,ast.Add):return a+b
            if isinstance(t.op,ast.Sub):return a-b
            if isinstance(t.op,ast.Mult):return a*b
            if isinstance(t.op,ast.Div):return a/b
        raise ValueError('Unsupported expression syntax: '+ast.dump(t))
    return read(tree)


def _sides(statement: str):
    parts=re.split(r'\s*(?:==|=|≃|∼|~)\s*',statement)
    if len(parts)!=2: raise ValueError('Use one equality sign, for example (P0+P1)/2 = (P2+P3)/2')
    return tuple(_parse_linear(s) for s in parts)


def _independent_rows(rows: list[list[F]], rhs: list[F]):
    M=sp.Matrix([[sp.Rational(x.numerator,x.denominator) for x in row] for row in rows])
    b=sp.Matrix([sp.Rational(x.numerator,x.denominator) for x in rhs])
    if M.rows==0:return [],[]
    if M.rank()!=M.row_join(b).rank():raise ValueError('Inconsistent operational-equivalence constraints')
    pivot_rows=list(M.T.rref()[1])
    return [rows[i] for i in pivot_rows],[rhs[i] for i in pivot_rows]


def _fraction_solve(A:list[list[F]],b:list[F]):
    """Pure rational Gauss-Jordan solve; None when matrix is singular."""
    n=len(A)
    work=[[F(a) for a in row]+[F(bi)] for row,bi in zip(A,b)]
    for j in range(n):
        pivot=next((i for i in range(j,n) if work[i][j]),None)
        if pivot is None:return None
        work[j],work[pivot]=work[pivot],work[j]
        u=work[j][j]
        work[j]=[v/u for v in work[j]]
        for k in range(n):
            if k==j:continue
            factor=work[k][j]
            if factor:
                work[k]=[v-factor*w for v,w in zip(work[k],work[j])]
    return tuple(row[-1] for row in work)


def _vertices_cdd(rows:list[list[F]],rhs:list[F]):
    """Use the GMP rational cddlib backend, if available."""
    import cdd.gmp as cdd
    n=len(rows[0]);r=len(rows)
    inequalities=[[F(0),*[F(int(i==j)) for i in range(n)]] for j in range(n)]
    equalities=[[-value,*row] for row,value in zip(rows,rhs)]
    matrix=cdd.matrix_from_array(inequalities+equalities,
        lin_set=set(range(n,n+r)),rep_type=cdd.RepType.INEQUALITY)
    poly=cdd.polyhedron_from_matrix(matrix)
    generators=cdd.copy_generators(poly)
    result=[]
    for entry in generators.array:
        if entry[0]!=1:raise ValueError('The input component polytope is unbounded.')
        q=tuple(F(v) for v in entry[1:])
        if not all(v>=0 for v in q):raise RuntimeError('cdd returned a negative vertex.')
        if not all(sum(c*x for c,x in zip(row,q))==b for row,b in zip(rows,rhs)):
            raise RuntimeError('cdd vertex failed exact equality check.')
        result.append(q)
    if not result:raise ValueError('The component polytope is infeasible.')
    return sorted(set(result))


def _enumerate_standard_vertices(rows: list[list[F]], rhs: list[F],
                                 max_bases: int, *, label: str):
    """Enumerate basic feasible solutions of {x>=0, Ax=b}; verify each exactly."""
    rows,rhs=_independent_rows(rows,rhs)
    if not rows: raise ValueError('A polytope requires at least one independent equality.')
    n=len(rows[0]); r=len(rows)
    if r>n:raise ValueError('More independent equalities than variables')
    bases=math.comb(n,r)
    if bases>max_bases:
        raise ComplexityLimitExceeded(f'{label}: {bases:,} bases exceed max_bases={max_bases:,}. '
                                      'Reduce the scenario or increase the budget.')
    # Small enumeration is genuinely exact from beginning to end.
    # For larger combinatorial basis counts, prefer exact GMP cddlib.
    if bases>10_000:
        try:import cdd.gmp
        except ImportError:pass
        else:return _vertices_cdd(rows,rhs)
    exact=(bases<=10_000)
    if not exact:
        Am=np.array([[float(v) for v in row] for row in rows],dtype=float)
        bm=np.array([float(x) for x in rhs],dtype=float)
    unique: dict[tuple[F,...],None]={}
    for idx in itertools.combinations(range(n),r):
        if exact:
            mat=[[row[i] for i in idx] for row in rows]
            sol=_fraction_solve(mat,rhs)
            if sol is None or any(v<0 for v in sol):continue
            xf=sol
        else:
            A=Am[:,idx]
            try:x=np.linalg.solve(A,bm)
            except np.linalg.LinAlgError:continue
            if min(x)<-1e-8 or np.max(np.abs(A@x-bm))>1e-7:continue
            xf=tuple(F(float(t)).limit_denominator(1_000_000) for t in x)
            if any(a<0 for a in xf):continue
            if any(sum(row[i]*xf[j] for j,i in enumerate(idx))!=bb for row,bb in zip(rows,rhs)):
                mat=[[row[i] for i in idx] for row in rows]
                sol=_fraction_solve(mat,rhs)
                if sol is None or any(v<0 for v in sol):continue
                xf=sol
        full=[F(0)]*n
        for j,i in enumerate(idx):full[i]=xf[j]
        unique[tuple(full)]=None
    if not unique:raise ValueError(f'{label} has no feasible vertices. Check the equivalences.')
    return list(unique)


def _reduce_fraction_vector(v):
    denom=math.lcm(*(x.denominator for x in v))
    integers=[int(x*denom) for x in v]
    gcd=math.gcd(*integers)
    if gcd:integers=[a//gcd for a in integers]
    return tuple(integers)


@dataclass(frozen=True)
class Facet:
    """Inequality sum_i coefficients[i] * coordinates[i] <= bound."""
    coefficients: tuple[int,...]
    bound: int
    coordinates: tuple[str,...]
    tight_vertices: int

    def __str__(self):
        terms=[]
        for c,s in zip(self.coefficients,self.coordinates):
            if c==0:continue
            if c==1:terms.append('+'+s)
            elif c==-1:terms.append('-'+s)
            else:terms.append(('+' if c>0 else '')+str(c)+'*'+s)
        lhs=' '.join(terms).lstrip('+').strip().replace('+ -','- ')
        return f'{lhs or "0"} <= {self.bound}'

    def to_dict(self):
        return {'coefficients':list(self.coefficients),'bound':self.bound,
                'coordinates':list(self.coordinates),'inequality':str(self),
                'tight_vertices':self.tight_vertices}


@dataclass
class FacetResult:
    facets: list[Facet]
    coordinates: tuple[str,...]
    preparation_vertices: int
    measurement_vertices: int
    preparation_weight_vertices: int
    behavior_vertices: int
    affine_dimension: int
    preparation_equivalences: list[str]
    measurement_equivalences: list[str]
    backend: str = "qhull-validated"
    completeness: str = "candidate facets individually exact-verified; numerical enumeration not an exact completeness certificate"

    def to_dict(self):
        return {'metadata':{
            'preparation_vertices':self.preparation_vertices,
            'measurement_vertices':self.measurement_vertices,
            'preparation_weight_vertices':self.preparation_weight_vertices,
            'behavior_vertices':self.behavior_vertices,
            'affine_dimension':self.affine_dimension,
            'independent_coordinates':list(self.coordinates),
            'preparation_equivalences':self.preparation_equivalences,
            'measurement_equivalences':self.measurement_equivalences,
            'backend':self.backend,'completeness':self.completeness},
            'facets':[f.to_dict() for f in self.facets]}

    def save_json(self,path):
        Path(path).write_text(json.dumps(self.to_dict(),indent=2)+'\n',encoding='utf8')

    def save_csv(self,path):
        import csv
        with Path(path).open('w',encoding='utf8',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow([*self.coordinates,'bound','tight_vertices'])
            for facet in self.facets:
                writer.writerow([*facet.coefficients,facet.bound,facet.tight_vertices])

    def save_latex(self,path):
        lines=[r'% Automatically generated noncontextual facet inequalities',
               r'\begin{align*}']
        for i,facet in enumerate(self.facets):
            parts=[]
            for coef,var in zip(facet.coefficients,facet.coordinates):
                if not coef: continue
                match=re.fullmatch(r'p\((\d+)\|(\d+),(\d+)\)',var)
                if match:
                    z,x,y=match.groups()
                    term=rf'p({z}\mid {x},{y})'
                else:term=var
                sign='+' if coef>0 else '-'
                mag=abs(coef)
                parts.append(f'{sign}{"" if mag==1 else mag}'+term)
            expr=' '.join(parts).lstrip('+').strip() or '0'
            ending=r'\\' if i<len(self.facets)-1 else ''
            lines.append(rf'{expr} &\le {facet.bound}{ending}')
        lines.append(r'\end{align*}')
        Path(path).write_text('\n'.join(lines)+'\n',encoding='utf8')

    def save(self,path):
        ext=Path(path).suffix.lower()
        if ext=='.json':return self.save_json(path)
        if ext=='.csv':return self.save_csv(path)
        if ext=='.tex':return self.save_latex(path)
        raise ValueError('Output extension must be .json, .csv, or .tex')

    def show(self,limit=30):
        print(f'{len(self.facets)} facets | dimension {self.affine_dimension} | '
              f'{self.behavior_vertices} behavior vertices')
        for f in self.facets[:limit]:print(' ',f)
        if len(self.facets)>limit:print(f'  ... {len(self.facets)-limit} more')


class Scenario:
    """A prepare-and-measure scenario with linear operational equivalences.

    Parameters
    ----------
    preparations : int
      Number of preparation procedures.
    outcomes : int | list[int]
      Outcomes for each measurement setting (or same integer for all settings).
    measurements : int | None
      Required when outcomes is a single integer.
    """
    def __init__(self,preparations:int,measurements:int|None=None,outcomes:int|list[int]=2):
        if preparations<1:raise ValueError('preparations must be positive')
        if isinstance(outcomes,int):
            if measurements is None:raise ValueError('Specify measurements when outcomes is a single integer')
            outs=[outcomes]*measurements
        else:
            outs=list(outcomes)
            if measurements is not None and measurements!=len(outs):raise ValueError('Measurement/outcome counts disagree')
        if not outs or min(outs)<2:raise ValueError('Each measurement needs at least 2 outcomes')
        self.preparations=preparations
        self.outcomes=outs
        self._peqs=[]
        self._meqs=[]

    def prepare(self,expression:str):
        lhs,rhs=_sides(expression)
        if lhs.constant or rhs.constant:raise ValueError('Preparation equivalences cannot contain constants')
        if any(not re.fullmatch(r'P\d+',k) or int(k[1:])>=self.preparations for k in list(lhs.coefficients)+list(rhs.coefficients)):
            raise ValueError('Preparation variable outside the scenario')
        a,b=sum(lhs.coefficients.values()),sum(rhs.coefficients.values())
        if a!=b or a<=0:raise ValueError('Both sides must have the same positive total preparation weight')
        if any(v<0 for v in list(lhs.coefficients.values())+list(rhs.coefficients.values())):
            raise ValueError('Preparation mixtures must have nonnegative coefficients')
        self._peqs.append((lhs*(F(1)/a),rhs*(F(1)/b),expression))
        if hasattr(self,'_cache'):self._cache.clear()
        return self

    def measure(self,expression:str):
        lhs,rhs=_sides(expression)
        for k in list(lhs.coefficients)+list(rhs.coefficients):
            if not re.fullmatch(r'M\d+_\d+',k):raise ValueError('Measurement equivalences must use M[z|y]')
            z,y=map(int,k[1:].split('_'))
            if y>=len(self.outcomes) or z>=self.outcomes[y]:raise ValueError('Measurement effect outside the scenario')
        self._meqs.append((lhs,rhs,expression))
        if hasattr(self,'_cache'):self._cache.clear()
        return self

    def preparation_relation(self,left,right,label=None):
        """Declare an equivalence between two preparation mixtures as vectors.

        Accepts integer, float, string fraction or Fraction inputs.  Floating
        literals are decimal-rationalized, e.g. 0.1 -> 1/10.
        """
        if len(left)!=self.preparations or len(right)!=self.preparations:
            raise ValueError('Each preparation mixture vector must have one entry per preparation.')
        lhs=[F(str(v)) for v in left]; rhs=[F(str(v)) for v in right]
        if any(v<0 for v in lhs+rhs):raise ValueError('Mixture weights must be nonnegative.')
        if sum(lhs)!=sum(rhs) or sum(lhs)==0:
            raise ValueError('Preparation mixtures must have the same positive total weight.')
        lhs=[v/sum(lhs) for v in lhs];rhs=[v/sum(rhs) for v in rhs]
        L=Linear({f'P{i}':v for i,v in enumerate(lhs) if v},F(0))
        R=Linear({f'P{i}':v for i,v in enumerate(rhs) if v},F(0))
        self._peqs.append((L,R,label or f'coefficient vectors {left} = {right}'))
        if hasattr(self,'_cache'):self._cache.clear()
        return self

    def effect_relation(self,coefficients,rhs=0,label=None):
        """Specify sum_{z,y} coefficients[(z,y)] M[z|y] == rhs*I."""
        c={}
        for (z,y),v in coefficients.items():
            if not (0<=y<len(self.outcomes) and 0<=z<self.outcomes[y]):
                raise ValueError(f'Unknown measurement effect {(z,y)}')
            v=F(str(v))
            if v:c[f'M{z}_{y}']=v
        self._meqs.append((Linear(c,F(0)),Linear.scalar(F(str(rhs))),
                           label or f'effect relation {coefficients} = {rhs}*I'))
        if hasattr(self,'_cache'):self._cache.clear()
        return self

    def to_dict(self):
        """Serialize a scenario with exact rational constraints as strings."""
        def encode(e):return {'terms':{k:str(v) for k,v in e.coefficients.items()},
                              'constant':str(e.constant)}
        return {'preparations':self.preparations,'outcomes':list(self.outcomes),
                'preparation_relations':[{'lhs':encode(a),'rhs':encode(b),'label':lab}
                                        for a,b,lab in self._peqs],
                'measurement_relations':[{'lhs':encode(a),'rhs':encode(b),'label':lab}
                                        for a,b,lab in self._meqs]}

    def _preparation_matrix(self):
        n=self.preparations
        # Use the uniform aggregate mixture (1/n) sum_x mu(lambda|x) as
        # the reference distribution. Unlike normalizing by just one side
        # of one operational equivalence, this is well-defined for *all*
        # preparations, including those outside the first equivalence.
        # It yields an always bounded q-polytope and the same normalized
        # noncontextual behaviors after the weight-polytope intersection.
        rows=[[F(1,n)]*n];rhs=[F(1)]
        for lhs,rr,_ in self._peqs:
            diff=lhs-rr
            rows.append([diff.coefficients.get(f'P{x}',F(0)) for x in range(n)])
            rhs.append(F(0))
        return rows,rhs

    def _measurement_matrix(self):
        effects=[(z,y) for y,k in enumerate(self.outcomes) for z in range(k)]
        cols={f'M{z}_{y}':i for i,(z,y) in enumerate(effects)}
        rows=[];rhs=[]
        for y in range(len(self.outcomes)):
            rows.append([F(int(setting==y)) for z,setting in effects]);rhs.append(F(1))
        for lhs,rr,_ in self._meqs:
            diff=lhs-rr
            rows.append([diff.coefficients.get(f'M{z}_{y}',F(0)) for z,y in effects]); rhs.append(-diff.constant)
        return effects,rows,rhs

    def _component_vertices(self,max_bases=200_000):
        """Enumerate rational preparation, effect, and normalization-weight vertices."""
        if not hasattr(self,'_cache'):
            self._cache={}
        if max_bases in self._cache:
            return self._cache[max_bases]
        prows,pb=self._preparation_matrix()
        prep=_enumerate_standard_vertices(prows,pb,max_bases,label='Preparation polytope')
        effects,mrows,mb=self._measurement_matrix()
        meas=_enumerate_standard_vertices(mrows,mb,max_bases,label='Measurement polytope')
        # The normalized sum of epistemic distributions induces a compact
        # weight polytope for q vertices. Grouping weights by q is exact.
        weights=_enumerate_standard_vertices([list(col) for col in zip(*prep)],
                                            [F(1)]*self.preparations,
                                            max_bases,label='Normalization-weight polytope')
        if not hasattr(self,'_cert_cache'):self._cert_cache={}
        matrices=[(prows,pb),(mrows,mb),([list(c) for c in zip(*prep)],[F(1)]*self.preparations)]
        basis_counts=[]
        for rowset,bset in matrices:
            indep,_=_independent_rows(rowset,bset)
            basis_counts.append(math.comb(len(indep[0]),len(indep)))
        try:import cdd.gmp
        except ImportError:has_gmp=False
        else:has_gmp=True
        self._cert_cache[max_bases] = has_gmp or all(b<=10_000 for b in basis_counts)
        self._cache[max_bases]=(prep,meas,weights)
        return prep,meas,weights

    def vertices(self,max_bases=200_000,max_products=400_000):
        """Enumerate exact rational behavior vertices (potentially exponential)."""
        prep,meas,weights=self._component_vertices(max_bases)
        effects=self._measurement_matrix()[0]
        observable=[(x,y,z) for y,k in enumerate(self.outcomes) for z in range(k-1)
                     for x in range(self.preparations)]
        effect_map={ef:i for i,ef in enumerate(effects)}
        expected=sum(len(meas)**sum(w>0 for w in weight) for weight in weights)
        if expected>max_products:
            raise ComplexityLimitExceeded(f'{expected:,} behavior combinations exceed '
                                          f'max_products={max_products:,}.')
        behaviors:dict[tuple[F,...],None]={}
        for weight in weights:
            support=[i for i,a in enumerate(weight) if a>0]
            for assigned in itertools.product(range(len(meas)),repeat=len(support)):
                row=[]
                for x,y,z in observable:
                    j=effect_map[z,y]
                    row.append(sum(weight[i]*prep[i][x]*meas[k][j]
                                   for i,k in zip(support,assigned)))
                behaviors[tuple(row)]=None
        return prep,meas,weights,list(behaviors),observable

    def facets(self,*,max_bases=200_000,max_products=400_000,max_hull_vertices=10_000,backend='auto'):
        """Generate all irredundant facet inequalities, with exact rational verification.

        SciPy's Qhull proposes supporting hyperplanes; SymPy reconstructs their
        exact rational coefficients and validates them against every vertex.
        Algorithm is exhaustive when the exact vertex enumeration and Qhull
        succeed. Coordinates are an independent subset of observable probabilities.
        """
        if backend not in ('auto','qhull','cdd'):
            raise ValueError("backend must be 'auto', 'qhull', or 'cdd'")
        # When preparations are unrestricted, the noncontextual polytope
        # factorizes as a product of response polytopes.  Avoid exponentially
        # many explicit Cartesian-product vertices in this important case.
        if not self._peqs:
            return self._unrestricted_facets(max_bases=max_bases)
        prep,meas,weights,behaviors,observables=self.vertices(max_bases,max_products)
        if len(behaviors)>max_hull_vertices:
            raise ComplexityLimitExceeded('Too many behavior vertices for convex-hull processing.')
        b0=behaviors[0]
        D=sp.Matrix([[sp.Rational(x-y) for x,y in zip(b,b0)] for b in behaviors[1:]])
        pivots=tuple(int(i) for i in D.rref()[1]) if D.rows else ()
        d=len(pivots)
        names=tuple(f'p({z}|{x},{y})' for x,y,z in observables)
        chosen=tuple(names[i] for i in pivots)
        if d==0: facets=[]
        else:
            coords=[tuple(b[i] for i in pivots) for b in behaviors]
            if d==1:
                mins=min(x[0] for x in coords); maxs=max(x[0] for x in coords)
                raw=[_reduce_fraction_vector((F(-1),-mins)),
                     _reduce_fraction_vector((F(1),maxs))]
            else:
                cnum=np.array([[float(v) for v in row] for row in coords])
                use_cdd=(backend=='cdd')
                if backend=='auto':
                    try:import cdd.gmp
                    except ImportError:use_cdd=False
                    else:use_cdd=True
                if use_cdd:
                    raw=_facets_cdd(coords)
                else:
                    try:
                        hull=ConvexHull(cnum)
                    except QhullError as exc:
                        raise RuntimeError('Numerical hull failed. Try backend="cdd" with pycddlib installed.') from exc
                    raw=[]
                    seen_supports=set()
                    for plane in hull.equations:
                        # Numerical hull only *discovers* candidates. Every
                        # candidate normal is reconstructed using rational
                        # algebra and validated over all rational vertices.
                        close=tuple(i for i,p in enumerate(cnum)
                                    if abs(float(np.dot(plane[:-1],p)+plane[-1]))<1e-7)
                        if close in seen_supports:continue
                        seen_supports.add(close)
                        matrix=sp.Matrix([[sp.Rational(coords[i][j]) for j in range(d)]+[1]
                                          for i in close])
                        pivrow=matrix.T.rref()[1]
                        if len(pivrow)!=d:
                            raise RuntimeError('Numerical facet candidate lacks exact affine rank. '
                                               'Use backend="cdd" for an exact H-representation.')
                        kernel=matrix[list(pivrow),:].nullspace()
                        if len(kernel)!=1:
                            raise RuntimeError('Failed exact reconstruction of a numerical facet candidate.')
                        vals=tuple(F(int(v.p),int(v.q)) for v in kernel[0])
                        ints=_reduce_fraction_vector(vals)
                        normal=ints[:-1];const=ints[-1]
                        evaluations=[sum(a*b for a,b in zip(normal,x))+const for x in coords]
                        if max(evaluations)<=0:raw.append((*normal,-const))
                        elif min(evaluations)>=0:raw.append((*(-a for a in normal),const))
                        else:
                            raise RuntimeError('Numerical facet not an exact supporting hyperplane. '
                                               'Try backend="cdd".')
            found={}
            for *n,bound in raw:
                n=tuple(n)
                dot=[sum(a*x for a,x in zip(n,p)) for p in coords]
                if max(dot)!=bound:raise RuntimeError('A facet candidate failed exact equality validation.')
                active=[x for x,v in zip(coords,dot) if v==bound]
                if d>1:
                    aff=sp.Matrix([[sp.Rational(v-w) for v,w in zip(x,active[0])]
                                   for x in active[1:]])
                    if aff.rank()!=d-1:raise RuntimeError('Facet candidate has incorrect face dimension.')
                found[(*n,bound)]=len(active)
            facets=[Facet(tuple(k[:-1]),int(k[-1]),chosen,t) for k,t in sorted(found.items())]
        engine=('cdd-exact' if d>1 and use_cdd else 'qhull-rationally-validated' if d>1
                else 'exact-1d')
        claim=('exact rational H-representation' if engine=='cdd-exact' or d<=1
               else 'every candidate is an exact facet; numerical hull completeness not independently proven')
        if not self._cert_cache[max_bases]:
            claim += '; component-vertex numerical prescreen is not completeness-certified'
        return FacetResult(facets,chosen,len(prep),len(meas),len(weights),len(behaviors),d,
                           [x[2] for x in self._peqs],[x[2] for x in self._meqs],engine,claim)

    def noncontextual_bound(self,expression:str,*,max_bases=200_000):
        """Return the EXACT noncontextual bound for a linear probability expression.

        This avoids generating all facets and all normalized behaviors.
        """
        from .bounds import noncontextual_bound
        return noncontextual_bound(self,expression,max_bases=max_bases)

    def analyze(self,*,max_bases=200_000):
        """Preparation, response, weight-polytope complexity without building hull."""
        prep,meas,weights=self._component_vertices(max_bases)
        return {'preparations':self.preparations,'measurements':len(self.outcomes),
                'outcomes_per_measurement':list(self.outcomes),
                'preparation_equivalences':len(self._peqs),
                'measurement_equivalences':len(self._meqs),
                'preparation_vertices':len(prep),'measurement_vertices':len(meas),
                'normalization_weight_vertices':len(weights),
                'candidate_behavior_combinations':sum(len(meas)**sum(w>0 for w in t) for t in weights),
                'factorized_unrestricted_preparations':not self._peqs,
                'component_vertex_enumeration_certified':self._cert_cache[max_bases]}

    def is_noncontextual(self,data,*,tol=1e-8,max_bases=200_000):
        """Numerical LP membership for a normalized conditional probability table."""
        from .behavior import is_noncontextual
        return is_noncontextual(self,data,tol=tol,max_bases=max_bases)

    def contextual_fraction(self,data,*,tol=1e-8,max_bases=200_000):
        """Maximum noncontextual weight of a generalized operational behavior."""
        from .behavior import contextual_fraction
        return contextual_fraction(self,data,tol=tol,max_bases=max_bases)

    def _unrestricted_facets(self,*,max_bases=200_000):
        """Exact factorization for no preparation equivalences."""
        effects,rows,rhs=self._measurement_matrix()
        meas=_enumerate_standard_vertices(rows,rhs,max_bases,label='Measurement polytope')
        delta=sp.Matrix([[sp.Rational(a-b) for a,b in zip(m,meas[0])]
                         for m in meas[1:]])
        pivots=tuple(delta.rref()[1]) if delta.rows else ()
        d=len(pivots)
        names=tuple(f'p({effects[j][0]}|{x},{effects[j][1]})'
                    for x in range(self.preparations) for j in pivots)
        result_facets=[]
        if d:
            basis=sp.Matrix([[1]+[sp.Rational(m[j]) for j in pivots] for m in meas])
            independent_rows=list(basis.T.rref()[1])
            inverse=basis[independent_rows,:].inv()
            for x in range(self.preparations):
                for i,(z,y) in enumerate(effects):
                    target=sp.Matrix([sp.Rational(meas[k][i]) for k in independent_rows])
                    v=inverse*target
                    if any((basis*v)[k]!=sp.Rational(meas[k][i]) for k in range(len(meas))):
                        raise RuntimeError('Measurement coordinate reconstruction failed.')
                    tight=[m for m in meas if m[i]==0]
                    if not tight:continue
                    if d>1:
                        active_diff=sp.Matrix([[sp.Rational(a-b) for a,b in zip(
                                          (m[j] for j in pivots), (tight[0][j] for j in pivots))]
                                          for m in tight[1:]])
                        if active_diff.rank()!=d-1:continue
                    normal=[F(0)]*(self.preparations*d)
                    for k in range(d):normal[x*d+k]=-F(v[k+1])
                    bound=F(v[0])
                    prim=_reduce_fraction_vector([*normal,bound])
                    result_facets.append(Facet(tuple(prim[:-1]),prim[-1],names,
                                               len(tight)*len(meas)**(self.preparations-1)))
        result_facets=sorted(set(result_facets),key=lambda f:(f.coefficients,f.bound))
        independent,_=_independent_rows(rows,rhs)
        bases=math.comb(len(rows[0]),len(independent))
        try:import cdd.gmp
        except ImportError:has_gmp=False
        else:has_gmp=True
        certified=has_gmp or bases<=10_000
        statement=('exact: product of independently selectable measurement response polytopes'
                   if certified else 'analytic product factorization; component vertices only numerically prescreened')
        return FacetResult(result_facets,names,self.preparations,len(meas),1,
                           len(meas)**self.preparations,self.preparations*d,
                           [],[e[2] for e in self._meqs],
                           'exact-product-factorization' if certified else 'product-factorization-rational-checks',statement)

    def orbits(self,result:FacetResult,*,max_relabelings=5000):
        """Return all inequivalent facet classes when exhaustive relabeling is tractable."""
        from .symmetry import facet_orbits
        return facet_orbits(self,result,max_relabelings=max_relabelings)

    @classmethod
    def from_dict(cls,config:dict):
        obj=cls(config['preparations'],config.get('measurements'),config.get('outcomes',2))
        for eq in config.get('preparation_equivalences',[]):obj.prepare(eq)
        for eq in config.get('measurement_equivalences',[]):obj.measure(eq)
        def decode(d):return Linear({k:F(v) for k,v in d['terms'].items()},F(d.get('constant','0')))
        for relation in config.get('preparation_relations',[]):
            lhs,rhs=decode(relation['lhs']),decode(relation['rhs'])
            if lhs.constant or rhs.constant:raise ValueError('Invalid preparation relation constant')
            for x in (lhs,rhs):
                if any(not re.fullmatch(r'P\d+',k) or int(k[1:])>=obj.preparations for k in x.coefficients):
                    raise ValueError('Unknown preparation in structured input')
                if any(v<0 for v in x.coefficients.values()):raise ValueError('Negative preparation mixture')
            if sum(lhs.coefficients.values())!=sum(rhs.coefficients.values()):
                raise ValueError('Unbalanced preparation mixture')
            obj._peqs.append((lhs,rhs,relation.get('label','structured preparation relation')))
        for relation in config.get('measurement_relations',[]):
            lhs,rhs=decode(relation['lhs']),decode(relation['rhs'])
            for x in (lhs,rhs):
                for k in x.coefficients:
                    if not re.fullmatch(r'M\d+_\d+',k):raise ValueError('Unknown measurement symbol')
                    z,y=map(int,k[1:].split('_'))
                    if y>=len(obj.outcomes) or z>=obj.outcomes[y]:raise ValueError('Unknown measurement effect')
            obj._meqs.append((lhs,rhs,relation.get('label','structured measurement relation')))
        return obj


def _facets_cdd(coords):
    """Exact H-conversion using the GMP-rational pycddlib backend."""
    try:import cdd.gmp as cdd
    except ImportError as exc:
        raise ImportError('Exact backend needs pycddlib with GMP: pip install pycddlib') from exc
    generators=[[F(1),*map(F,v)] for v in coords]
    mat=cdd.matrix_from_array(generators,rep_type=cdd.RepType.GENERATOR)
    poly=cdd.polyhedron_from_matrix(mat)
    inequalities=cdd.copy_inequalities(poly)
    if inequalities.lin_set:
        raise RuntimeError('Expected full-dimensional coordinates; cdd returned affine equalities.')
    return [_reduce_fraction_vector([*[-F(a) for a in row[1:]],F(row[0])])
            for row in inequalities.array]
