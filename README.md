# bp5CUDA

A C++/CUDA rewrite of SEAS benchmark **BP5-QD** in 3-D, fusing the physics of
`Thrase.jl/src/3D/3D_structured` with the factor-once solver architecture of
`cudabasin/basin`.

This is **Tier 1, step 1**: single-block SBP-SAT assembly of the elasticity
operator `HM`, plus a sparse Cholesky factorisation. Nothing time-dependent yet.

## Why factor instead of iterate

BP5-QD's operator is constant for the whole simulation. Only the right-hand side
changes each step, through slip on the fault (face 0) and the affine remote
load `Vp*t/2` on face 1. Thrase runs a full CG on 6.4M unknowns for every one of
~1e5 ODE right-hand sides. Factoring once and reusing the factors turns each of
those into two triangular solves. That is the entire point of this rewrite.

## Build

```sh
module load gcc/13.1.0 eigen/3.4.0
make                    # host-only, no GPU needed
make run                # build + solve a small case

module load cuda/13.0
make CUDSS=1 CUDSS_ROOT=/path/to/cudss ARCH=sm_80
```

The default build has **no CUDA dependency**. `make_cudss()` returns `nullptr`
and the driver falls back to Eigen's host factorisation, which is what makes
the assembly testable on a login node.

## Run

```sh
./bp5_assemble ../Thrase.jl/examples/bp5-qd.dat --n 16 --backend eigen
sbatch scripts/submit.sh 32
```

| flag | meaning |
|---|---|
| `--n N` | override `Nx = Ny = Nz = N` (the `.dat` says 128) |
| `--backend eigen\|cudss` | which factorisation |
| `--dump DIR` | write `HM` as binary CSR for cross-checking against Julia |

## What it does

```
.dat  ->  Params        parse the BP5 parameter file
      ->  Metrics       affine metrics + the C[a][i][b][j] tensor
      ->  Sbp1D x3      1-D SBP operators, p = 2
      ->  assemble()    A, T, Z, S  ->  HM = -H*(A + S)
      ->  checks        symmetry, positive diagonal, consistency
      ->  factor+solve  Eigen LLT or cuDSS
```

### Correspondence to `ops_bp5.jl`

| here | Thrase |
|---|---|
| `Metrics::C[a][i][b][j]` | `metrics.C[a,i,b,j]` from `create_metrics_bp5` |
| `Operators::A[i][j]` | `D11..D33[i,j]` summed |
| `Operators::T[f][i][j]` | `T11_1 .. T33_6` |
| `Z[f][i][j]` (local to `assemble`) | `Z11_1 .. Z33_6` |
| `Operators::S[i][j]` | `S11 .. S33`, `mode="bp5"` branch |
| `Operators::HM` | `HM = HA + HS` |

Face indices here are 0-based; Thrase's face *f* is our *f-1*.

| face | plane | physical | condition |
|---|---|---|---|
| 0 | q = 0 | x = Lx1 | Dirichlet — **fault** |
| 1 | q = Nq | x = Lx2 | Dirichlet — remote load |
| 2,3 | r = 0, Nr | y = Ly1, Ly2 | traction-free |
| 4,5 | s = 0, Ns | z = Lz1, Lz2 | traction-free |

### Two things that are easy to get wrong

1. **The Dirichlet SAT uses transposed component indices** (`S[i][j]` takes
   `T[j][i]`) while the Neumann SAT uses natural ones (`S[i][j]` takes
   `T[i][j]`). That is the adjoint term versus the direct term, not a typo.
   Swap them and `HM` stops being symmetric.
2. **`BS` is orientation-neutral here** — both rows give `d/dq`, and the
   outward sign is applied per-face inside `T`. cudabasin's `bsx/bsy` bake the
   outward normal into the stencil instead. Do not copy those values across
   without flipping the low-end sign.

## Simplifications, and where to undo them

**Affine map.** `BP5-QD_Driver.jl` uses pure affine stretches, so the Jacobian
is `diag(ax,ay,az)`, all off-diagonal metric terms vanish, and the rank-4
coefficient array collapses from 81 grids to 81 scalars:

```
C[a][i][b][j] = J*( lambda*d(i,a)*d(j,b)*G_i*G_j
                  + mu    *d(i,j)*d(a,b)*G_a^2
                  + mu    *d(j,a)*d(i,b)*G_i*G_j ),   G_a = 1/alpha_a
```

This is also why `HM` only has ~15 nonzeros per row: most of those 81 entries
are zero, so the mixed `D_a*D_b` terms contribute far less fill than a general
curvilinear map would.

**Homogeneous material.** BP5 specifies uniform elastic properties, so `C` is
constant and `D2(c) == c*D2`.

To generalise (non-planar mesh, sedimentary basin): `Metrics::C` becomes a
field, `sbp_d2_var` gets called with a real coefficient vector instead of a
constant one, and the scalar multiplies in `assemble_A` become diagonal-matrix
multiplies. `sbp_d2_var` already takes a nodal coefficient array for this reason.

**p = 2 only.** `Sbp1D::make` throws on anything else. The p = 4 operators drop
in behind the same interface; nothing downstream is order-aware. BP5 accuracy
will want p = 4.

## Status — measured, on a login node

All four checks pass at every size tried (N = 8, 12, 16, 24):

```
symmetry  max|HM - HM^T| / max|HM| = 0.000e+00   (exactly zero, not just small)
diagonal  all positive
A*const   max|A u| = 2.2e-16      A annihilates rigid translation
A*linear  max|A u| = 8.5e-14      A annihilates linear fields in the interior
Cholesky  succeeds  =>  HM is symmetric positive definite
solve     ||HM x - b||/||b|| ~ 1e-15
```

Exact symmetry is the strong signal: it is a structural property, so it
confirms the SAT signs and index transposes are consistent. The two `A`
consistency tests pin down the volume operator, which symmetry alone does not
constrain.

### Order of accuracy — the 1-D MMS test

Those checks are all *consistency* checks: an operator of any order satisfies
them. Order of accuracy needs a problem with a known exact answer at every `h`,
which means a source term. `make test_converge && ./test_converge` solves

```
mu u'' + f = 0,   x in [-1,1],   f = cos(x) + x sin(x),   u = (3cos x + x sin x)/mu
```

and reports `r = mu*D2*u_exact + f`, zero in the continuum:

```
    N          h      ||r||_H    rate       max|r|    rate
    8       0.25   4.8768e-03       -   5.1758e-03       -
   16      0.125   1.2263e-03    1.99   1.3000e-03    1.99
   32     0.0625   3.0750e-04    2.00   3.2539e-04    2.00
   64    0.03125   7.6995e-05    2.00   8.1372e-05    2.00
  128    0.01562   1.9264e-05    2.00   2.0345e-05    2.00
```

Deliberately 1-D and scalar: it depends on `sbp1d.cpp` alone — no metrics, no
`C` tensor, no SAT, no faces — so a bad rate localises to one file.

### Solution error — `test_solution_cg` (needs a GPU)

`D2` alone is singular (`D2*1 = 0`, `D2*x = 0`), so the test above can only
measure truncation error. Adding a Dirichlet SAT at both ends — mirroring
`assemble.cpp` at one dimension, same `beta = 1, d = 3` penalty — gives an SPD
`HM`, and the boundary data enters through the same kernel, so `HB = -acc`
falls straight out. `test_solution_cg` then solves `HM u = H.*f + HB g` with
CG + IC(0) and compares to `u_exact`:

```
      N      ||e||_H    rate       max|e|    rate  iters   CG resid
      4   1.3744e-02       -   1.0482e-02       -      1    1.5e-16
      8   2.2327e-03    2.62   2.0097e-03    2.38      1    1.5e-16
     16   5.0112e-04    2.16   4.9843e-04    2.01      1    2.0e-16
     32   1.2328e-04    2.02   1.2436e-04    2.00      1    2.0e-16
     64   3.0749e-05    2.00   3.1074e-05    2.00      1    5.0e-16
    128   7.6846e-06    2.00   7.7675e-06    2.00      1    4.6e-16
```

The wobble at N = 4, 8 (rate 2.62, 2.38) is the small-N noise mentioned
earlier for the truncation test; it settles to a flat 2.00 by N = 64 and holds
there through N = 128.

**The truncation and solution rates differ, and that is the point:**

| quantity | rate (H) | rate (max) |
|---|---|---|
| truncation, interior only | 2.00 | 2.00 |
| truncation, full system incl. boundary | 1.51 | 1.01 |
| solution error | 2.02 | 2.00 |

The one-sided closure truncates at first order, but `HM^-1` smooths that away
and the solution is still `O(h^2)` — Gustafsson's theorem, confirmed. So
`--margin 0` reporting 1.01 is expected behaviour, not a defect in `sbp1d.cpp`.

Two cross-checks worth knowing:

- `tau = HM u_exact - b`, scaled by `1/H`, comes out **bit-identical** to the
  raw `--margin 0` residual. That is only true if `HB` is consistent with the
  SAT: `acc*u_exact` and `acc*g` cancel exactly. A sign error in `HB` breaks it.
- `max|e| / h^2` is 0.0319 at both N = 16 and N = 32 — a stable constant, which
  is what second order means.

CG converging in **one** iteration is genuine (residual at machine epsilon), not
a short circuit: `HM` here is tridiagonal apart from two corner entries, and
IC(0) on a tridiagonal matrix drops nothing, so it *is* the exact Cholesky
factor. This does not carry over to 3-D, where `HM` has ~15 nnz/row, IC(0) drops
real fill, and the measured result is 3.2–3.5x fewer iterations for a
wall-clock wash.

### Scaling, and why cuDSS is not optional

Assembly is cheap and scales linearly. The **host** factorisation does not:

| N | dofs | nnz | assemble | Eigen LLT factor |
|---|---|---|---|---|
| 12 | 6,591 | 102k | 0.02 s | 1.5 s |
| 16 | 14,739 | 226k | 0.05 s | 17.7 s |
| 24 | 46,875 | 714k | 0.15 s | 362 s |
| 32 | 107,811 | 1.6M | 0.4 s | > 10 min (killed) |

Eigen's `SimplicialLLT` is scalar, single-threaded and AMD-ordered — fine for
validation, hopeless past N ~ 24. **Treat the host path as a correctness
oracle only.**

Projecting the assembly to the benchmark's N = 128: 6.44M dofs, ~98M nonzeros,
~1.2 GB for `HM` itself. That part is comfortable. The open question is
factorisation fill-in, which is what the cuDSS run has to answer, and which is
the fork in the road between Tier 1 and Tier 2.

## Not yet verified

- **The cuDSS backend has never been compiled or run.** Neither cuDSS nor a GPU
  is present on the login node, and cuDSS does not ship with the `cuda/13.0`
  module — it needs a separate install. The code follows the documented
  `cudssCreate` / `ANALYSIS` / `FACTORIZATION` / `SOLVE` flow with
  `CUDSS_MTYPE_SPD` + `CUDSS_MVIEW_UPPER`, but expect to debug it on first run.
- **Nothing has been compared against Julia numerically.** `--dump` writes `HM`
  as CSR for exactly that purpose, but `ops_bp5.jl` cannot currently run —
  `var_3D_D2q_fast` and its `r`/`s` siblings are called and never defined. That
  has to be fixed on the Julia side before rung 2 of the validation ladder is
  possible. The checks above are self-consistency, not cross-validation.

## Next

1. `HB` in 3-D — the boundary-data lifting operator, and `bdry_vec_strip!`.
   Gives a physically meaningful right-hand side instead of a manufactured one.
   **The 1-D version now exists** (`converge.cpp`, `build()`), and it showed the
   construction is simpler than it looks: the SAT acts on `(u_f - g_f)`, so the
   same kernel serves both halves and `HB = -acc`. Scaling that up needs two
   changes in `assemble.cpp` — promote `Z` from a local to an `Operators`
   member, and build the rectangular `lift_f = e_f sJ_f H_f` alongside the
   square `face_mass`. Neither touches `HM`.
2. `computetraction` on face 0.
3. Rate-and-state friction as a CUDA kernel, one thread per fault node.
   Port `newtbndv_vectorized`, but fix the convergence test — `ops_bp5.jl:1818`
   reads `abs.(dV .< tol)`, so `abs` wraps the comparison rather than `dV` and
   the test accepts any negative Newton step.
4. Tsit5 with step rejection.
5. Measure the N = 128 factorisation. If the fill fits, build the fault DtN
   precompute (Tier 3). If not, go multiblock (Tier 2).

## Layout

```
include/bp5/  sparse.hpp   CSR/kron/lift helpers over Eigen, index conventions
              params.hpp   .dat reader
              sbp1d.hpp    1-D SBP operators
              metrics.hpp  affine metrics + C tensor
              assemble.hpp A, T, Z, S, HM
              factor.hpp   factorisation interface
src/          matching .cpp, plus factor_cudss.cu and its stub
scripts/      submit.sh    SLURM
```
