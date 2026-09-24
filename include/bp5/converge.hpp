#pragma once
/*
 Method of Manufactured Solutions, 1-D scalar case.
/   mu * u''(x) + f(x) = 0,        x in [-1, 1]

Pick u, differentiate it analytically to get f, feed f to the discrete
 operator, and the gap between them is the truncation error.

D2 operator

   D2 u ~= u'' so mu * D2 * u + f  ->  0   as h -> 0

`residual` forms exactly that. `source_rhs` forms the H-weighted source
 vector H .* f, which is the shape the right-hand side takes once this scales
 up: the 3-D system is HM u = Hvol .* f + HB g, weighted by the same H that
 the operator absorbs.

 The boundary rows of D2 carry a one-sided closure that is lower order than
 the interior stencil, so `margin` drops them before scoring. margin = 1 is
 exactly the closure rows for p = 2.

 assemble.hpp is included for kPenaltyBeta / kPenaltyD alone. They are
 constexpr, so this costs nothing at link time, and sharing them is the point:
 if the 3-D code changes its penalty and this test does not, the test stops
 validating anything.
*/
#include "bp5/assemble.hpp"
#include "bp5/sbp1d.hpp"

#include <vector>

namespace bp5 {

// The manufactured pair.
//
//   f(x) = cos(x) + x sin(x)
//   u(x) = ( 3 cos(x) + x sin(x) ) / mu
//
// Check:  u'' = ( -3cos(x) + 2cos(x) - x sin(x) ) / mu = -( cos(x) + x sin(x) ) / mu
//         so   mu*u'' + f = -f + f = 0.   Exact, for any mu.
//
// On [-1,1] this is under one period of the cosine, so it is well resolved even
// at N = 8. No wavenumber scaling is needed.
struct Mms {
  double mu = 1.0;

  double f(double x) const;
  double u(double x) const;
};

struct Mms2D 
{
  double mu;

  double f(double x, double y) const;
  double ux(double x, double y) const; 
  double uy(double x, double y) const; 
};

// Node coordinates on the reference interval: x_i = -1 + 2i/N.
std::vector<double> nodes(const Sbp1D& d);

// Diagonal of the SBP norm H. h/2 at the ends, h inside, for p = 2.
std::vector<double> h_diag(const Sbp1D& d);

// Sample u or f at the nodes.
std::vector<double> sample_u(const Sbp1D& d, const Mms& s);
std::vector<double> sample_f(const Sbp1D& d, const Mms& s);

// H .* f -- the source vector as the operator wants it weighted.
std::vector<double> source_rhs(const Sbp1D& d, const Mms& s);

// Residual of the discrete operator against the manufactured pair:
//
// r = mu * D2 * u_exact + f        (identically zero in the continuum)
//
// scored on nodes at least `margin` away from each end.
struct Residual {
  double h       = 0.0;   // grid spacing, 2/N
  double l2      = 0.0;   // sqrt( sum H_ii r_i^2 ), the SBP norm
  double linf    = 0.0;   // max |r|e                           
  int    nscored = 0;
};
Residual residual(const Sbp1D& d, const Mms& s, int margin = 1);

// Observed order between two grids differing by a factor `refine` in N.
double rate(double err_coarse, double err_fine, double refine = 2.0);

// - solution error --
//
// `residual` measures TRUNCATION error: given u_exact, apply the operator
// and see what is left over. Solution error is the other direction; solve for
// u_h and compare it to u_exact and it needs an invertible operator.
//
// D2 alone is not invertible. D2*1 = 0 and D2*x = 0, which is the discrete
// statement that u'' = g has no unique solution; add any a + b*x. Boundary
// conditions are what remove that null space.
//
// This mirrors assemble.cpp at one dimension and one component. Both ends are
// Dirichlet, matching BP5 faces 0 and 1 (fault and remote load). In 1-D a face
// is a single point, so the surface Jacobian and the face norm are both 1 and
// the face mass M_f collapses to the 0/1 mask P_f:
//
//   T^f = sign_f * mu * P_f * BS            traction   (assemble.cpp:112-126)
//   Z^f = (beta*d / H_00) * mu * P_f        penalty    (assemble.cpp:135-142)
//   acc = sum_f (T^f - Z^f)^T P_f           SAT kernel (assemble.cpp:154-164)
//   HM  = -H*(mu*D2) - acc                  since H*HI = I
//
// The boundary data goes through the SAME kernel, which is the whole content of
// HB: the SAT acts on (u_f - g_f), so splitting it gives acc*u for the operator
// and -acc*g for the right-hand side, where g is any volume vector carrying the
// boundary values at the end nodes. Hence simply
//
//   HB = -acc      and      HM u = H.*f + HB g.
//
struct System {
  Sp HM;                 // n x n, expected symmetric positive definite
  Sp HB;                 // n x n, boundary-data lifting, = -acc
  Sp T[2], Z[2];         // per end: 0 = left (x = -1), 1 = right (x = +1)
  Eigen::VectorXd b;     // H.*f + HB*g, assembled with the exact g
};

// Volume vector carrying the exact boundary data at the end nodes, zero inside.
Eigen::VectorXd boundary_data(const Sbp1D& d, const Mms& s);

System build(const Sbp1D& d, const Mms& s);

}  // namespace bp5
