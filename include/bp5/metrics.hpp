#pragma once
//
// Curvilinear metrics for the BP5 box, specialised to the affine map that
// BP5-QD_Driver.jl actually uses.
//
// The driver's xt/yt/zt are pure affine stretches from the reference cube
// [-1,1]^3 to the physical box, so the Jacobian matrix is diag(ax, ay, az) and
// every off-diagonal metric term vanishes:
//
//     J  = ax*ay*az                    (constant)
//     dq/dx = 1/ax = G[0],  dr/dy = G[1],  ds/dz = G[2],  all others 0
//
// With that, create_metrics_bp5's rank-4 coefficient array collapses from 81
// grids to 81 scalars:
//
//     C[a][i][b][j] = J * ( lambda * d(i,a) * d(j,b) * G[i]*G[j]
//                         + mu     * d(i,j) * d(a,b) * G[a]*G[a]
//                         + mu     * d(j,a) * d(i,b) * G[i]*G[j] )
//
// where a,b are reference directions (q,r,s) and i,j are displacement
// components (x,y,z). Sanity checks: C[0][0][0][0] = J*G0^2*(lambda+2mu) is the
// P-wave modulus; C[0][0][1][1] = J*lambda*G0*G1; C[0][1][1][0] = J*mu*G0*G1.
//
// TO GENERALISE (non-affine mesh or heterogeneous material): C becomes a field,
// and the three places that assume a scalar are (1) this struct, (2) the
// diagonal terms in assemble_A, which must call sbp_d2_var with a real
// coefficient vector, and (3) the mixed terms, where the scalar multiply
// becomes a diagonal-matrix multiply.
//
#include "bp5/params.hpp"
#include "bp5/sparse.hpp"

namespace bp5 {

struct Metrics {
  double alpha[3] = {0, 0, 0};   // physical half-width per direction
  double beta[3]  = {0, 0, 0};   // physical centre per direction
  double G[3]     = {0, 0, 0};   // d(reference)/d(physical) = 1/alpha
  double J        = 0.0;         // Jacobian determinant, constant
  double sJ[NFACE] = {0, 0, 0, 0, 0, 0};   // surface Jacobian per face, constant

  // C[a][i][b][j], indices as described above.
  double C[3][3][3][3] = {};

  static Metrics make(const Params& p);

  // Physical coordinate of a node along one direction.
  double coord(Dir d, int i, int n) const {
    const double ref = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(n - 1);
    return alpha[d] * ref + beta[d];
  }

  void print() const;
};

}  // namespace bp5
