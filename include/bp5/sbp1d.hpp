#pragma once
//
// One-dimensional diagonal-norm SBP operators on the reference interval.
//
// ops_bp5.jl calls `diagonal_sbp_D1`/`diagonal_sbp_D2` with xc = (-1,1), so the
// reference domain is [-1,1] in every direction and h = 2/N.
//
// Order: p = 2 only, for now. The p = 4 operators slot in behind the same
// interface (see `make` below); everything downstream is order-agnostic.
//
#include "bp5/sparse.hpp"

namespace bp5 {

// The five operators the 3-D assembly needs, for one direction.
//
//   h    reference spacing, 2/N
//   H    diagonal norm,  h*diag(1/2, 1, ..., 1, 1/2) for p = 2
//   HI   H^-1
//   D1   first derivative
//   D2   second derivative, constant coefficient
//   BS   boundary derivative: row 0 and row n-1 hold the one-sided d/dq
//        stencils, all other rows zero. This is `S0q + SNq` in ops_bp5.jl.
//
// Sign convention for BS: both rows give d/dq, NOT the outward normal
// derivative. ops_bp5.jl applies the outward sign explicitly in the traction
// operators (T_face1 carries a leading minus, T_face2 a plus), so BS must stay
// orientation-neutral. Note this differs from cudabasin, whose `bsx/bsy` bake
// the outward sign in -- do not copy those values across without flipping.
struct Sbp1D {
  int    n = 0;      // number of points (N+1)
  double h = 0.0;    // reference spacing
  Sp H, HI, D1, D2, BS;

  // p: interior order of accuracy. N: number of intervals.
  static Sbp1D make(int p, int N);
};

// Variable-coefficient second derivative, narrow stencil (Mattsson).
//
// Satisfies the SBP identity
//     H * D2(c) = -A(c) + c_{n-1} e_{n-1} bs_{n-1}^T - c_0 e_0 bs_0^T
// with A(c) symmetric positive semi-definite. `c` is the coefficient sampled at
// the n nodes.
//
// BP5 is homogeneous and the map is affine, so every coefficient this code
// currently feeds in is constant, and D2(c) == c * D2. It is kept as a separate
// entry point because heterogeneous material (a sedimentary basin, say) is the
// obvious next physics step and this is the one function that has to change.
Sp sbp_d2_var(int p, int N, const std::vector<double>& c);

}  // namespace bp5
