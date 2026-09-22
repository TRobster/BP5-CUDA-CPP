#pragma once
//
// Single-block SBP-SAT assembly for 3-D linear elasticity, BP5 boundary
// conditions. This is the C++ counterpart of `locoperator_bp5(..., mode="bp5")`
// in Thrase.jl/src/3D/3D_structured/ops_bp5.jl.
//
// What gets built, in the notation of that file:
//
//   A[i][j]   volume operator,   A = (1/J) * sum_ab C[a][i][b][j] d_a d_b
//   T^f[i][j] traction operator on face f
//   Z^f[i][j] penalty  operator on face f
//   S[i][j]   SAT terms
//   HM        = -H * (A + S)        <- the symmetric positive definite system
//
// Boundary conditions (ops_bp5.jl:1046-1059):
//   faces 0,1 (x = Lx1 fault, x = Lx2 remote)  Dirichlet
//   faces 2,3,4,5                               traction-free Neumann
//
// The Dirichlet SAT uses transposed component indices (S[i][j] takes T[j][i])
// while the Neumann SAT uses natural ones (S[i][j] takes T[i][j]). That
// asymmetry is not a typo -- it is the adjoint term versus the direct term --
// and getting it backwards is the single easiest way to lose symmetry of HM.
//
#include "bp5/metrics.hpp"
#include "bp5/params.hpp"
#include "bp5/sbp1d.hpp"
#include "bp5/sparse.hpp"

#include <array>

namespace bp5 {

// Penalty constants, ops_bp5.jl:751-753.
constexpr double kPenaltyBeta = 1.0;
constexpr double kPenaltyD    = 3.0;

struct Operators {
  Grid grid;
  std::array<Sbp1D, 3> d1;      // one-dimensional operators per direction

  std::vector<double> Hvol;     // volume norm, diagonal (Hs (X) Hr (X) Hq)
  Sp A[3][3];                   // volume operator blocks
  Sp S[3][3];                   // SAT blocks
  Sp T[NFACE][3][3];            // traction operators per face
  Sp HM;                        // -H * (A + S), size 3np x 3np

  size_t nnz() const { return static_cast<size_t>(HM.nonZeros()); }
};

// Build everything. `p.SBPp` selects the operator order (only 2 for now).
Operators assemble(const Params& p, const Metrics& m);

// Rough host memory estimate for the assembled HM, in bytes. Assembly peaks
// well above this because of Eigen's intermediates -- see README.
size_t estimate_bytes(const Grid& g);

}  // namespace bp5
