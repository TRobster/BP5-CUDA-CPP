#pragma once
//
// Thin sparse-matrix helpers over Eigen.
//
// Everything on the host side is assembled with Eigen's SparseMatrix. Assembly
// is a one-time offline cost, so clarity beats cleverness here; the GPU only
// ever sees the finished CSR arrays.
//
// Index conventions (these must match ops_bp5.jl exactly or nothing lines up):
//
//   volume node  ->  idx = q + nqp*(r + nrp*s)          (q fastest)
//   global dof   ->  comp*Np + idx      for comp = 0,1,2
//
// which is Julia's `Is (X) Ir (X) Dq` under column-major storage, and the
// [A11 A12 A13; A21 ...] block layout of `locoperator_bp5`.
//
#include <Eigen/SparseCore>
#include <vector>

namespace bp5 {

using Sp  = Eigen::SparseMatrix<double>;   // column-major (Eigen default)
using Trip = Eigen::Triplet<double>;

// Reference-direction index. Matches the first index of C[a,i,b,j] and the
// 1-based (q,r,s) = (1,2,3) of ops_bp5.jl.
enum Dir : int { Q = 0, R = 1, S = 2 };

// Face indexing. 0-based here; ops_bp5.jl uses 1-based, so our face f is
// Thrase's face f+1.
//
//   0 : q = 0    x = Lx1   FAULT           Dirichlet
//   1 : q = Nq   x = Lx2   REMOTE          Dirichlet
//   2 : r = 0    y = Ly1                   traction-free
//   3 : r = Nr   y = Ly2                   traction-free
//   4 : s = 0    z = Lz1   FREE SURFACE    traction-free
//   5 : s = Ns   z = Lz2                   traction-free
//
constexpr int NFACE = 6;
inline Dir  face_dir (int f) { return static_cast<Dir>(f / 2); }  // 0,1->Q 2,3->R 4,5->S
inline bool face_is_low(int f) { return (f % 2) == 0; }           // low side => outward normal is -e_a
inline double face_normal_sign(int f) { return face_is_low(f) ? -1.0 : +1.0; }

// Grid sizes, carried together so nothing gets transposed by accident.
struct Grid {
  int nqp, nrp, nsp;                       // points per direction (N+1)
  int np()  const { return nqp * nrp * nsp; }        // nodes per component
  int ndof() const { return 3 * np(); }              // total dofs
  int nface(int f) const {                           // nodes on face f
    switch (face_dir(f)) {
      case Q: return nrp * nsp;
      case R: return nqp * nsp;
      default: return nqp * nrp;
    }
  }
  int vidx(int q, int r, int s) const { return q + nqp * (r + nrp * s); }
};

// I (X) I (X) M   for dir==Q,  I (X) M (X) I  for dir==R,  M (X) I (X) I  for dir==S.
// Built directly rather than by nested kron: cheaper and much easier to read.
Sp lift3(Dir dir, const Sp& m1d, const Grid& g);

// Generic Kronecker product, kron(A,B)[ra*B.rows()+rb, ca*B.cols()+cb] = A(ra,ca)*B(rb,cb).
// Same convention as Julia's `A (X) B`. Used for the face norm matrices.
Sp kron(const Sp& a, const Sp& b);

// Face restriction e_f : (np x nface(f)). e_f^T pulls a volume field onto face f.
// Column ordering per face:  Q-faces: r + nrp*s,  R-faces: q + nqp*s,  S-faces: q + nqp*r.
// Matches `e1 = Is (X) Ir (X) eq0` etc.
Sp face_restrict(int f, const Grid& g);

// Diagonal 0/1 projector onto the plane of face f, as an (np x np) matrix.
// This is the masking role that ops_bp5.jl's sJI_f plays: it is 1/sJ on the
// face plane and zero everywhere else, so it both scales and selects.
Sp face_projector(int f, const Grid& g);

// n x n diagonal matrix from a vector.
Sp diag(const std::vector<double>& d);

// n x n identity.
Sp eye(int n);

// Assemble a 3x3 block matrix (each block np x np) into one (3np x 3np) matrix.
// blocks[i][j] is the (i,j) component block, matching [A11 A12 A13; A21 ...].
Sp block3(const Sp blocks[3][3]);

// Largest |A - A^T| entry, relative to max|A|. Zero (to roundoff) for a
// correctly assembled HM; this is the cheapest test that catches a flipped
// sign in the SAT terms.
double symmetry_error(const Sp& a);

}  // namespace bp5
