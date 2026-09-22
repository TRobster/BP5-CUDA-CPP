#pragma once
//
// Incomplete Cholesky with zero fill-in, IC(0):   A ~= L * L^T
//
// L is constrained to exactly the sparsity pattern of the lower triangle of A.
// No fill-in is allowed, which is what makes it cheap -- and what makes it an
// approximation rather than a factorisation. Applying it means two sparse
// triangular solves:
//
//     L z = x        then        L^T y = z
//
// Usage:
//     IncompleteCholesky ic;
//     ic.setup(n, rowptr, colind, val);   // host CSR in, factors on the device
//     ic.apply(d_x, d_y);                 // y = (L L^T)^-1 x, device pointers
//
// The input may be the full symmetric matrix or just its lower triangle; only
// the lower triangle is ever read. A must be symmetric positive definite --
// IC(0) does no pivoting, so an indefinite matrix shows up as a zero pivot and
// setup() fails with the offending row.
//
#include <cusparse.h>

namespace bp5 {

class IncompleteCholesky {
 public:
  IncompleteCholesky() = default;
  ~IncompleteCholesky();

  IncompleteCholesky(const IncompleteCholesky&) = delete;
  IncompleteCholesky& operator=(const IncompleteCholesky&) = delete;

  // Factor. Returns false and explains why on failure.
  //
  // `shift` adds alpha*I to the diagonal before factoring. IC(0) has no
  // pivoting and is only guaranteed for M-matrices, so a matrix that is SPD but
  // not an M-matrix can still break down. Shifting is the standard remedy
  // (Manteuffel 1980): raise alpha until it factors, at the cost of
  // preconditioner quality. Default 0 means factor the matrix as given.
  bool setup(int n, const int* rowptr, const int* colind, const double* val,
             double shift = 0.0);

  // y = (L L^T)^-1 x. Both are device pointers of length n, and must not alias.
  bool apply(const double* d_x, double* d_y);

  int  size()  const { return n_; }
  bool ready() const { return ready_; }

  // Range of |L[i][i]| after a successful factorisation. A minimum that is tiny
  // relative to the maximum is the signature of a *near* breakdown: csric02
  // returns success, but the preconditioner it produced is badly conditioned.
  // Nothing else reports this, so check it before trusting a clean factor.
  bool factor_diagonal_range(double& lo, double& hi) const;

 private:
  void release();

  bool ready_ = false;
  int  n_ = 0;
  int  nnz_ = 0;

  cusparseHandle_t handle_ = nullptr;

  int*    d_rowptr_ = nullptr;
  int*    d_colind_ = nullptr;
  double* d_val_    = nullptr;   // overwritten with L by the factorisation
  double* d_z_      = nullptr;   // scratch for the intermediate solve

  cusparseSpMatDescr_t mat_L_   = nullptr;
  cusparseDnVecDescr_t vec_x_   = nullptr;
  cusparseDnVecDescr_t vec_y_   = nullptr;
  cusparseDnVecDescr_t vec_z_   = nullptr;
  cusparseSpSVDescr_t  spsv_L_  = nullptr;   // for  L z = x
  cusparseSpSVDescr_t  spsv_Lt_ = nullptr;   // for  L^T y = z
  void* buf_L_  = nullptr;
  void* buf_Lt_ = nullptr;
};

}  // namespace bp5
