#pragma once
//
// Preconditioned Conjugate Gradient on the GPU.
//
// Solves A x = b for symmetric positive definite A. The matrix-vector product
// is cuSPARSE SpMV, the vector arithmetic is cuBLAS, and the preconditioner is
// optional -- pass nullptr for plain CG.
//
// Usage:
//     ConjugateGradient cg;
//     cg.setup(n, rowptr, colind, val);          // FULL symmetric CSR
//     auto r = cg.solve(d_b, d_x, 1e-10, 2000, &ic);
//
// Note the asymmetry with IncompleteCholesky: that one wants the *lower
// triangle* because csric02 documents it that way, while CG wants the *full*
// matrix because cuSPARSE's generic SpMV has no symmetric mode. Both live on
// the device at once; at BP5 resolution that is roughly 1.2 GB plus 0.6 GB.
//
#include "bp5/conditioner.cuh"

#include <cublas_v2.h>
#include <cusparse.h>

namespace bp5 {

class ConjugateGradient {
 public:
  struct Result 
  {
    bool   converged = false;
    int    iterations = 0;
    double relative_residual = 0.0;   // ||b - A x|| / ||b||
  };

  ConjugateGradient() = default;
  ~ConjugateGradient();

  ConjugateGradient(const ConjugateGradient&) = delete;
  ConjugateGradient& operator=(const ConjugateGradient&) = delete;

  // Full symmetric CSR, zero-based.
  bool setup(int n, const int* rowptr, const int* colind, const double* val);

  // Solve A x = b. d_x is both the initial guess and the output, so zero it
  // first unless you have something better. Stops when ||r||/||b|| < tol or
  // after maxiter iterations. `precond` may be null.
  Result solve(const double* d_b, double* d_x, double tol, int maxiter,
               IncompleteCholesky* precond = nullptr);

  int  size()  const { return n_; }
  bool() const { return ready_; }

 private:
  void release();
  bool spmv(const double* d_in, double* d_out);   // d_out = A * d_in

  bool ready_ = false;
  int  n_ = 0;
  int  nnz_ = 0;

  cusparseHandle_t sparse_ = nullptr;
  cublasHandle_t   blas_   = nullptr;

  int*    d_rowptr_ = nullptr;
  int*    d_colind_ = nullptr;
  double* d_val_    = nullptr;

  // Work vectors: residual, preconditioned residual, search direction, A*p.
  double* d_r_  = nullptr;
  double* d_z_  = nullptr;
  double* d_p_  = nullptr;
  double* d_Ap_ = nullptr;

  cusparseSpMatDescr_t mat_A_   = nullptr;
  cusparseDnVecDescr_t vec_in_  = nullptr;
  cusparseDnVecDescr_t vec_out_ = nullptr;
  void* buf_ = nullptr;
};

}  // namespace bp5
