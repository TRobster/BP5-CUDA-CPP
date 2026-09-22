#include "bp5/conjugateGR.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace bp5 {
namespace {

bool ok(cudaError_t e, const char* what, int line) {
  if (e == cudaSuccess) return true;
  std::fprintf(stderr, "conjugateGR.cu:%d  %s -> %s\n", line, what, cudaGetErrorString(e));
  return false;
}

bool ok(cusparseStatus_t s, const char* what, int line) {
  if (s == CUSPARSE_STATUS_SUCCESS) return true;
  std::fprintf(stderr, "conjugateGR.cu:%d  %s -> %s\n", line, what, cusparseGetErrorString(s));
  return false;
}

bool ok(cublasStatus_t s, const char* what, int line) {
  if (s == CUBLAS_STATUS_SUCCESS) return true;
  std::fprintf(stderr, "conjugateGR.cu:%d  %s -> cublas status %d\n", line, what, (int)s);
  return false;
}

#define TRY(x) do { if (!ok((x), #x, __LINE__)) return false; } while (0)

}  // namespace

ConjugateGradient::~ConjugateGradient() { release(); }

void ConjugateGradient::release() {
  if (mat_A_)   cusparseDestroySpMat(mat_A_);
  if (vec_in_)  cusparseDestroyDnVec(vec_in_);
  if (vec_out_) cusparseDestroyDnVec(vec_out_);
  if (sparse_)  cusparseDestroy(sparse_);
  if (blas_)    cublasDestroy(blas_);

  cudaFree(buf_);
  cudaFree(d_rowptr_);
  cudaFree(d_colind_);
  cudaFree(d_val_);
  cudaFree(d_r_);
  cudaFree(d_z_);
  cudaFree(d_p_);
  cudaFree(d_Ap_);

  mat_A_ = nullptr;
  vec_in_ = vec_out_ = nullptr;
  sparse_ = nullptr;
  blas_ = nullptr;
  buf_ = nullptr;
  d_rowptr_ = d_colind_ = nullptr;
  d_val_ = d_r_ = d_z_ = d_p_ = d_Ap_ = nullptr;
  ready_ = false;
  n_ = nnz_ = 0;
}

bool ConjugateGradient::setup(int n, const int* rowptr, const int* colind,
                              const double* val) {
  release();
  n_ = n;
  nnz_ = rowptr[n];

  TRY(cusparseCreate(&sparse_));
  TRY(cublasCreate(&blas_));

  TRY(cudaMalloc(&d_rowptr_, sizeof(int) * (n_ + 1)));
  TRY(cudaMalloc(&d_colind_, sizeof(int) * nnz_));
  TRY(cudaMalloc(&d_val_,    sizeof(double) * nnz_));
  TRY(cudaMemcpy(d_rowptr_, rowptr, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
  TRY(cudaMemcpy(d_colind_, colind, sizeof(int) * nnz_, cudaMemcpyHostToDevice));
  TRY(cudaMemcpy(d_val_,    val,    sizeof(double) * nnz_, cudaMemcpyHostToDevice));

  TRY(cudaMalloc(&d_r_,  sizeof(double) * n_));
  TRY(cudaMalloc(&d_z_,  sizeof(double) * n_));
  TRY(cudaMalloc(&d_p_,  sizeof(double) * n_));
  TRY(cudaMalloc(&d_Ap_, sizeof(double) * n_));

  TRY(cusparseCreateCsr(&mat_A_, n_, n_, nnz_, d_rowptr_, d_colind_, d_val_,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  TRY(cusparseCreateDnVec(&vec_in_,  n_, d_p_,  CUDA_R_64F));
  TRY(cusparseCreateDnVec(&vec_out_, n_, d_Ap_, CUDA_R_64F));

  const double one = 1.0, zero = 0.0;
  size_t bufsize = 0;
  TRY(cusparseSpMV_bufferSize(sparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_A_,
                              vec_in_, &zero, vec_out_, CUDA_R_64F,
                              CUSPARSE_SPMV_ALG_DEFAULT, &bufsize));
  TRY(cudaMalloc(&buf_, bufsize));

  ready_ = true;
  return true;
}

bool ConjugateGradient::spmv(const double* d_in, double* d_out) {
  const double one = 1.0, zero = 0.0;
  // cuSPARSE only reads the input vector, so casting away const is safe.
  TRY(cusparseDnVecSetValues(vec_in_,  const_cast<double*>(d_in)));
  TRY(cusparseDnVecSetValues(vec_out_, d_out));
  TRY(cusparseSpMV(sparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_A_, vec_in_,
                   &zero, vec_out_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf_));
  return true;
}

ConjugateGradient::Result ConjugateGradient::solve(const double* d_b, double* d_x,
                                                   double tol, int maxiter,
                                                   IncompleteCholesky* precond) {
  Result out;
  if (!ready_) {
    std::fprintf(stderr, "ConjugateGradient::solve called before setup\n");
    return out;
  }

  // cuBLAS defaults to CUBLAS_POINTER_MODE_HOST, so every dot product below
  // synchronises. If the sync cost
  // ever matters, switch to device pointer mode and keep the scalars resident.
  const double one = 1.0, minus_one = -1.0;
  double bnorm = 0.0;
  cublasDnrm2(blas_, n_, d_b, 1, &bnorm);
  if (bnorm == 0.0) 
  {  // b = 0, so x = 0
    cudaMemset(d_x, 0, sizeof(double) * n_);
    out.converged = true;
    return out;
  }

  // r = b - A x
  if (!spmv(d_x, d_r_)) return out;
  cublasDscal(blas_, n_, &minus_one, d_r_, 1);
  cublasDaxpy(blas_, n_, &one, d_b, 1, d_r_, 1);

  double rnorm = 0.0;
  cublasDnrm2(blas_, n_, d_r_, 1, &rnorm);
  out.relative_residual = rnorm / bnorm;
  if (out.relative_residual < tol) {        // initial guess already good enough
    out.converged = true;
    return out;
  }

  // z = M^-1 r   (or z = r when unpreconditioned)
  // also same as double solve for Lz =x -> L^Ty = z where y becomes y = (LL^T)^-1 * r where y = M^-1
  if (precond) {
    if (!precond->apply(d_r_, d_z_)) return out;
  } else {
    cudaMemcpy(d_z_, d_r_, sizeof(double) * n_, cudaMemcpyDeviceToDevice);
  }

  // p = z
  cudaMemcpy(d_p_, d_z_, sizeof(double) * n_, cudaMemcpyDeviceToDevice);

  double rz = 0.0;
  cublasDdot(blas_, n_, d_r_, 1, d_z_, 1, &rz);

  for (int k = 1; k <= maxiter; ++k) {
    if (!spmv(d_p_, d_Ap_)) return out;

    double pAp = 0.0;
    // p(k)^T * A * p(k)
    cublasDdot(blas_, n_, d_p_, 1, d_Ap_, 1, &pAp);
    if (!(pAp > 0.0)) {
      std::fprintf(stderr,
                   "ConjugateGradient: p'Ap = %g at iteration %d -- the matrix is "
                   "not positive definite\n", pAp, k);
      out.iterations = k;
      return out;
    }

    const double alpha = rz / pAp;
    const double neg_alpha = -alpha;
    cublasDaxpy(blas_, n_, &alpha, d_p_, 1, d_x, 1);        // x += alpha p
    cublasDaxpy(blas_, n_, &neg_alpha, d_Ap_, 1, d_r_, 1);  // r -= alpha A p

    cublasDnrm2(blas_, n_, d_r_, 1, &rnorm);
    out.iterations = k;
    out.relative_residual = rnorm / bnorm;
    if (out.relative_residual < tol) {
      out.converged = true;
      return out;
    }
    // where z = M^1 * r, M^-1 then is M^-1 = I or M = I so z = r 
    if (precond) {
      if (!precond->apply(d_r_, d_z_)) return out;
    } else {
      cudaMemcpy(d_z_, d_r_, sizeof(double) * n_, cudaMemcpyDeviceToDevice);
    }

    double rz_new = 0.0;
    cublasDdot(blas_, n_, d_r_, 1, d_z_, 1, &rz_new);

    // p = z + beta p
    const double beta = rz_new / rz;
    cublasDscal(blas_, n_, &beta, d_p_, 1);
    cublasDaxpy(blas_, n_, &one, d_z_, 1, d_p_, 1);
    rz = rz_new;
  }

  return out;   // hit maxiter without converging
}

}  // namespace bp5
