#include "bp5/conditioner.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

namespace bp5 {
namespace {

bool ok(cudaError_t e, const char* what, int line) {
  if (e == cudaSuccess) return true;
  std::fprintf(stderr, "conditioner.cu:%d  %s -> %s\n", line, what, cudaGetErrorString(e));
  return false;
}

bool ok(cusparseStatus_t s, const char* what, int line) {
  if (s == CUSPARSE_STATUS_SUCCESS) return true;
  std::fprintf(stderr, "conditioner.cu:%d  %s -> %s\n", line, what, cusparseGetErrorString(s));
  return false;
}

// Bail out of the enclosing bool-returning function on any failure.
#define TRY(x) do { if (!ok((x), #x, __LINE__)) return false; } while (0)

// val[i][i] += shift, applied in CSR without touching anything else.
__global__ void add_diag_shift(int n, const int* rowptr, const int* colind,
                               double* val, double shift) 
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (int k = rowptr[i]; k < rowptr[i + 1]; ++k)
    if (colind[k] == i) { val[k] += shift; return; }
}

// Gather L[i][i] into a dense vector so the host can inspect the pivot range.
__global__ void gather_diag(int n, const int* rowptr, const int* colind,
                            const double* val, double* out) 
  {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double d = 0.0;
  for (int k = rowptr[i]; k < rowptr[i + 1]; ++k)
    if (colind[k] == i) { d = val[k]; break; }
  out[i] = d;
  }

}  // namespace

IncompleteCholesky::~IncompleteCholesky() { release(); }

void IncompleteCholesky::release() {
  // garbage destroyer function, deconstructor 
  if (spsv_L_)  cusparseSpSV_destroyDescr(spsv_L_);
  if (spsv_Lt_) cusparseSpSV_destroyDescr(spsv_Lt_);
  if (mat_L_)   cusparseDestroySpMat(mat_L_);
  if (vec_x_)   cusparseDestroyDnVec(vec_x_);
  if (vec_y_)   cusparseDestroyDnVec(vec_y_);
  if (vec_z_)   cusparseDestroyDnVec(vec_z_);
  if (handle_)  cusparseDestroy(handle_);

  cudaFree(buf_L_);
  cudaFree(buf_Lt_);
  cudaFree(d_rowptr_);
  cudaFree(d_colind_);
  cudaFree(d_val_);
  cudaFree(d_z_);

  spsv_L_ = spsv_Lt_ = nullptr;
  mat_L_ = nullptr;
  vec_x_ = vec_y_ = vec_z_ = nullptr;
  handle_ = nullptr;
  buf_L_ = buf_Lt_ = nullptr;
  d_rowptr_ = d_colind_ = nullptr;
  d_val_ = d_z_ = nullptr;
  ready_ = false;
  n_ = nnz_ = 0;
}

bool IncompleteCholesky::setup(int n, const int* rowptr, const int* colind,
                               const double* val, double shift) {
  release();
  n_ = n;
  nnz_ = rowptr[n];

  TRY(cusparseCreate(&handle_));

  //  upload 
  TRY(cudaMalloc(&d_rowptr_, sizeof(int) * (n_ + 1)));
  TRY(cudaMalloc(&d_colind_, sizeof(int) * nnz_));
  TRY(cudaMalloc(&d_val_,    sizeof(double) * nnz_));
  TRY(cudaMalloc(&d_z_,      sizeof(double) * n_));
  TRY(cudaMemcpy(d_rowptr_, rowptr, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
  TRY(cudaMemcpy(d_colind_, colind, sizeof(int) * nnz_, cudaMemcpyHostToDevice));
  TRY(cudaMemcpy(d_val_,    val,    sizeof(double) * nnz_, cudaMemcpyHostToDevice));

  if (shift != 0.0) {
    const int block = 256, grid = (n_ + block - 1) / block;
    add_diag_shift<<<grid, block>>>(n_, d_rowptr_, d_colind_, d_val_, shift);
    TRY(cudaGetLastError());
  }

  // factor
  // csric02 works in place: d_val_ goes in holding A and comes out holding L
  // in the lower-triangle entries. The upper triangle is left as it was and is
  // never read again, because every solve below declares FILL_MODE_LOWER.
  //
  // These csric02 entry points are marked CUSPARSE_DEPRECATED as of CUDA 12,
  // but they are still present and are the only incomplete-factorisation
  // routines cuSPARSE offers -- the generic API has no replacement.
  {
    cusparseMatDescr_t descr = nullptr;
    csric02Info_t info = nullptr;
    void* buf = nullptr;
    int bufsize = 0;
    int pivot = -1;

    TRY(cusparseCreateMatDescr(&descr));
    TRY(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    TRY(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    TRY(cusparseCreateCsric02Info(&info));

    TRY(cusparseDcsric02_bufferSize(handle_, n_, nnz_, descr, d_val_, d_rowptr_,
                                    d_colind_, info, &bufsize));
    TRY(cudaMalloc(&buf, bufsize));

    TRY(cusparseDcsric02_analysis(handle_, n_, nnz_, descr, d_val_, d_rowptr_,
                                  d_colind_, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                  buf));
    if (cusparseXcsric02_zeroPivot(handle_, info, &pivot) == CUSPARSE_STATUS_ZERO_PIVOT) {
      std::fprintf(stderr, "IncompleteCholesky: structural zero at row %d\n", pivot);
      cudaFree(buf);
      return false;
    }

    TRY(cusparseDcsric02(handle_, n_, nnz_, descr, d_val_, d_rowptr_, d_colind_,
                         info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buf));
    if (cusparseXcsric02_zeroPivot(handle_, info, &pivot) == CUSPARSE_STATUS_ZERO_PIVOT) {
      std::fprintf(stderr,
                   "IncompleteCholesky: numerical zero pivot at row %d "
                   "(matrix is not positive definite, or IC(0) broke down)\n",
                   pivot);
      cudaFree(buf);
      return false;
    }

    cudaFree(buf);
    TRY(cusparseDestroyCsric02Info(info));
    TRY(cusparseDestroyMatDescr(descr));
  }

  // triangular solve
  // One CSR descriptor serves both solves: L is the lower triangle, and L-trans is
  // the same storage read with OPERATION_TRANSPOSE. No transposed copy needed.
  TRY(cusparseCreateCsr(&mat_L_, n_, n_, nnz_, d_rowptr_, d_colind_, d_val_,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

  cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
  cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
  TRY(cusparseSpMatSetAttribute(mat_L_, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill)));
  TRY(cusparseSpMatSetAttribute(mat_L_, CUSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));

  // The vector descriptors need a valid device pointer at creation time so the
  // analysis pass can run. apply() re-points them at the caller's buffers.
  TRY(cusparseCreateDnVec(&vec_x_, n_, d_z_, CUDA_R_64F));
  TRY(cusparseCreateDnVec(&vec_y_, n_, d_z_, CUDA_R_64F));
  TRY(cusparseCreateDnVec(&vec_z_, n_, d_z_, CUDA_R_64F));

  const double one = 1.0;
  size_t bufL = 0, bufLt = 0;

  TRY(cusparseSpSV_createDescr(&spsv_L_));
  TRY(cusparseSpSV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_L_,
                              vec_x_, vec_z_, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                              spsv_L_, &bufL));
  TRY(cudaMalloc(&buf_L_, bufL));
  TRY(cusparseSpSV_analysis(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_L_,
                            vec_x_, vec_z_, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                            spsv_L_, buf_L_));

  TRY(cusparseSpSV_createDescr(&spsv_Lt_));
  TRY(cusparseSpSV_bufferSize(handle_, CUSPARSE_OPERATION_TRANSPOSE, &one, mat_L_,
                              vec_z_, vec_y_, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                              spsv_Lt_, &bufLt));
  TRY(cudaMalloc(&buf_Lt_, bufLt));
  TRY(cusparseSpSV_analysis(handle_, CUSPARSE_OPERATION_TRANSPOSE, &one, mat_L_,
                            vec_z_, vec_y_, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                            spsv_Lt_, buf_Lt_));

  ready_ = true;
  return true;
}

bool IncompleteCholesky::factor_diagonal_range(double& lo, double& hi) const {
  if (!ready_) return false;

  double* d_diag = nullptr;
  TRY(cudaMalloc(&d_diag, sizeof(double) * n_));
  const int block = 256, grid = (n_ + block - 1) / block;
  gather_diag<<<grid, block>>>(n_, d_rowptr_, d_colind_, d_val_, d_diag);
  TRY(cudaGetLastError());

  std::vector<double> h(n_);
  TRY(cudaMemcpy(h.data(), d_diag, sizeof(double) * n_, cudaMemcpyDeviceToHost));
  cudaFree(d_diag);

  lo = hi = std::fabs(h[0]);
  for (int i = 1; i < n_; ++i) {
    const double v = std::fabs(h[i]);
    lo = v < lo ? v : lo;
    hi = v > hi ? v : hi;
  }
  return true;
}

bool IncompleteCholesky::apply(const double* d_x, double* d_y) {
  if (!ready_) {
    std::fprintf(stderr, "IncompleteCholesky::apply called before setup\n");
    return false;
  }

  // cuSPARSE will only read from x, so casting away const is safe here.
  TRY(cusparseDnVecSetValues(vec_x_, const_cast<double*>(d_x)));
  TRY(cusparseDnVecSetValues(vec_y_, d_y));
  TRY(cusparseDnVecSetValues(vec_z_, d_z_));

  const double one = 1.0;

  // L z = x
  TRY(cusparseSpSV_solve(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_L_,
                         vec_x_, vec_z_, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                         spsv_L_));
  // L^T y = z
  TRY(cusparseSpSV_solve(handle_, CUSPARSE_OPERATION_TRANSPOSE, &one, mat_L_,
                         vec_z_, vec_y_, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                         spsv_Lt_));

  TRY(cudaDeviceSynchronize());
  return true;
}

}  // namespace bp5
