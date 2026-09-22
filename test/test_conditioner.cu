// Exercise IncompleteCholesky on two matrices with known behaviour.
//
//   1. 1-D Laplacian (tridiagonal). Cholesky of a tridiagonal matrix produces
//      no fill-in, so IC(0) is the *exact* factorisation and apply() must
//      recover the solution to machine precision. Unambiguous pass/fail.
//
//   2. 2-D Laplacian (5-point). Here IC(0) genuinely approximates, so the
//      result is only close. This checks it runs on a realistic pattern.
#include "bp5/conditioner.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

struct Csr {
  int n = 0;
  std::vector<int> rowptr, colind;
  std::vector<double> val;
};

// Full symmetric CSR for the 1-D Laplacian, tridiag(-1, 2, -1).
Csr laplace1d(int n) {
  Csr a;
  a.n = n;
  a.rowptr.push_back(0);
  for (int i = 0; i < n; ++i) {
    if (i > 0)     { a.colind.push_back(i - 1); a.val.push_back(-1.0); }
                     a.colind.push_back(i);     a.val.push_back(2.0);
    if (i < n - 1) { a.colind.push_back(i + 1); a.val.push_back(-1.0); }
    a.rowptr.push_back(static_cast<int>(a.colind.size()));
  }
  return a;
}

// Full symmetric CSR for the 2-D 5-point Laplacian on an m x m grid.
Csr laplace2d(int m) {
  Csr a;
  a.n = m * m;
  a.rowptr.push_back(0);
  for (int j = 0; j < m; ++j)
    for (int i = 0; i < m; ++i) {
      const int row = i + m * j;
      if (j > 0)     { a.colind.push_back(row - m); a.val.push_back(-1.0); }
      if (i > 0)     { a.colind.push_back(row - 1); a.val.push_back(-1.0); }
                       a.colind.push_back(row);     a.val.push_back(4.0);
      if (i < m - 1) { a.colind.push_back(row + 1); a.val.push_back(-1.0); }
      if (j < m - 1) { a.colind.push_back(row + m); a.val.push_back(-1.0); }
      a.rowptr.push_back(static_cast<int>(a.colind.size()));
    }
  return a;
}

std::vector<double> spmv(const Csr& a, const std::vector<double>& x) {
  std::vector<double> y(a.n, 0.0);
  for (int i = 0; i < a.n; ++i)
    for (int k = a.rowptr[i]; k < a.rowptr[i + 1]; ++k) y[i] += a.val[k] * x[a.colind[k]];
  return y;
}

double relerr(const std::vector<double>& a, const std::vector<double>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    num = std::max(num, std::fabs(a[i] - b[i]));
    den = std::max(den, std::fabs(b[i]));
  }
  return den > 0 ? num / den : num;
}

// Factor `a`, then apply the preconditioner to b = A*x_exact and see how close
// the result comes back to x_exact.
bool run(const char* label, const Csr& a, double tol, bool must_be_exact) {
  bp5::IncompleteCholesky ic;
  if (!ic.setup(a.n, a.rowptr.data(), a.colind.data(), a.val.data())) {
    std::printf("  %-16s setup FAILED\n", label);
    return false;
  }

  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> xe(a.n);
  for (auto& v : xe) v = u(rng);
  const std::vector<double> b = spmv(a, xe);

  double *d_b = nullptr, *d_y = nullptr;
  cudaMalloc(&d_b, sizeof(double) * a.n);
  cudaMalloc(&d_y, sizeof(double) * a.n);
  cudaMemcpy(d_b, b.data(), sizeof(double) * a.n, cudaMemcpyHostToDevice);

  const bool applied = ic.apply(d_b, d_y);

  std::vector<double> y(a.n);
  cudaMemcpy(y.data(), d_y, sizeof(double) * a.n, cudaMemcpyDeviceToHost);
  cudaFree(d_b);
  cudaFree(d_y);

  if (!applied) {
    std::printf("  %-16s apply FAILED\n", label);
    return false;
  }

  const double e = relerr(y, xe);
  const bool pass = must_be_exact ? (e < tol) : std::isfinite(e);
  std::printf("  %-16s n=%-6d nnz=%-7zu  max|y-x|/max|x| = %.3e   %s\n", label, a.n,
              a.val.size(), e, pass ? "OK" : "FAIL");
  if (!must_be_exact)
    std::printf("  %-16s (IC(0) is approximate here, so this is expected to be loose)\n", "");
  return pass;
}

}  // namespace

int main() {
  std::printf("\n== IncompleteCholesky ==\n");
  bool ok = true;
  // Tridiagonal: no fill-in, so IC(0) == exact Cholesky. Must be exact.
  ok &= run("laplace1d", laplace1d(64), 1e-12, true);
  // 5-point stencil: IC(0) drops real fill. Only checked for sanity.
  ok &= run("laplace2d", laplace2d(16), 0.0, false);
  std::printf("\n%s\n\n", ok ? "all checks passed" : "FAILURES");
  return ok ? 0 : 1;
}
