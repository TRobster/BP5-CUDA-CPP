// Feed the real HM operator to the IC(0) preconditioner.
//
// This closes the loop between the two halves of the project that had never
// spoken: the CPU assembly (Eigen) and the GPU preconditioner (cuSPARSE).
// It answers three questions, in order of how much they matter:
//
//   1. Does csric02 factor HM at all, or break down on a zero pivot?
//      IC(0) has no pivoting and is only guaranteed for M-matrices. HM is SPD
//      but NOT an M-matrix -- roughly 30% of its off-diagonals are positive,
//      because the Lame coupling that makes rock bulge sideways under
//      compression puts them there.
//
//   2. If it factors, did it *nearly* break down? A minimum |L[i][i]| that is
//      tiny relative to the maximum means csric02 returned success while
//      producing a badly conditioned preconditioner. Nothing warns you.
//
//   3. How good is it? ||(M^-1 HM - I)x|| / ||x|| is the quantity that governs
//      CG convergence. 0 is perfect, 1 means no better than no preconditioner.
//      For reference the 2-D Laplacian test gave 0.31.
//
// Usage: test_hm_precond <params.dat> [--n N] [--shift ALPHA]
#include "bp5/assemble.hpp"
#include "bp5/conditioner.cuh"
#include "bp5/factor.hpp"
#include "bp5/metrics.hpp"
#include "bp5/params.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace bp5;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <params.dat> [--n N] [--shift ALPHA]\n", argv[0]);
    return 1;
  }
  int noverride = 0;
  double shift = 0.0;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--n") && i + 1 < argc) noverride = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--shift") && i + 1 < argc) shift = std::atof(argv[++i]);
  }

  Params p = Params::read(argv[1]);
  if (noverride > 0) { p.Nx = p.Ny = p.Nz = noverride; p.finalize(); }
  const Metrics m = Metrics::make(p);

  std::printf("\n== assembly ==\n");
  Operators ops = assemble(p, m);
  const Csr lower = to_csr_lower(ops.HM);
  const int n = static_cast<int>(ops.HM.rows());
  std::printf("  grid      %d^3\n", p.Nx);
  std::printf("  n         %d\n", n);
  std::printf("  nnz(HM)   %zu   nnz(lower) %zu\n", ops.nnz(), lower.nnz());

  // ------------------------------------------------------------- factorise --
  std::printf("\n== IC(0) factorisation ==\n");
  std::printf("  shift     %g\n", shift);

  IncompleteCholesky ic;
  if (!ic.setup(n, lower.rowptr.data(), lower.colind.data(), lower.val.data(), shift)) {
    std::printf("  result    BROKE DOWN\n");
    std::printf("\n  IC(0) failed on HM. This is the M-matrix issue: try a\n"
                "  diagonal shift, e.g. --shift %.3g and increase until it factors.\n\n",
                0.01 * ops.HM.coeff(0, 0));
    return 2;
  }
  std::printf("  result    OK\n");

  double lo = 0, hi = 0;
  if (ic.factor_diagonal_range(lo, hi)) {
    std::printf("  |L[i][i]|  min = %.4e   max = %.4e   max/min = %.3e\n", lo, hi,
                lo > 0 ? hi / lo : INFINITY);
    std::printf("            %s\n",
                (lo > 0 && hi / lo < 1e8)
                    ? "healthy spread"
                    : "WARNING: near-breakdown, preconditioner is ill-conditioned");
  }

  // --------------------------------------------------------------- quality --
  // y = M^-1 (HM x). If M were exactly HM, y would come back as x.
  std::printf("\n== preconditioner quality ==\n");

  double *d_b = nullptr, *d_y = nullptr;
  cudaMalloc(&d_b, sizeof(double) * n);
  cudaMalloc(&d_y, sizeof(double) * n);

  std::mt19937 rng(2024);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  double worst = 0.0;

  for (int trial = 0; trial < 3; ++trial) {
    Eigen::VectorXd x(n);
    for (int i = 0; i < n; ++i) x(i) = u(rng);
    const Eigen::VectorXd b = ops.HM * x;

    cudaMemcpy(d_b, b.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    if (!ic.apply(d_b, d_y)) { std::printf("  apply FAILED\n"); return 3; }

    std::vector<double> y(n);
    cudaMemcpy(y.data(), d_y, sizeof(double) * n, cudaMemcpyDeviceToHost);

    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
      num = std::max(num, std::fabs(y[i] - x(i)));
      den = std::max(den, std::fabs(x(i)));
    }
    const double e = num / den;
    worst = std::max(worst, e);
    std::printf("  trial %d   ||(M^-1 HM - I)x|| / ||x|| = %.4f\n", trial, e);
  }

  cudaFree(d_b);
  cudaFree(d_y);

  std::printf("\n  worst     %.4f\n", worst);
  std::printf("  reference 0.00 = perfect, 1.00 = useless; laplace2d IC(0) = 0.31\n");
  std::printf("\n%s\n\n",
              worst < 0.5 ? "IC(0) looks usable for CG."
                          : "IC(0) is weak here -- CG will converge slowly.");
  return 0;
}
