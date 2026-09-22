// CG on the real HM operator, with and without the IC(0) preconditioner.
//
// The question this answers: does IC(0) pay for itself? It reduces the
// iteration count, but each iteration now costs two extra sparse triangular
// solves, and triangular solves are the least GPU-friendly thing in the loop
// (bandwidth-bound with sequential dependencies). Iteration count alone can
// flatter a preconditioner that loses on wall clock -- so both are reported.
//
// Usage: test_cg <params.dat> [--n N] [--tol T] [--maxiter K]
#include "bp5/assemble.hpp"
#include "bp5/conditioner.cuh"
#include "bp5/conjugateGR.cuh"
#include "bp5/factor.hpp"
#include "bp5/metrics.hpp"
#include "bp5/params.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace bp5;

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Solve once and report. Returns wall time.
double run(const char* label, ConjugateGradient& cg, IncompleteCholesky* pre,
           const double* d_b, double* d_x, int n, const std::vector<double>& xe,
           double tol, int maxiter) {
  cudaMemset(d_x, 0, sizeof(double) * n);
  cudaDeviceSynchronize();

  const auto t0 = std::chrono::steady_clock::now();
  const ConjugateGradient::Result r = cg.solve(d_b, d_x, tol, maxiter, pre);
  cudaDeviceSynchronize();
  const double dt = seconds_since(t0);

  std::vector<double> x(n);
  cudaMemcpy(x.data(), d_x, sizeof(double) * n, cudaMemcpyDeviceToHost);
  double num = 0.0, den = 0.0;
  for (int i = 0; i < n; ++i) {
    num = std::max(num, std::fabs(x[i] - xe[i]));
    den = std::max(den, std::fabs(xe[i]));
  }

  std::printf("  %-14s %s  iters = %-5d  ||r||/||b|| = %.2e  err = %.2e  %.3f s\n",
              label, r.converged ? "converged" : "MAXITER  ", r.iterations,
              r.relative_residual, num / den, dt);
  return dt;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <params.dat> [--n N] [--tol T] [--maxiter K]\n", argv[0]);
    return 1;
  }
  int noverride = 0, maxiter = 5000;
  double tol = 1e-10;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--n") && i + 1 < argc) noverride = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--tol") && i + 1 < argc) tol = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--maxiter") && i + 1 < argc) maxiter = std::atoi(argv[++i]);
  }

  Params p = Params::read(argv[1]);
  if (noverride > 0) { p.Nx = p.Ny = p.Nz = noverride; p.finalize(); }
  const Metrics m = Metrics::make(p);

  Operators ops = assemble(p, m);
  const int n = static_cast<int>(ops.HM.rows());
  const Csr full  = to_csr(ops.HM, /*upper_only=*/false);   // SpMV needs both triangles
  const Csr lower = to_csr_lower(ops.HM);                   // csric02 wants the lower one

  std::printf("\n== CG on HM ==\n");
  std::printf("  grid %d^3   n = %d   nnz = %zu   tol = %.1e\n", p.Nx, n, full.nnz(), tol);

  // Manufactured system so the answer is known independently of the residual.
  std::mt19937 rng(99);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Eigen::VectorXd xe(n);
  for (int i = 0; i < n; ++i) xe(i) = u(rng);
  const Eigen::VectorXd b = ops.HM * xe;
  const std::vector<double> xe_h(xe.data(), xe.data() + n);

  double *d_b = nullptr, *d_x = nullptr;
  cudaMalloc(&d_b, sizeof(double) * n);
  cudaMalloc(&d_x, sizeof(double) * n);
  cudaMemcpy(d_b, b.data(), sizeof(double) * n, cudaMemcpyHostToDevice);

  ConjugateGradient cg;
  if (!cg.setup(n, full.rowptr.data(), full.colind.data(), full.val.data())) return 2;

  const double t_plain = run("plain CG", cg, nullptr, d_b, d_x, n, xe_h, tol, maxiter);

  IncompleteCholesky ic;
  if (!ic.setup(n, lower.rowptr.data(), lower.colind.data(), lower.val.data())) {
    std::printf("  IC(0) setup failed\n");
    return 3;
  }
  const double t_pre = run("CG + IC(0)", cg, &ic, d_b, d_x, n, xe_h, tol, maxiter);

  std::printf("\n  wall-clock speedup from preconditioning: %.2fx\n",
              t_pre > 0 ? t_plain / t_pre : 0.0);

  cudaFree(d_b);
  cudaFree(d_x);
  std::printf("\n");
  return 0;
}
