// Solution error of the 1-D SBP-SAT system, solved with CG + IC(0) on the GPU.
//
// This is the rung above test_converge. That one measures TRUNCATION error --
// apply the operator to the known solution, look at what is left over. This one
// measures SOLUTION error: actually solve
//
//     HM u = H.*f + HB g
//
// and compare u to u_exact. The two do not converge at the same rate, which is
// the entire reason to run this. The full system's truncation error converges
// at only ~1.5 in the H norm, because the one-sided SBP closure at the boundary
// is first order. Gustafsson's theorem says the solution error should still be
// O(h^p) -- the inverse operator smooths the boundary defect away. Nothing
// short of a solve can confirm that.
//
// ------------------------------------------------------------- tolerance --
//
// CG stops on ||b - HM u|| / ||b|| < tol, and the error that leaves behind in u
// is bounded by roughly cond(HM) * tol. cond grows like O(1/h^2) -- measured at
// 493 for N = 32 -- while the discretization error being measured is ~1e-4
// there. So tol has to sit many orders below the thing being measured or the
// rate flattens into the solver's noise floor. Default is 1e-14, and the
// achieved residual is printed beside the error so contamination is visible
// rather than silent.
//
// HM is tridiagonal apart from two corner entries, so IC(0) is very nearly an
// exact factorisation of it and CG should converge in a handful of iterations.
// That is a property of this 1-D test problem, not of BP5 in 3-D.
//
// Usage: test_solution_cg [--tol T] [--maxiter K] [--mu M] [N...]
#include "bp5/conditioner.cuh"
#include "bp5/conjugateGR.cuh"
#include "bp5/converge.hpp"
#include "bp5/factor.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace bp5;

namespace {

struct Err {
  double l2   = 0.0;   // sqrt( sum H_ii e_i^2 )
  double linf = 0.0;
  int    iters = 0;
  double cgres = 0.0;
  bool   converged = false;
};

}  // namespace

int main(int argc, char** argv) {
  Mms s;
  double tol = 1e-14;
  int maxiter = 5000;
  std::vector<int> sizes;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--tol") && i + 1 < argc) tol = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--maxiter") && i + 1 < argc) maxiter = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--mu") && i + 1 < argc) s.mu = std::atof(argv[++i]);
    else sizes.push_back(std::atoi(argv[i]));
  }
  if (sizes.empty()) sizes = {4, 8, 16, 32};

  const int p = 2;
  std::printf("\n== solution error,  HM u = H.*f + HB g ==\n");
  std::printf("  PDE       mu u'' + f = 0,  x in [-1,1],  both ends Dirichlet\n");
  std::printf("  f(x)      cos(x) + x sin(x)\n");
  std::printf("  u(x)      ( 3 cos(x) + x sin(x) ) / mu\n");
  std::printf("  mu        %g\n", s.mu);
  std::printf("  solver    CG + IC(0), tol = %.1e, maxiter = %d\n", tol, maxiter);
  std::printf("  expect    rate ~ %d  (Gustafsson, despite tau converging at ~1.5)\n", p);

  std::printf("\n  %5s %12s %7s %12s %7s %6s %10s\n", "N", "||e||_H", "rate", "max|e|",
              "rate", "iters", "CG resid");

  double prev2 = 0.0, previ = 0.0, last2 = 0.0, lasti = 0.0;
  bool all_ok = true;
  double worst_margin = 1e300;

  for (size_t k = 0; k < sizes.size(); ++k) {
    const Sbp1D d = Sbp1D::make(p, sizes[k]);
    const System sys = build(d, s);
    const int n = d.n;

    // SpMV needs both triangles; csric02 documents the lower one.
    const Csr full  = to_csr(sys.HM, /*upper_only=*/false);
    const Csr lower = to_csr_lower(sys.HM);

    double *d_b = nullptr, *d_x = nullptr;
    cudaMalloc(&d_b, sizeof(double) * n);
    cudaMalloc(&d_x, sizeof(double) * n);
    cudaMemcpy(d_b, sys.b.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, sizeof(double) * n);

    ConjugateGradient cg;
    if (!cg.setup(n, full.rowptr.data(), full.colind.data(), full.val.data())) {
      std::printf("  N = %d: CG setup failed\n", sizes[k]);
      return 2;
    }
    IncompleteCholesky ic;
    if (!ic.setup(n, lower.rowptr.data(), lower.colind.data(), lower.val.data())) {
      std::printf("  N = %d: IC(0) setup failed\n", sizes[k]);
      return 3;
    }

    const ConjugateGradient::Result r = cg.solve(d_b, d_x, tol, maxiter, &ic);
    cudaDeviceSynchronize();

    std::vector<double> uh(n);
    cudaMemcpy(uh.data(), d_x, sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaFree(d_b);
    cudaFree(d_x);

    // e = u_h - u_exact, over every node. No margin: solution error is
    // meaningful at the boundary too, which is exactly what is being tested.
    const std::vector<double> ue = sample_u(d, s);
    const std::vector<double> hd = h_diag(d);
    Err e;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      const double v = uh[i] - ue[i];
      sum += hd[i] * v * v;
      e.linf = std::max(e.linf, std::fabs(v));
    }
    e.l2 = std::sqrt(sum);
    e.iters = r.iterations;
    e.cgres = r.relative_residual;
    e.converged = r.converged;
    if (!e.converged) all_ok = false;

    // How far the solver's noise floor sits below the thing being measured.
    // Anything under ~100x and the rate is suspect.
    double unorm = 0.0;
    for (int i = 0; i < n; ++i) unorm = std::max(unorm, std::fabs(ue[i]));
    worst_margin = std::min(worst_margin, e.linf / std::max(e.cgres * unorm, 1e-300));

    char c2[16] = "      -", ci[16] = "      -";
    if (k > 0) {
      const double refine = static_cast<double>(sizes[k]) / static_cast<double>(sizes[k - 1]);
      last2 = rate(prev2, e.l2, refine);
      lasti = rate(previ, e.linf, refine);
      std::snprintf(c2, sizeof c2, "%7.2f", last2);
      std::snprintf(ci, sizeof ci, "%7.2f", lasti);
    }
    std::printf("  %5d %12.4e %7s %12.4e %7s %6d %10.1e%s\n", sizes[k], e.l2, c2, e.linf,
                ci, e.iters, e.cgres, e.converged ? "" : "  MAXITER");
    prev2 = e.l2;
    previ = e.linf;
  }

  std::printf("\n== verdict ==\n");
  if (!all_ok) std::printf("  WARNING: CG hit maxiter; rates below are not trustworthy\n");
  std::printf("  solver noise floor sits %.0e x below the measured error   %s\n", worst_margin,
              worst_margin > 100.0 ? "OK" : "TOO CLOSE -- tighten --tol");

  const bool ok = std::fabs(last2 - p) < 0.25 && all_ok && worst_margin > 100.0;
  std::printf("  last ||e||_H rate = %.2f, expected ~%d   %s\n", last2, p, ok ? "OK" : "OFF");
  std::printf("  last max|e|  rate = %.2f\n\n", lasti);
  return ok ? 0 : 4;
}
