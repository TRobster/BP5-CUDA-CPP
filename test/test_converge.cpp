// MMS convergence sweep for the 1-D SBP second derivative.
//
//   ./test_converge [--mu M] [--margin M] [N...]
//
//   --mu M      modulus in mu*u'' + f = 0. Default 1. The rate does not depend
//               on it; it is here to show the pair is exact for any mu.
//   --margin M  node layers dropped at each end before scoring. Default 1,
//               which is exactly the one-sided closure rows for p = 2.
//   N...        grid sizes to sweep. Default 8 16 32 64 128.
//
// Prints  r = mu * D2 * u_exact + f,  which is identically zero in the
// continuum. For p = 2 the interior stencil is second order, so both norms
// should fall like h^2 and the rate should settle near 2.
#include "bp5/converge.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace bp5;

int main(int argc, char** argv) {
  Mms s;
  int margin = 1;
  std::vector<int> sizes;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--mu") && i + 1 < argc) s.mu = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--margin") && i + 1 < argc) margin = std::atoi(argv[++i]);
    else sizes.push_back(std::atoi(argv[i]));
  }
  if (sizes.empty()) sizes = {8, 16, 32, 64, 128};

  const int p = 2;
  std::printf("\n== manufactured solution, x in [-1,1] ==\n");
  std::printf("  PDE       mu u'' + f = 0\n");
  std::printf("  f(x)      cos(x) + x sin(x)\n");
  std::printf("  u(x)      ( 3 cos(x) + x sin(x) ) / mu\n");
  std::printf("  mu        %g\n", s.mu);
  std::printf("  SBP p     %d   (expect rate ~ %d away from the closure)\n", p, p);
  std::printf("  margin    %d node(s) dropped at each end\n", margin);

  std::printf("\n== r = mu D2 u_exact + f ==\n");
  std::printf("  %5s %10s %12s %7s %12s %7s %8s\n", "N", "h", "||r||_H", "rate", "max|r|",
              "rate", "scored");

  double prev_l2 = 0.0, prev_linf = 0.0, last_l2 = 0.0, last_linf = 0.0;
  int nrates = 0;

  for (size_t i = 0; i < sizes.size(); ++i) {
    const Sbp1D d = Sbp1D::make(p, sizes[i]);
    const Residual r = residual(d, s, margin);

    char cl2[16] = "      -", cinf[16] = "      -";
    if (i > 0) {
      const double refine = static_cast<double>(sizes[i]) / static_cast<double>(sizes[i - 1]);
      last_l2   = rate(prev_l2,   r.l2,   refine);
      last_linf = rate(prev_linf, r.linf, refine);
      std::snprintf(cl2,  sizeof cl2,  "%7.2f", last_l2);
      std::snprintf(cinf, sizeof cinf, "%7.2f", last_linf);
      ++nrates;
    }
    std::printf("  %5d %10.4g %12.4e %7s %12.4e %7s %8d\n", sizes[i], r.h, r.l2, cl2,
                r.linf, cinf, r.nscored);

    prev_l2 = r.l2;
    prev_linf = r.linf;
  }

  if (nrates == 0) {
    std::printf("\n  need at least two grid sizes to measure a rate\n\n");
    return 1;
  }

  // ------------------------------------------------- the solvable system --
  // D2 alone is singular (D2*1 = 0, D2*x = 0). Adding the Dirichlet SAT gives
  // HM, which should be symmetric positive definite. Everything below is
  // solver-free: a matvec, a symmetry scan, and a dense eigenvalue
  // decomposition used as a diagnostic, not as a solve.
  std::printf("\n== HM u = H.*f + HB g  (both ends Dirichlet) ==\n");
  std::printf("  %5s %6s %11s %11s %11s %11s %8s\n", "N", "nnz", "symmetry",
              "min eig", "max eig", "cond", "D2*1");
  bool spd = true;
  for (size_t i = 0; i < sizes.size(); ++i) {
    const Sbp1D d = Sbp1D::make(p, sizes[i]);
    const System sys = build(d, s);

    const Eigen::MatrixXd dense(sys.HM);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense, Eigen::EigenvaluesOnly);
    const double lo = es.eigenvalues().minCoeff(), hi = es.eigenvalues().maxCoeff();
    if (lo <= 0.0) spd = false;

    // The null vector D2 used to have, for contrast: HM should not annihilate it.
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(d.n);
    std::printf("  %5d %6d %11.3e %11.3e %11.3e %11.3e %8.1e\n", sizes[i],
                (int)sys.HM.nonZeros(), symmetry_error(sys.HM), lo, hi, hi / lo,
                (d.D2 * ones).cwiseAbs().maxCoeff());
  }

  // Truncation error of the COMPLETE system, SAT included. Still no solver:
  // this is HM applied to the known exact solution, minus the assembled
  // right-hand side. If this is not small, solving is pointless.
  std::printf("\n== tau = HM u_exact - b   (full system, no solve) ==\n");
  std::printf("  %5s %12s %7s %12s %7s\n", "N", "||tau||_H", "rate", "max|tau|", "rate");
  double p2 = 0.0, pinf = 0.0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    const Sbp1D d = Sbp1D::make(p, sizes[i]);
    const System sys = build(d, s);
    const std::vector<double> uev = sample_u(d, s);
    const Eigen::VectorXd ue = Eigen::Map<const Eigen::VectorXd>(uev.data(), d.n);
    const Eigen::VectorXd tau = sys.HM * ue - sys.b;

    // Scaled by 1/H so it is comparable to the first table's units.
    double sum = 0.0, mx = 0.0;
    const std::vector<double> hd = h_diag(d);
    for (int j = 0; j < d.n; ++j) {
      const double v = tau(j) / hd[j];
      sum += hd[j] * v * v;
      mx = std::max(mx, std::fabs(v));
    }
    const double l2 = std::sqrt(sum);

    char c2[16] = "      -", ci[16] = "      -";
    if (i > 0) {
      const double refine = static_cast<double>(sizes[i]) / static_cast<double>(sizes[i - 1]);
      std::snprintf(c2, sizeof c2, "%7.2f", rate(p2, l2, refine));
      std::snprintf(ci, sizeof ci, "%7.2f", rate(pinf, mx, refine));
    }
    std::printf("  %5d %12.4e %7s %12.4e %7s\n", sizes[i], l2, c2, mx, ci);
    p2 = l2;
    pinf = mx;
  }
  std::printf("\n  HM is %s\n", spd ? "symmetric positive definite -- solvable"
                                    : "NOT positive definite -- penalty too weak");

  // ||r||_H is the honest number. max|r| is set by whichever single node sits
  // nearest the closure, so it converges more raggedly.
  const bool ok = std::fabs(last_l2 - p) < 0.25;
  std::printf("\n== verdict ==\n");
  std::printf("  last ||r||_H rate = %.2f, expected ~%d   %s\n", last_l2, p,
              ok ? "OK" : "OFF -- check the sign of f, or widen the margin");
  std::printf("  last max|r|  rate = %.2f\n\n", last_linf);
  return ok ? 0 : 4;
}
