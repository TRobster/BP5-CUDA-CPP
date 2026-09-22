// bp5 CUDA -- Tier 1, step 1.
//
// Assemble the single-block BP5 elasticity operator HM, check it, factor it,
// and solve once. Everything after this (boundary data, traction extraction,
// rate-and-state friction, time stepping) hangs off the same HM.
//
// Usage:  bp5_assemble <params.dat> [--n N] [--backend eigen|cudss] [--dump DIR]
//
//   --n N        override Nx = Ny = Nz = N (for quick runs; the .dat is 128)
//   --backend    which factorisation to use, default cudss if built in
//   --dump DIR   write HM to DIR as CSR for cross-checking against Julia
#include "bp5/assemble.hpp"
#include "bp5/factor.hpp"
#include "bp5/metrics.hpp"
#include "bp5/params.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>

using namespace bp5;

namespace {

struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double lap() {
    const auto t1 = std::chrono::steady_clock::now();
    const double s = std::chrono::duration<double>(t1 - t0).count();
    t0 = t1;
    return s;
  }
};

void dump_csr(const Csr& c, const std::string& dir) {
  auto w = [&](const std::string& name, const void* p, size_t bytes) {
    std::ofstream f(dir + "/" + name, std::ios::binary);
    f.write(static_cast<const char*>(p), static_cast<std::streamsize>(bytes));
  };
  const int64_t nnz = static_cast<int64_t>(c.nnz());
  w("HM.n", &c.n, sizeof(int));
  w("HM.nnz", &nnz, sizeof(int64_t));
  w("HM.rowptr", c.rowptr.data(), sizeof(int) * c.rowptr.size());
  w("HM.colind", c.colind.data(), sizeof(int) * c.colind.size());
  w("HM.val", c.val.data(), sizeof(double) * c.val.size());
  std::printf("  wrote HM to %s (n = %d, nnz = %lld)\n", dir.c_str(), c.n,
              static_cast<long long>(nnz));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <params.dat> [--n N] [--backend eigen|cudss] [--dump DIR]"
                " [--assemble-only]\n",
                argv[0]);
    return 1;
  }

  std::string backend = "cudss";
  std::string dump;
  int noverride = 0;
  bool assemble_only = false;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--n") && i + 1 < argc) noverride = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
    else if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
    else if (!std::strcmp(argv[i], "--assemble-only")) assemble_only = true;
  }

  Params p;
  try {
    p = Params::read(argv[1]);
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
  if (noverride > 0) {
    p.Nx = p.Ny = p.Nz = noverride;
    p.finalize();
  }

  std::printf("\n== parameters ==\n");
  p.print();

  const Metrics m = Metrics::make(p);
  std::printf("\n== metrics ==\n");
  m.print();

  const Grid probe{p.Nx + 1, p.Ny + 1, p.Nz + 1};
  std::printf("\n== problem size ==\n");
  std::printf("  nodes     %d  (%d per component)\n", probe.ndof(), probe.np());
  std::printf("  HM        ~%.2f GB estimated\n", estimate_bytes(probe) / 1e9);

  Timer t;
  std::printf("\n== assembly ==\n");
  Operators ops;
  try {
    ops = assemble(p, m);
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
  std::printf("  built HM in %.2f s   n = %d   nnz = %zu   (%.1f per row)\n", t.lap(),
              static_cast<int>(ops.HM.rows()), ops.nnz(),
              static_cast<double>(ops.nnz()) / static_cast<double>(ops.HM.rows()));

  // Symmetry is the cheapest check that the SAT signs and the transposed
  // component indices are right. A correct HM is symmetric to roundoff.
  std::printf("\n== checks ==\n");
  const double sym = symmetry_error(ops.HM);
  std::printf("  symmetry  max|HM - HM^T| / max|HM| = %.3e  %s\n", sym,
              sym < 1e-12 ? "OK" : "FAIL");

  double dmin = 1e300, dmax = -1e300;
  for (int k = 0; k < ops.HM.outerSize(); ++k)
    for (Sp::InnerIterator it(ops.HM, k); it; ++it)
      if (it.row() == it.col()) {
        dmin = std::min(dmin, it.value());
        dmax = std::max(dmax, it.value());
      }
  std::printf("  diagonal  [%.6g, %.6g]  %s\n", dmin, dmax,
              dmin > 0 ? "OK (all positive)" : "FAIL (expect all positive for SPD)");

  // Consistency of the volume operator. Symmetry constrains the SAT terms but
  // says little about A itself, so test A directly against displacement fields
  // whose stress divergence is identically zero:
  //
  //   rigid translation  u = e_c            => A u = 0 everywhere (row sums)
  //   linear field       u = x * e_c        => A u = 0 on interior nodes
  //
  // The linear field is only checked away from the boundary because the
  // one-sided closures of D1/D2 are exact there but the SAT is not part of A.
  {
    const Grid& g = ops.grid;
    double econst = 0.0, elin = 0.0, scale = 0.0;
    for (int c = 0; c < 3; ++c) {
      Eigen::VectorXd ones = Eigen::VectorXd::Ones(g.np());
      Eigen::VectorXd lin(g.np());
      for (int s = 0; s < g.nsp; ++s)
        for (int r = 0; r < g.nrp; ++r)
          for (int q = 0; q < g.nqp; ++q) {
            const int idx[3] = {q, r, s};
            lin(g.vidx(q, r, s)) = m.coord(static_cast<Dir>(c), idx[c],
                                           c == 0 ? g.nqp : c == 1 ? g.nrp : g.nsp);
          }
      for (int i = 0; i < 3; ++i) {
        scale = std::max(scale, Eigen::VectorXd(ops.A[i][c] * lin).cwiseAbs().maxCoeff());
        econst = std::max(econst,
                          Eigen::VectorXd(ops.A[i][c] * ones).cwiseAbs().maxCoeff());
        const Eigen::VectorXd rl = ops.A[i][c] * lin;
        for (int s = 1; s < g.nsp - 1; ++s)
          for (int r = 1; r < g.nrp - 1; ++r)
            for (int q = 1; q < g.nqp - 1; ++q)
              elin = std::max(elin, std::fabs(rl(g.vidx(q, r, s))));
      }
    }
    const double ref = std::max(scale, 1.0);
    std::printf("  A*const   max|A u| = %.3e  %s\n", econst,
                econst / ref < 1e-10 ? "OK" : "FAIL");
    std::printf("  A*linear  max|A u| interior = %.3e  %s\n", elin,
                elin / ref < 1e-10 ? "OK" : "FAIL");
  }

  if (assemble_only) {
    if (!dump.empty()) { std::printf("\n== dump ==\n"); dump_csr(to_csr(ops.HM, false), dump); }
    std::printf("\n");
    return (sym < 1e-12 && dmin > 0) ? 0 : 4;
  }

  // ------------------------------------------------------------ factorise --
  std::printf("\n== factorisation ==\n");
  std::unique_ptr<Factorization> f;
  if (backend == "cudss") {
    f = make_cudss();
    if (!f) {
      std::printf("  cuDSS not available in this build, falling back to Eigen\n");
      f = make_eigen_llt();
    }
  } else {
    f = make_eigen_llt();
  }
  std::printf("  backend   %s\n", f->name().c_str());

  t.lap();
  if (!f->factor(ops.HM)) {
    std::printf("  factorisation FAILED\n");
    return 2;
  }
  std::printf("  factored in %.2f s\n", t.lap());

  // ---------------------------------------------------------------- solve --
  // Manufactured right-hand side: b = HM * x_exact for a known x_exact, so the
  // solve can be scored without the boundary-data operator HB (next step).
  const int n = static_cast<int>(ops.HM.rows());
  std::vector<double> xe(n), b(n), x;
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int i = 0; i < n; ++i) xe[i] = u(rng);
  {
    Eigen::Map<const Eigen::VectorXd> xv(xe.data(), n);
    const Eigen::VectorXd bv = ops.HM * xv;
    std::copy(bv.data(), bv.data() + n, b.begin());
  }

  t.lap();
  if (!f->solve(b, x)) {
    std::printf("  solve FAILED\n");
    return 3;
  }
  const double dt = t.lap();

  double err = 0, nrm = 0;
  for (int i = 0; i < n; ++i) {
    err = std::max(err, std::fabs(x[i] - xe[i]));
    nrm = std::max(nrm, std::fabs(xe[i]));
  }
  std::printf("\n== solve ==\n");
  std::printf("  time      %.4f s\n", dt);
  std::printf("  residual  ||HM x - b|| / ||b|| = %.3e\n", residual(ops.HM, x, b));
  std::printf("  error     max|x - x_exact| / max|x_exact| = %.3e\n", err / nrm);

  if (!dump.empty()) {
    std::printf("\n== dump ==\n");
    dump_csr(to_csr(ops.HM, /*upper_only=*/false), dump);
  }

  std::printf("\n");
  return (sym < 1e-12 && dmin > 0) ? 0 : 4;
}
