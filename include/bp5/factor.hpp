#pragma once
//
// Sparse Cholesky factorisation, behind one interface with two backends.
//
//   eigen  SimplicialLLT on the host. Slow and memory hungry, but it runs
//          anywhere, so it is what validates the assembly.
//   cudss  NVIDIA cuDSS on the device. The real target: BP5-QD's operator is
//          constant in time, so this factorisation is paid once and every one
//          of the ~1e5 ODE right-hand sides afterwards is two triangular
//          solves instead of a full CG.
//
#include "bp5/sparse.hpp"

#include <memory>
#include <string>
#include <vector>

namespace bp5 {

// Compressed sparse row, zero-based. What both backends and the on-disk
// format consume.
struct Csr {
  int n = 0;
  std::vector<int> rowptr;    // size n+1
  std::vector<int> colind;
  std::vector<double> val;
  size_t nnz() const { return val.size(); }
};

// Extract CSR. `upper_only` keeps just the upper triangle including the
// diagonal, which is what cuDSS wants for a matrix declared SPD.
Csr to_csr(const Sp& a, bool upper_only);

// Lower triangle including the diagonal. This is the form cusparse's csric02
// documents as its input, so it is what the IC(0) preconditioner is fed.
Csr to_csr_lower(const Sp& a);

class Factorization {
 public:
  virtual ~Factorization() = default;
  virtual bool factor(const Sp& a) = 0;
  virtual bool solve(const std::vector<double>& b, std::vector<double>& x) = 0;
  virtual std::string name() const = 0;
};

std::unique_ptr<Factorization> make_eigen_llt();

// Returns nullptr when the build did not include cuDSS (BP5_WITH_CUDSS off).
std::unique_ptr<Factorization> make_cudss();

// Relative residual ||A x - b|| / ||b||, for reporting.
double residual(const Sp& a, const std::vector<double>& x, const std::vector<double>& b);

}  // namespace bp5
