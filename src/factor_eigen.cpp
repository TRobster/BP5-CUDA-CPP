#include "bp5/factor.hpp"

#include <Eigen/SparseCholesky>
#include <cmath>
#include <cstdio>

namespace bp5 {

Csr to_csr(const Sp& a, bool upper_only) {
  Eigen::SparseMatrix<double, Eigen::RowMajor> r(a);
  r.makeCompressed();

  Csr c;
  c.n = static_cast<int>(r.rows());
  c.rowptr.resize(c.n + 1);
  c.rowptr[0] = 0;
  for (int i = 0; i < c.n; ++i) {
    for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(r, i); it; ++it) {
      const int j = static_cast<int>(it.col());
      if (upper_only && j < i) continue;
      c.colind.push_back(j);
      c.val.push_back(it.value());
    }
    c.rowptr[i + 1] = static_cast<int>(c.colind.size());
  }
  return c;
}

Csr to_csr_lower(const Sp& a) {
  Eigen::SparseMatrix<double, Eigen::RowMajor> r(a);
  r.makeCompressed();

  Csr c;
  c.n = static_cast<int>(r.rows());
  c.rowptr.resize(c.n + 1);
  c.rowptr[0] = 0;
  for (int i = 0; i < c.n; ++i) {
    for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(r, i); it; ++it) {
      const int j = static_cast<int>(it.col());
      if (j > i) continue;
      c.colind.push_back(j);
      c.val.push_back(it.value());
    }
    c.rowptr[i + 1] = static_cast<int>(c.colind.size());
  }
  return c;
}

double residual(const Sp& a, const std::vector<double>& x, const std::vector<double>& b) {
  const int n = static_cast<int>(a.rows());
  Eigen::Map<const Eigen::VectorXd> xv(x.data(), n);
  Eigen::Map<const Eigen::VectorXd> bv(b.data(), n);
  const double nb = bv.norm();
  const double nr = (a * xv - bv).norm();
  return nb > 0 ? nr / nb : nr;
}

namespace {

class EigenLlt final : public Factorization {
 public:
  bool factor(const Sp& a) override {
    llt_.compute(a);
    if (llt_.info() != Eigen::Success) {
      std::printf("  [eigen] factorisation failed: matrix is not positive definite\n");
      return false;
    }
    return true;
  }

  bool solve(const std::vector<double>& b, std::vector<double>& x) override {
    const int n = static_cast<int>(b.size());
    Eigen::Map<const Eigen::VectorXd> bv(b.data(), n);
    const Eigen::VectorXd xv = llt_.solve(bv);
    if (llt_.info() != Eigen::Success) return false;
    x.assign(xv.data(), xv.data() + n);
    return true;
  }

  std::string name() const override { return "Eigen SimplicialLLT (host)"; }

 private:
  Eigen::SimplicialLLT<Sp> llt_;
};

}  // namespace

std::unique_ptr<Factorization> make_eigen_llt() { return std::make_unique<EigenLlt>(); }

}  // namespace bp5
