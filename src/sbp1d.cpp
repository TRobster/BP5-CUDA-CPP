#include "bp5/sbp1d.hpp"

#include <stdexcept>

namespace bp5 {
namespace {

// One-sided second-order derivative rows used by BS (and by D2's boundary
// closure). Returned as (column offset, weight) pairs relative to the end node,
// scaled by 1/h.
//
//   low  end:  du/dq|_0     = (-3/2 u_0 + 2 u_1 - 1/2 u_2) / h
//   high end:  du/dq|_{n-1} = ( 1/2 u_{n-3} - 2 u_{n-2} + 3/2 u_{n-1}) / h
void bs_rows(int n, double h, std::vector<Trip>& t) {
  const double s = 1.0 / h;
  t.emplace_back(0, 0, -1.5 * s);
  t.emplace_back(0, 1, 2.0 * s);
  t.emplace_back(0, 2, -0.5 * s);
  t.emplace_back(n - 1, n - 3, 0.5 * s);
  t.emplace_back(n - 1, n - 2, -2.0 * s);
  t.emplace_back(n - 1, n - 1, 1.5 * s);
}

}  // namespace

Sbp1D Sbp1D::make(int p, int N) {
  if (p != 2) throw std::runtime_error("sbp1d: only p = 2 is implemented");
  if (N < 4) throw std::runtime_error("sbp1d: need N >= 4");

  Sbp1D o;
  o.n = N + 1;
  o.h = 2.0 / static_cast<double>(N);   // reference domain is [-1, 1]
  const int n = o.n;
  const double h = o.h;

  // H = h * diag(1/2, 1, ..., 1, 1/2)
  std::vector<double> hd(n, h);
  hd.front() = hd.back() = 0.5 * h;
  o.H = diag(hd);
  std::vector<double> hi(n);
  for (int i = 0; i < n; ++i) hi[i] = 1.0 / hd[i];
  o.HI = diag(hi);

  // D1: centred interior, one-sided first order at the ends. This is the
  // standard p = 2 diagonal-norm D1 satisfying H*D1 + D1^T*H = e_n e_n^T - e_0 e_0^T.
  {
    std::vector<Trip> t;
    t.emplace_back(0, 0, -1.0 / h);
    t.emplace_back(0, 1, 1.0 / h);
    for (int i = 1; i < n - 1; ++i) {
      t.emplace_back(i, i - 1, -0.5 / h);
      t.emplace_back(i, i + 1, 0.5 / h);
    }
    t.emplace_back(n - 1, n - 2, -1.0 / h);
    t.emplace_back(n - 1, n - 1, 1.0 / h);
    o.D1 = Sp(n, n);
    o.D1.setFromTriplets(t.begin(), t.end());
  }

  // BS: one-sided d/dq at both ends, zero elsewhere.
  {
    std::vector<Trip> t;
    bs_rows(n, h, t);
    o.BS = Sp(n, n);
    o.BS.setFromTriplets(t.begin(), t.end());
  }

  o.D2 = sbp_d2_var(p, N, std::vector<double>(n, 1.0));
  return o;
}

Sp sbp_d2_var(int p, int N, const std::vector<double>& c) {
  if (p != 2) throw std::runtime_error("sbp_d2_var: only p = 2 is implemented");
  const int n = N + 1;
  if (static_cast<int>(c.size()) != n) throw std::runtime_error("sbp_d2_var: bad coefficient length");
  const double h = 2.0 / static_cast<double>(N);

  // A(c) = sum over cell edges of  cbar_{i+1/2}/h * (e_i - e_{i+1})(e_i - e_{i+1})^T
  // Symmetric positive semi-definite by construction, which is what makes
  // -H*D2 (and hence HM) come out symmetric.
  std::vector<Trip> ta;
  ta.reserve(4 * (n - 1));
  for (int i = 0; i < n - 1; ++i) {
    const double cb = 0.5 * (c[i] + c[i + 1]) / h;
    ta.emplace_back(i, i, cb);
    ta.emplace_back(i, i + 1, -cb);
    ta.emplace_back(i + 1, i, -cb);
    ta.emplace_back(i + 1, i + 1, cb);
  }
  Sp A(n, n);
  A.setFromTriplets(ta.begin(), ta.end());

  // Boundary closure:  + c_{n-1} e_{n-1} bs_{n-1}^T  - c_0 e_0 bs_0^T
  std::vector<Trip> tb;
  bs_rows(n, h, tb);
  std::vector<Trip> tc;
  for (const Trip& e : tb) {
    const double sgn = (e.row() == 0) ? -c[0] : c[n - 1];
    tc.emplace_back(e.row(), e.col(), sgn * e.value());
  }
  Sp Bnd(n, n);
  Bnd.setFromTriplets(tc.begin(), tc.end());

  std::vector<double> hd(n, h);
  hd.front() = hd.back() = 0.5 * h;
  std::vector<double> hi(n);
  for (int i = 0; i < n; ++i) hi[i] = 1.0 / hd[i];

  return Sp(diag(hi) * Sp(Sp(-A) + Bnd));
}

}  // namespace bp5
