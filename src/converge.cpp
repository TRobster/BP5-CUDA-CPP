#include "bp5/converge.hpp"

#include <cmath>

namespace bp5 {

// flips diag() logic. take sparse matrix, pull out double-vals for vector. only useful for converge testing
std::vector<double> h_diag(const Sbp1D& d) 
{
  std::vector<double> hd(d.n, 0.0);
  for (int k = 0; k < d.H.outerSize(); ++k)
    for (Sp::InnerIterator it(d.H, k); it; ++it)
      if (it.row() == it.col()) hd[it.row()] = it.value();
  return hd;
}

namespace {

// Node index of end f: 0 = left (x = -1), 1 = right (x = +1).
int end_index(const Sbp1D& d, int f) { return f == 0 ? 0 : d.n - 1; }

// P_f, the 0/1 mask onto end f. In 1-D this is also the face mass M_f, because
// a face is a single point: sJ = 1 and there is no in-plane norm.
Sp mask(const Sbp1D& d, int f) {
  std::vector<double> v(d.n, 0.0);
  v[end_index(d, f)] = 1.0;
  return diag(v);
}

}  // namespace

double Mms::f(double x) const { return std::cos(x) + x * std::sin(x); }

double Mms::u(double x) const { return (3.0 * std::cos(x) + x * std::sin(x)) / mu; }

// Coordinate geometry
std::vector<double> nodes(const Sbp1D& d) {
  std::vector<double> x(d.n);
  for (int i = 0; i < d.n; ++i) x[i] = -1.0 + d.h * static_cast<double>(i);
  return x;
}

std::vector<double> sample_u(const Sbp1D& d, const Mms& s) {
  const std::vector<double> x = nodes(d);
  std::vector<double> v(d.n);
  for (int i = 0; i < d.n; ++i) v[i] = s.u(x[i]);
  return v;
}

std::vector<double> sample_f(const Sbp1D& d, const Mms& s) {
  const std::vector<double> x = nodes(d);
  std::vector<double> v(d.n);
  for (int i = 0; i < d.n; ++i) v[i] = s.f(x[i]);
  return v;
}

std::vector<double> source_rhs(const Sbp1D& d, const Mms& s) {
  std::vector<double> b = sample_f(d, s);
  const std::vector<double> hd = h_diag(d);
  for (int i = 0; i < d.n; ++i) b[i] *= hd[i];
  return b;
}

// r = mu * ''(D2) * u_exact + f, build!
Residual residual(const Sbp1D& d, const Mms& s, int margin) {
  const std::vector<double> ue = sample_u(d, s);
  const std::vector<double> fv = sample_f(d, s);

  // r = mu * D2 * u_exact + f
  // map uv to eigen vector dynamically sized, same type (double)
  Eigen::Map<const Eigen::VectorXd> uv(ue.data(), d.n);
  // same, bookkeeping 
  const Eigen::VectorXd r = s.mu * (d.D2 * uv) + Eigen::Map<const Eigen::VectorXd>(fv.data(), d.n);

  const std::vector<double> hd = h_diag(d);

  Residual out;
  out.h = d.h;
  double sum = 0.0;
  for (int i = margin; i < d.n - margin; ++i) {
    out.linf = std::max(out.linf, std::fabs(r(i)));
    sum += hd[i] * r(i) * r(i);
    ++out.nscored;
  }
  out.l2 = std::sqrt(sum);
  return out;
}

Eigen::VectorXd boundary_data(const Sbp1D& d, const Mms& s) {
  Eigen::VectorXd g = Eigen::VectorXd::Zero(d.n);
  g(end_index(d, 0)) = s.u(-1.0);
  g(end_index(d, 1)) = s.u(+1.0);
  return g;
}

// construct system 
System build(const Sbp1D& d, const Mms& s) {
  System sys;
  const int n = d.n;
  const std::vector<double> hd = h_diag(d);

  // T^f = sign_f * mu * P_f * BS.  sign_f is the outward normal: -1 at the left
  // end, +1 at the right. BS is orientation-neutral (both rows give d/dx), so
  // the sign has to be applied here ; see the note in sbp1d.hpp.

  const double pen = kPenaltyBeta * kPenaltyD / hd.front();
  for (int f = 0; f < 2; ++f) {
    const double sign = (f == 0) ? -1.0 : +1.0;
    const Sp P = mask(d, f);
    sys.T[f] = Sp((sign * s.mu) * Sp(P * d.BS));
    sys.Z[f] = Sp((pen * s.mu) * P);
  }

  // acc = sum_f (T^f - Z^f)^T M_f, with M_f = P_f. Both ends Dirichlet, so both
  // use the transposed (adjoint) form ; assemble.cpp:157-160.
  // SAT TERMS! Acc = Accumulated. Takes Traction T, takes Penalty Z
  Sp acc(n, n);
  for (int f = 0; f < 2; ++f) {
    const Sp K = Sp(sys.T[f] - sys.Z[f]);
    acc = acc + Sp(Sp(K.transpose()) * mask(d, f));
  }

  // HM = -H*(A + S) with S = HI*acc, and H*HI = I, so the SAT enters undivided.
  std::vector<double> nh(n);
  for (int i = 0; i < n; ++i) nh[i] = -hd[i];
  sys.HM = Sp(Sp(diag(nh) * Sp(s.mu * d.D2)) - acc);
  sys.HM.makeCompressed();

  // The SAT acts on (u_f - g_f). The u half is acc; the g half is its negative.
  sys.HB = Sp(-acc);
  sys.HB.makeCompressed();

  const std::vector<double> fv = source_rhs(d, s);
  sys.b = Eigen::Map<const Eigen::VectorXd>(fv.data(), n) + sys.HB * boundary_data(d, s);
  return sys;
}

double rate(double err_coarse, double err_fine, double refine) {
  if (!std::isfinite(err_coarse) || !std::isfinite(err_fine)) return 0.0;
  if (err_coarse <= 0.0 || err_fine <= 0.0) return 0.0;
  return std::log(err_coarse / err_fine) / std::log(refine);
}

}  // namespace bp5
