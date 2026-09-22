#include "bp5/sparse.hpp"

#include <algorithm>
#include <cmath>

namespace bp5 {

Sp lift3(Dir dir, const Sp& m1d, const Grid& g) {
  const int np = g.np();
  std::vector<Trip> t;
  t.reserve(static_cast<size_t>(m1d.nonZeros()) *
            static_cast<size_t>(np) / std::max(1, static_cast<int>(m1d.rows())));

  for (int k = 0; k < m1d.outerSize(); ++k) {
    for (Sp::InnerIterator it(m1d, k); it; ++it) {
      const int a = static_cast<int>(it.row());   // output index in `dir`
      const int b = static_cast<int>(it.col());   // input  index in `dir`
      const double v = it.value();
      switch (dir) {
        case Q:
          for (int s = 0; s < g.nsp; ++s)
            for (int r = 0; r < g.nrp; ++r)
              t.emplace_back(g.vidx(a, r, s), g.vidx(b, r, s), v);
          break;
        case R:
          for (int s = 0; s < g.nsp; ++s)
            for (int q = 0; q < g.nqp; ++q)
              t.emplace_back(g.vidx(q, a, s), g.vidx(q, b, s), v);
          break;
        case S:
          for (int r = 0; r < g.nrp; ++r)
            for (int q = 0; q < g.nqp; ++q)
              t.emplace_back(g.vidx(q, r, a), g.vidx(q, r, b), v);
          break;
      }
    }
  }
  Sp out(np, np);
  out.setFromTriplets(t.begin(), t.end());
  return out;
}
// sparse kronecker product 
Sp kron(const Sp& a, const Sp& b) {
  std::vector<Trip> t;
  t.reserve(static_cast<size_t>(a.nonZeros()) * static_cast<size_t>(b.nonZeros()));
  for (int ka = 0; ka < a.outerSize(); ++ka)
    for (Sp::InnerIterator ia(a, ka); ia; ++ia)
      for (int kb = 0; kb < b.outerSize(); ++kb)
        for (Sp::InnerIterator ib(b, kb); ib; ++ib)
          t.emplace_back(static_cast<int>(ia.row() * b.rows() + ib.row()),
                         static_cast<int>(ia.col() * b.cols() + ib.col()),
                         ia.value() * ib.value());
  Sp out(a.rows() * b.rows(), a.cols() * b.cols());
  out.setFromTriplets(t.begin(), t.end());
  return out;
}

Sp face_restrict(int f, const Grid& g) {
  const int nf = g.nface(f);
  std::vector<Trip> t;
  t.reserve(nf);
  const bool low = face_is_low(f);
  switch (face_dir(f)) {
    case Q: {
      const int q = low ? 0 : g.nqp - 1;
      for (int s = 0; s < g.nsp; ++s)
        for (int r = 0; r < g.nrp; ++r) t.emplace_back(g.vidx(q, r, s), r + g.nrp * s, 1.0);
      break;
    }
    case R: {
      const int r = low ? 0 : g.nrp - 1;
      for (int s = 0; s < g.nsp; ++s)
        for (int q = 0; q < g.nqp; ++q) t.emplace_back(g.vidx(q, r, s), q + g.nqp * s, 1.0);
      break;
    }
    default: {
      const int s = low ? 0 : g.nsp - 1;
      for (int r = 0; r < g.nrp; ++r)
        for (int q = 0; q < g.nqp; ++q) t.emplace_back(g.vidx(q, r, s), q + g.nqp * r, 1.0);
      break;
    }
  }
  Sp out(g.np(), nf);
  out.setFromTriplets(t.begin(), t.end());
  return out;
}

Sp face_projector(int f, const Grid& g) {
  const Sp e = face_restrict(f, g);
  return Sp(e * Sp(e.transpose()));
}

// takes vector plain doubles (&d) and constructs n * n sparse matrix. vector to matrix 
Sp diag(const std::vector<double>& d) {
  const int n = static_cast<int>(d.size());
  std::vector<Trip> t;
  t.reserve(n);
  for (int i = 0; i < n; ++i) t.emplace_back(i, i, d[i]);
  Sp out(n, n);
  out.setFromTriplets(t.begin(), t.end());
  return out;
}

Sp eye(int n) { return diag(std::vector<double>(n, 1.0)); }

Sp block3(const Sp blocks[3][3]) {
  const int np = static_cast<int>(blocks[0][0].rows());
  std::vector<Trip> t;
  size_t nnz = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) nnz += static_cast<size_t>(blocks[i][j].nonZeros());
  t.reserve(nnz);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < blocks[i][j].outerSize(); ++k)
        for (Sp::InnerIterator it(blocks[i][j], k); it; ++it)
          t.emplace_back(static_cast<int>(it.row()) + i * np,
                         static_cast<int>(it.col()) + j * np, it.value());

  Sp out(3 * np, 3 * np);
  out.setFromTriplets(t.begin(), t.end());
  return out;
}

double symmetry_error(const Sp& a) {
  const Sp d = Sp(a - Sp(a.transpose()));
  double amax = 0.0, dmax = 0.0;
  for (int k = 0; k < a.outerSize(); ++k)
    for (Sp::InnerIterator it(a, k); it; ++it) amax = std::max(amax, std::fabs(it.value()));
  for (int k = 0; k < d.outerSize(); ++k)
    for (Sp::InnerIterator it(d, k); it; ++it) dmax = std::max(dmax, std::fabs(it.value()));
  return amax > 0.0 ? dmax / amax : dmax;
}

}  // namespace bp5
