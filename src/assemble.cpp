#include "bp5/assemble.hpp"

#include <cstdio>
#include <stdexcept>

namespace bp5 {
namespace {

// Diagonal of the 1-D norm H as a plain vector.
std::vector<double> diag_of(const Sp& h) {
  std::vector<double> d(h.rows(), 0.0);
  for (int k = 0; k < h.outerSize(); ++k)
    for (Sp::InnerIterator it(h, k); it; ++it)
      if (it.row() == it.col()) d[it.row()] = it.value();
  return d;
}

// Volume norm H = Hs (X) Hr (X) Hq, as a diagonal vector over volume nodes.
std::vector<double> volume_norm(const std::array<Sbp1D, 3>& d1, const Grid& g) {
  const std::vector<double> hq = diag_of(d1[Q].H);
  const std::vector<double> hr = diag_of(d1[R].H);
  const std::vector<double> hs = diag_of(d1[S].H);
  std::vector<double> hv(g.np());
  for (int s = 0; s < g.nsp; ++s)
    for (int r = 0; r < g.nrp; ++r)
      for (int q = 0; q < g.nqp; ++q) hv[g.vidx(q, r, s)] = hq[q] * hr[r] * hs[s];
  return hv;
}

// e_f * sJ_f * H_f * e_f^T. Since sJ is constant on the face and H_f is
// diagonal, this whole product is a diagonal (np x np) matrix: the product of
// the two in-plane 1-D norms, times sJ, on the face plane and zero elsewhere.
Sp face_mass(int f, const std::array<Sbp1D, 3>& d1, const Grid& g, double sJ) {
  const std::vector<double> h[3] = {diag_of(d1[Q].H), diag_of(d1[R].H), diag_of(d1[S].H)};
  const int a = static_cast<int>(face_dir(f));
  const int b = (a + 1) % 3, c = (a + 2) % 3;

  std::vector<double> dv(g.np(), 0.0);
  const int nq = g.nqp, nr = g.nrp, ns = g.nsp;
  const int fixed = face_is_low(f) ? 0 : (a == 0 ? nq - 1 : a == 1 ? nr - 1 : ns - 1);

  for (int s = 0; s < ns; ++s)
    for (int r = 0; r < nr; ++r)
      for (int q = 0; q < nq; ++q) {
        const int idx[3] = {q, r, s};
        if (idx[a] != fixed) continue;
        dv[g.vidx(q, r, s)] = sJ * h[b][idx[b]] * h[c][idx[c]];
      }
  return diag(dv);
}

}  // namespace

size_t estimate_bytes(const Grid& g) {
  // Measured at ~15.2 nonzeros per row and flat in N (15.6 at N=8, 15.2 at
  // N=24). It is this low because the affine diagonal metric zeroes most of
  // C[a][i][b][j]: only a handful of the 81 entries survive, so the mixed
  // D_a*D_b terms contribute far less fill than a general curvilinear map
  // would. 12 bytes per entry (double value + int32 column index).
  return static_cast<size_t>(g.ndof()) * 16ull * 12ull;
}

Operators assemble(const Params& p, const Metrics& m) {
  Operators o;
  o.grid = Grid{p.Nx + 1, p.Ny + 1, p.Nz + 1};
  const Grid& g = o.grid;
  const int np = g.np();

  const int N[3] = {p.Nx, p.Ny, p.Nz};
  for (int a = 0; a < 3; ++a) o.d1[a] = Sbp1D::make(p.SBPp, N[a]);

  o.Hvol = volume_norm(o.d1, g);

  // Three-dimensional lifts of the one-dimensional operators.
  Sp D1[3], D2[3], BS[3];
  for (int a = 0; a < 3; ++a) {
    const Dir d = static_cast<Dir>(a);
    D1[a] = lift3(d, o.d1[a].D1, g);
    D2[a] = lift3(d, o.d1[a].D2, g);
    BS[a] = lift3(d, o.d1[a].BS, g);
  }

  // ---------------------------------------------------------------- volume --
  // A[i][j] = (1/J) * ( sum_a C[a][i][a][j] D2_a + sum_{a!=b} C[a][i][b][j] D_a D_b )
  //
  // The diagonal terms use the narrow-stencil second derivative; the mixed
  // terms use a product of first derivatives. This mirrors ops_bp5.jl:551-572,
  // where D11/D22/D33 come from var_3D_D2*_fast and the rest are Kronecker
  // products of D1. With constant coefficients the scalar multiply below is
  // exact; see the note in metrics.hpp for the heterogeneous case.
  const double invJ = 1.0 / m.J;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      Sp acc(np, np);
      for (int a = 0; a < 3; ++a) {
        const double c = m.C[a][i][a][j];
        if (c != 0.0) acc = acc + Sp(c * D2[a]);
      }
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          if (a == b) continue;
          const double c = m.C[a][i][b][j];
          if (c != 0.0) acc = acc + Sp(c * Sp(D1[a] * D1[b]));
        }
      o.A[i][j] = invJ * acc;
    }

  // ------------------------------------------------------------- traction --
  // T^f[i][j] = sign_f * (1/sJ_f) * P_f * sum_b C[a][i][b][j] * op_b
  // with op_a = BS_a (boundary derivative) and op_b = D1_b for b != a.
  // ops_bp5.jl:674-749.
  for (int f = 0; f < NFACE; ++f) {
    const int a = static_cast<int>(face_dir(f));
    const Sp P = face_projector(f, g);
    const double pre = face_normal_sign(f) / m.sJ[f];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        Sp acc(np, np);
        for (int b = 0; b < 3; ++b) {
          const double c = m.C[a][i][b][j];
          if (c == 0.0) continue;
          acc = acc + Sp(c * (b == a ? BS[a] : D1[b]));
        }
        o.T[f][i][j] = pre * Sp(P * acc);
      }
  }

  // -------------------------------------------------------------- penalty --
  // Z^f[i][j] = (beta * d / (h_a * sJ_f)) * C[a][i][a][j] * P_f
  //
  // The general form is (beta*d/h_a) * sJI_f * sum_pq N_p C[p][i][q][j] N_q.
  // On this box the outward normal is exactly +/- e_a, so only p = q = a
  // survives and the sign squares away. ops_bp5.jl:755-830.
  Sp Z[NFACE][3][3];
  for (int f = 0; f < NFACE; ++f) {
    const int a = static_cast<int>(face_dir(f));
    const Sp P = face_projector(f, g);
    const double ha = diag_of(o.d1[a].H).front();     // H[0,0], = h/2 for p = 2
    const double pre = kPenaltyBeta * kPenaltyD / (ha * m.sJ[f]);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) Z[f][i][j] = Sp((pre * m.C[a][i][a][j]) * P);
  }

  // ------------------------------------------------------------------ SAT --
  // S[i][j] = JHI * (  sum_{f in {0,1}} (T^f[j][i] - Z^f[j][i])^T * M_f
  //                  - sum_{f in {2..5}} M_f * T^f[i][j] )
  Sp M[NFACE];
  for (int f = 0; f < NFACE; ++f) M[f] = face_mass(f, o.d1, g, m.sJ[f]);

  std::vector<double> jhi(np);
  for (int k = 0; k < np; ++k) jhi[k] = 1.0 / (o.Hvol[k] * m.J);
  const Sp JHI = diag(jhi);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      Sp acc(np, np);
      for (int f = 0; f < 2; ++f) {                       // Dirichlet: indices transposed
        const Sp k = Sp(o.T[f][j][i] - Z[f][j][i]);
        acc = acc + Sp(Sp(k.transpose()) * M[f]);
      }
      for (int f = 2; f < NFACE; ++f)                     // Neumann: indices natural
        acc = acc - Sp(M[f] * o.T[f][i][j]);
      o.S[i][j] = Sp(JHI * acc);
    }

  // ------------------------------------------------------------------- HM --
  // HM = Hn * (A + S) with Hn = -H. The negation is what makes the result
  // positive definite rather than negative definite, so Cholesky applies.
  Sp Hn = diag([&] {
    std::vector<double> v(np);
    for (int k = 0; k < np; ++k) v[k] = -o.Hvol[k];
    return v;
  }());

  Sp blocks[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) blocks[i][j] = Sp(Hn * Sp(o.A[i][j] + o.S[i][j]));

  o.HM = block3(blocks);
  o.HM.makeCompressed();
  return o;
}

}  // namespace bp5
