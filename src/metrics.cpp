#include "bp5/metrics.hpp"

#include <cstdio>

namespace bp5 {
namespace {
inline double d(int a, int b) { return a == b ? 1.0 : 0.0; }
}  // namespace

Metrics Metrics::make(const Params& p) {
  Metrics m;
  const double lo[3] = {p.Lx1, p.Ly1, p.Lz1};
  const double hi[3] = {p.Lx2, p.Ly2, p.Lz2};

  for (int a = 0; a < 3; ++a) {
    m.alpha[a] = 0.5 * (hi[a] - lo[a]);
    m.beta[a]  = 0.5 * (hi[a] + lo[a]);
    m.G[a]     = 1.0 / m.alpha[a];
  }
  m.J = m.alpha[0] * m.alpha[1] * m.alpha[2];

  // Surface Jacobian: the product of the two in-plane half-widths. Outward
  // normals are exactly -e_a (low faces) and +e_a (high faces), which is why
  // the Z penalty below reduces to a single C entry.
  for (int f = 0; f < NFACE; ++f) {
    const int a = static_cast<int>(face_dir(f));
    m.sJ[f] = m.alpha[(a + 1) % 3] * m.alpha[(a + 2) % 3];
  }

  const double lam = p.lambda, mu = p.mu;
  for (int a = 0; a < 3; ++a)
    for (int i = 0; i < 3; ++i)
      for (int b = 0; b < 3; ++b)
        for (int j = 0; j < 3; ++j)
          m.C[a][i][b][j] =
              m.J * (lam * d(i, a) * d(j, b) * m.G[i] * m.G[j] +
                     mu  * d(i, j) * d(a, b) * m.G[a] * m.G[a] +
                     mu  * d(j, a) * d(i, b) * m.G[i] * m.G[j]);
  return m;
}

void Metrics::print() const {
  std::printf("  alpha     (%g, %g, %g)\n", alpha[0], alpha[1], alpha[2]);
  std::printf("  J         %g\n", J);
  std::printf("  sJ        faces 0..5: %g %g %g %g %g %g\n",
              sJ[0], sJ[1], sJ[2], sJ[3], sJ[4], sJ[5]);
  std::printf("  C[0000]   %g   (expect J*G0^2*(lambda+2mu))\n", C[0][0][0][0]);
  std::printf("  C[0011]   %g   (expect J*lambda*G0*G1)\n", C[0][0][1][1]);
  std::printf("  C[0110]   %g   (expect J*mu*G0*G1)\n", C[0][1][1][0]);
}

}  // namespace bp5
