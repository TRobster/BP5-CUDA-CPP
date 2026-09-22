#include "bp5/params.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace bp5 {
namespace {

std::string trim(const std::string& s) {
  const char* ws = " \t\r\n";
  const size_t a = s.find_first_not_of(ws);
  if (a == std::string::npos) return "";
  const size_t b = s.find_last_not_of(ws);
  return s.substr(a, b - a + 1);
}

}  // namespace

Params Params::read(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open parameter file: " + path);

  std::map<std::string, std::string> kv;
  std::string line;
  while (std::getline(in, line)) {
    const size_t hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string k = trim(line.substr(0, eq));
    const std::string v = trim(line.substr(eq + 1));
    if (!k.empty() && !v.empty()) kv[k] = v;
  }

  Params p;
  auto d = [&](const char* k, double& dst) {
    auto it = kv.find(k);
    if (it != kv.end()) dst = std::stod(it->second);
  };
  auto i = [&](const char* k, int& dst) {
    auto it = kv.find(k);
    if (it != kv.end()) dst = std::stoi(it->second);
  };
  auto s = [&](const char* k, std::string& dst) {
    auto it = kv.find(k);
    if (it != kv.end()) dst = it->second;
  };

  s("pth", p.pth);
  i("stride_space", p.stride_space);
  i("stride_time", p.stride_time);
  i("SBPp", p.SBPp);

  d("Lx1", p.Lx1); d("Lx2", p.Lx2);
  d("Ly1", p.Ly1); d("Ly2", p.Ly2);
  d("Lz1", p.Lz1); d("Lz2", p.Lz2);
  i("Nx", p.Nx); i("Ny", p.Ny); i("Nz", p.Nz);

  d("ρ", p.rho);      // rho
  d("cs", p.cs);
  d("ν", p.nu);       // nu

  d("RSamin", p.RSamin);
  d("RSamax", p.RSamax);
  d("RSb", p.RSb);
  d("σn", p.sigman);  // sigma_n
  d("RSDc", p.RSDc);
  d("Vp", p.Vp);
  d("RSVinit", p.RSVinit);
  d("RSV0", p.RSV0);
  d("RSf0", p.RSf0);
  d("RShs", p.RShs);
  d("RSht", p.RSht);
  d("RSH", p.RSH);
  d("RSl", p.RSl);
  d("RSWf", p.RSWf);
  d("RSlf", p.RSlf);
  d("w", p.RSw);
  d("sim_years", p.sim_years);

  p.finalize();
  return p;
}

void Params::finalize() {
  mu = cs * cs * rho;
  lambda = 2.0 * mu * nu / (1.0 - 2.0 * nu);
  eta = mu / (2.0 * cs);
}

void Params::print() const {
  std::printf("  domain    x [%g, %g]  y [%g, %g]  z [%g, %g]  (km)\n",
              Lx1, Lx2, Ly1, Ly2, Lz1, Lz2);
  std::printf("  grid      Nx=%d Ny=%d Nz=%d  -> %d x %d x %d points\n",
              Nx, Ny, Nz, Nx + 1, Ny + 1, Nz + 1);
  std::printf("  SBP order p = %d\n", SBPp);
  std::printf("  material  rho=%g cs=%g nu=%g  ->  mu=%g lambda=%g eta=%g\n",
              rho, cs, nu, mu, lambda, eta);
}

}  // namespace bp5
