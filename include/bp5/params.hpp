#pragma once
//
// Reader for the BP5 `.dat` parameter files (examples/bp5-qd.dat in Thrase.jl).
// Format is `key = value`, '#' starts a comment. Keys are UTF-8 and several are
// Greek, so they are matched as raw byte strings.
//
#include <map>
#include <string>

namespace bp5 {

struct Params {
  // Output / discretisation
  std::string pth = "./out/";
  int stride_space = 1, stride_time = 20;
  int SBPp = 2;

  // Domain, km. Physical box is (Lx1,Lx2) x (Ly1,Ly2) x (Lz1,Lz2).
  double Lx1 = 0, Lx2 = 128;
  double Ly1 = -64, Ly2 = 64;
  double Lz1 = 0, Lz2 = 128;
  int Nx = 128, Ny = 128, Nz = 128;   // intervals, so nqp = Nx+1 etc.

  // Material
  double rho = 2.670, cs = 3.464, nu = 0.25;

  // Rate-and-state (unused by the assembly, carried for the next stage)
  double RSamin = 0.004, RSamax = 0.04, RSb = 0.03;
  double sigman = 100, RSDc = 0.56;
  double Vp = 1e-9, RSVinit = 1e-9, RSV0 = 1e-6, RSf0 = 0.6;
  double RShs = 2, RSht = 2, RSH = 12, RSl = 60, RSWf = 40, RSlf = 100, RSw = 12;
  double sim_years = 1800;

  // Derived elastic moduli, filled by `finalize`.
  double mu = 0, lambda = 0, eta = 0;

  // Read a .dat file. Missing keys keep their defaults above.
  static Params read(const std::string& path);

  // mu = rho*cs^2,  lambda = 2*mu*nu/(1-2*nu),  eta = mu/(2*cs)
  // (same expressions as BP5-QD_Driver.jl).
  void finalize();

  void print() const;
};

}  // namespace bp5
