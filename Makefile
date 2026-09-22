# bp5CUDA
#
#   make                 host-only build (Eigen factorisation). No GPU needed.
#   make CUDSS=1         add the cuDSS backend. Needs CUDA_HOME and CUDSS_ROOT.
#   make run             build, then solve a small case
#   make clean
#
# On Talapas:
#   module load eigen/3.4.0 gcc/13.1.0
#   module load cuda/13.0                 # only for CUDSS=1
#
# ---------------------------------------------------------------- config --

CXX       ?= g++
NVCC      ?= $(CUDA_HOME)/bin/nvcc
EIGEN     ?= /packages/eigen/3.4.0/include/eigen3
CUDA_HOME ?= /packages/nvhpc/25.9/cuda/13.0
CUSPARSE  ?= /packages/nvhpc/25.9/math_libs/13.0/targets/x86_64-linux
CUDSS_ROOT ?= $(CUDA_HOME)
ARCH      ?= sm_80

BIN    = bp5_assemble
OBJDIR = obj

CXXFLAGS = -std=c++17 -O2 -g -Wall -Iinclude -I$(EIGEN)
LDLIBS   =

# --------------------------------------------------------------- sources --

SRC = src/params.cpp      \
      src/sparse.cpp      \
      src/sbp1d.cpp       \
      src/metrics.cpp     \
      src/assemble.cpp    \
      src/factor_eigen.cpp \
      src/main.cpp

# The cuDSS backend is either the real .cu or a stub that returns nullptr,
# never both. Keeping them in separate files means the default build never
# needs nvcc.
ifeq ($(CUDSS),1)
  CXXFLAGS += -DBP5_WITH_CUDSS -I$(CUDSS_ROOT)/include -I$(CUDA_HOME)/include
  NVCCFLAGS = -std=c++17 -O2 -arch=$(ARCH) -DBP5_WITH_CUDSS \
              -Iinclude -I$(EIGEN) -I$(CUDSS_ROOT)/include
  LDLIBS   += -L$(CUDSS_ROOT)/lib64 -L$(CUDSS_ROOT)/lib -lcudss \
              -L$(CUDA_HOME)/lib64 -lcudart
  CUOBJ     = $(OBJDIR)/factor_cudss.o
else
  SRC      += src/factor_cudss_stub.cpp
  CUOBJ     =
endif

OBJ = $(SRC:src/%.cpp=$(OBJDIR)/%.o) $(CUOBJ)

# ----------------------------------------------------------------- rules --

.PHONY: all run clean help

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDLIBS)

$(OBJDIR)/%.o: src/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/factor_cudss.o: src/factor_cudss.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

run: $(BIN)
	./$(BIN) ../Thrase.jl/examples/bp5-qd.dat --n 12 --backend eigen

# ---------------------------------------------------- conditioner test --
# Needs nvcc to build and a GPU to run:
#     make test_conditioner && srun -p gpu --gres=gpu:1 ./test_conditioner
# cusparse.h deprecates its own csric02 declarations, hence -Wno-deprecated.

CONDFLAGS = -std=c++17 -O2 -arch=$(ARCH) -Wno-deprecated-declarations \
            -Iinclude -I$(CUSPARSE)/include

test_conditioner: src/conditioner.cu test/test_conditioner.cu
	$(NVCC) $(CONDFLAGS) $^ -L$(CUSPARSE)/lib -lcusparse -o $@

# ------------------------------------------------- HM -> preconditioner --
# Links the Eigen assembly (g++) against the cuSPARSE preconditioner (nvcc).
# Only conditioner.cu goes through nvcc; the bridge is plain C++ because
# conditioner.cuh exposes no device syntax.
#     make test_hm_precond && srun -p gpu --gres=gpu:1 \
#         ./test_hm_precond ../Thrase.jl/examples/bp5-qd.dat --n 12

HMOBJ = $(OBJDIR)/params.o $(OBJDIR)/sparse.o $(OBJDIR)/sbp1d.o \
        $(OBJDIR)/metrics.o $(OBJDIR)/assemble.o $(OBJDIR)/factor_eigen.o

test_hm_precond: $(HMOBJ) $(OBJDIR)/conditioner.o $(OBJDIR)/test_hm_precond.o
	$(CXX) $^ -o $@ -L$(CUSPARSE)/lib -lcusparse -L$(CUDA_HOME)/lib64 -lcudart

$(OBJDIR)/conditioner.o: src/conditioner.cu | $(OBJDIR)
	$(NVCC) $(CONDFLAGS) -c $< -o $@

$(OBJDIR)/test_hm_precond.o: test/test_hm_precond.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -I$(CUSPARSE)/include -I$(CUDA_HOME)/include -c $< -o $@

# ------------------------------------------------------------ CG on HM --
#     make test_cg && srun -p gpu --gres=gpu:1 \
#         ./test_cg ../Thrase.jl/examples/bp5-qd.dat --n 16

test_cg: $(HMOBJ) $(OBJDIR)/conditioner.o $(OBJDIR)/conjugateGR.o $(OBJDIR)/test_cg.o
	$(CXX) $^ -o $@ -L$(CUSPARSE)/lib -lcusparse -lcublas -L$(CUDA_HOME)/lib64 -lcudart

$(OBJDIR)/conjugateGR.o: src/conjugateGR.cu | $(OBJDIR)
	$(NVCC) $(CONDFLAGS) -c $< -o $@

$(OBJDIR)/test_cg.o: test/test_cg.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -I$(CUSPARSE)/include -I$(CUDA_HOME)/include -c $< -o $@

# ------------------------------------------------ 1-D MMS convergence test --
# Host-only, no GPU, no CUDA, no .dat file. This is the accuracy oracle: the
# checks in main.cpp confirm the operator is consistent, this one confirms the
# 1-D second derivative is p-th order. Depends on sbp1d alone.
#     make test_converge && ./test_converge

MMSOBJ = $(OBJDIR)/sparse.o $(OBJDIR)/sbp1d.o $(OBJDIR)/converge.o

test_converge: $(MMSOBJ) $(OBJDIR)/test_converge.o
	$(CXX) $^ -o $@

# Eigen's SelfAdjointEigenSolver trips -Wmaybe-uninitialized on gcc 13. The
# warning is inside Eigen, not this code.
$(OBJDIR)/test_converge.o: test/test_converge.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -Wno-maybe-uninitialized -c $< -o $@

# ------------------------------------- 1-D solution error, CG + IC(0) on GPU --
# The rung above test_converge: solves HM u = H.*f + HB g instead of just
# applying the operator. Needs nvcc to build and a GPU to run.
#     make test_solution_cg && srun -p gpu \
#         --gres=gpu:nvidia_a100_80gb_pcie:1 ./test_solution_cg

SOLOBJ = $(OBJDIR)/sparse.o $(OBJDIR)/sbp1d.o $(OBJDIR)/converge.o \
         $(OBJDIR)/factor_eigen.o $(OBJDIR)/conditioner.o $(OBJDIR)/conjugateGR.o

test_solution_cg: $(SOLOBJ) $(OBJDIR)/test_solution_cg.o
	$(CXX) $^ -o $@ -L$(CUSPARSE)/lib -lcusparse -lcublas -L$(CUDA_HOME)/lib64 -lcudart

$(OBJDIR)/test_solution_cg.o: test/test_solution_cg.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -I$(CUSPARSE)/include -I$(CUDA_HOME)/include -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BIN) test_conditioner test_hm_precond test_cg test_converge \
	       test_solution_cg

help:
	@sed -n '2,10p' Makefile
