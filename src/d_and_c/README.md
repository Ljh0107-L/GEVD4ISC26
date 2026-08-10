# CPU tridiagonal divide-and-conquer

This directory is the CPU-only boundary for the tridiagonal eigensolve. It
contains no CUDA, NCCL, or MPI dependency.

`d_and_c_solver.cpp` provides:

- `solveTridiagonal`, the synchronous MKL `DSTEDC` primitive;
- `AsyncTridiagonalSolve`, the one-shot asynchronous wrapper used by the
  distributed GPU EVD engine;
- ownership and reuse of the large host eigenvector matrix.

The asynchronous wrapper is intentional: the distributed engine starts D&C
after tridiagonal extraction and overlaps it with the GPU SBR/BC
backtransforms. Its destructor joins outstanding work, so an early return from
the caller cannot leave a joinable thread.

When integrated into the distributed engine, the rank's launcher/cgroup CPU
mask is restored before the host eigenvector buffer is first-touched and the
D&C worker is created. The mask is captured independently on every rank and
is dynamically sized, so no single-node or fixed-CPU-count assumption is
made. The standalone benchmark simply inherits the mask with which it is
launched.

## Build only this module

No CUDA or MPI configuration is needed:

```bash
cmake -S "D&C" -B build-d-and-c \
  -DCMAKE_BUILD_TYPE=Release \
  -DMKL_ROOT=/path/to/mkl
cmake --build build-d-and-c -j
```

The complete project also exposes focused build targets:

```bash
cmake --build build --target gevd4isc26_d_and_c d_and_c_benchmark -j
```

## Benchmark

```bash
OMP_NUM_THREADS=128 MKL_NUM_THREADS=128 MKL_DYNAMIC=FALSE \
OMP_PROC_BIND=FALSE \
./build-d-and-c/d_and_c_benchmark \
  --n 32000 --repeat 2 --validation-vectors 3
```

`prepare_seconds` covers host eigenvector-buffer acquisition, first touch, and
worker launch. `solve_seconds` measures only `DSTEDC`. `wall_seconds` covers
both. The sampled residual and orthogonality checks are O(kn), not O(n^2), so
they do not obscure the solve timing.

The following existing runtime controls remain supported:

- `EVD_HOST_Z_CACHE=0|1` controls process-local reuse of the host eigenvector
  matrix (enabled by default);
- `EVD_HOST_Z_TOUCH_THREADS=N` controls parallel first touch (falling back to
  `MKL_NUM_THREADS`, then 64);
- `MKL_NUM_THREADS`, `MKL_DYNAMIC`, and the GNU OpenMP affinity variables
  control the MKL worker team.
