# GEVD4ISC26

[中文](README.md) | **English**

GEVD4ISC26 is a standalone, distributed generalized eigensolver for the real
symmetric-definite problem

```text
A Q = B Q Lambda
Q^T B Q = I
```

The inputs are symmetric `A` and symmetric-positive-definite `B`. The outputs
are a distributed eigenvector matrix `Q` and the replicated diagonal of the
eigenvalue matrix `Lambda`. One MPI rank controls one NVIDIA GPU; matrices use
a two-dimensional block-cyclic distribution across ranks.

The solver provides the complete numerical path: Cholesky factorization,
generalized-to-standard reduction, distributed standard eigendecomposition,
and generalized eigenvector backtransform. The repository also provides a C++
library API, a matrix-file solver, a correctness demo, and performance
benchmarks.

## Provenance and acknowledgements

This repository is a fork of
[hansheng1001/EVD4SC2025](https://github.com/hansheng1001/EVD4SC2025), the
open-source implementation accompanying the SC'25 paper *Rethinking Back
Transformation in 2-Stage Eigenvalue Decomposition on Heterogeneous
Architectures*. On top of it we reproduced the pipelined two-stage EVD
algorithm from the team's follow-up paper
[*Pipelined Dense Symmetric Eigenvalue Decomposition on Multi-GPU
Architectures*](https://arxiv.org/abs/2511.16174) (Wang et al.,
arXiv:2511.16174) and extended it into a complete distributed generalized
eigensolver for the DFTB+ track of the ISC26 Student Cluster Competition.

**We sincerely thank the EVD4SC2025 team (Hansheng Wang et al.) for their
excellent work and open-source contribution — building on their algorithms
and code enabled us to win first place in the DFTB+ track at ISC26.**

This repository is distributed under the MIT license. The top-level
[`LICENSE`](LICENSE) carries both the upstream copyright notice and ours, and
the EVD engine additionally retains the unmodified upstream notice in
[`src/evd/LICENSE`](src/evd/LICENSE).

### References

If you use this repository, please cite the two papers whose algorithms it
builds on:

1. Hansheng Wang, Dajun Huang, Gaoyuan Zou, Lu Shi, Xu Jiang, Xi Wu, Hancong
   Duan, and Shaoshuai Zhang. 2025. *Rethinking Back Transformation in 2-stage
   Eigenvalue Decomposition on Heterogeneous Architectures.* In Proceedings of
   the International Conference for High Performance Computing, Networking,
   Storage and Analysis (SC '25). ACM.
   [doi:10.1145/3712285.3759770](https://doi.org/10.1145/3712285.3759770).
   Code: [hansheng1001/EVD4SC2025](https://github.com/hansheng1001/EVD4SC2025).
2. Hansheng Wang, Ruiyi Zhan, Dajun Huang, Xingchen Liu, Qiao Li, Hancong Duan,
   Dingwen Tao, Guangming Tan, and Shaoshuai Zhang. 2025. *Pipelined Dense
   Symmetric Eigenvalue Decomposition on Multi-GPU Architectures.*
   arXiv:2511.16174. <https://arxiv.org/abs/2511.16174>.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{wang2025rethinking,
  author    = {Wang, Hansheng and Huang, Dajun and Zou, Gaoyuan and Shi, Lu and
               Jiang, Xu and Wu, Xi and Duan, Hancong and Zhang, Shaoshuai},
  title     = {Rethinking Back Transformation in 2-stage Eigenvalue
               Decomposition on Heterogeneous Architectures},
  booktitle = {Proceedings of the International Conference for High Performance
               Computing, Networking, Storage and Analysis (SC '25)},
  year      = {2025},
  publisher = {ACM},
  doi       = {10.1145/3712285.3759770}
}

@misc{wang2025pipelined,
  author        = {Wang, Hansheng and Zhan, Ruiyi and Huang, Dajun and
                   Liu, Xingchen and Li, Qiao and Duan, Hancong and
                   Tao, Dingwen and Tan, Guangming and Zhang, Shaoshuai},
  title         = {Pipelined Dense Symmetric Eigenvalue Decomposition on
                   Multi-GPU Architectures},
  year          = {2025},
  eprint        = {2511.16174},
  archivePrefix = {arXiv}
}
```

</details>

## Numerical pipeline

`GeneralizedEigensolver::solve(A, B)` executes these stages:

1. Cholesky factorization: `B = L L^T`.
2. Generalized-to-standard reduction: `C = L^{-1} A L^{-T}`.
3. Redistribution from the public two-dimensional block-cyclic layout to the
   EVD engine's block-aligned column layout.
4. Standard symmetric EVD: `C Y = Y Lambda`.
   - dense-to-band reduction (SBR);
   - band-to-tridiagonal reduction (BC);
   - tridiagonal divide-and-conquer;
   - SBR backtransform;
   - BC backtransform;
   - eigenvector-composition GEMM.
5. Redistribution of `Y` back to the public layout.
6. Generalized backtransform: `Q = L^{-T}Y`.

The default backtransform prepares `L^{-T}` concurrently with the EVD
backtransform. A lower-memory `L^T Q = Y` triangular-solve path is available
through `BacktransformMethod::TriangularSolve`.

A long-lived solver can collectively reuse the distributed Cholesky factor
and inverse when repeated calls use the same `B`. The inputs on host memory
are never modified; only their lower triangles are consumed.

## Repository layout

```text
.
├── app/
│   ├── common/              shared problem generation and validation
│   ├── gevd_solve.cpp       binary matrix-file interface
│   ├── gevd_validate.cpp        small full-GEVD correctness test
│   ├── gevd_benchmark.cpp   scalable full-GEVD benchmark
│   └── evd_benchmark.cu     standard-EVD performance and tuning benchmark
├── include/gevd4isc26/
│   ├── gevd.hpp             one-stop public include
│   ├── solver.hpp           GEVD solver and result API
│   ├── distributed_matrix.hpp
│   └── matrix_io.hpp
├── src/
│   ├── gevd/                readable top-level GEVD orchestration
│   ├── evd/                 optimized distributed standard EVD engine
│   ├── d_and_c/             standalone-buildable CPU tridiagonal D&C library and benchmark
│   ├── distributed/         rank grid, layouts, and distributed context
│   ├── runtime/             MPI/GPU resources, descriptors, and memory
│   ├── diagnostics/         timing and collective validation
│   ├── io/                  matrix-file implementation
│   └── cmake/               dependency and package configuration
├── CMakeLists.txt
├── README.md
└── README_EN.md
```

The GEVD layer is intentionally stage-oriented:

```text
src/gevd/pipeline.cu             A,B -> Q,Lambda stage ordering
src/gevd/workspace.*             per-solve distributed storage
src/gevd/cholesky.*              B = L L^T
src/gevd/standard_reduction.*    C = L^-1 A L^-T
src/evd/adapter.*                layout boundary around the EVD engine
src/gevd/backtransform.*         Q = L^-T Y or L^T Q = Y
src/gevd/factor_cache.*          collective B-factor reuse
src/d_and_c/d_and_c_solver.*             CPU tridiagonal D&C, async task, and host cache
```

The EVD engine is organized by responsibility:

```text
src/evd/engine/          stage driver and shared types
src/evd/stages/          SBR, BC, tridiagonal extraction, and backtransforms
src/evd/communication/   MPI/NCCL layout and band exchanges
src/evd/kernels/         CUDA numerical kernels
src/evd/runtime/         EVD resource and error handling
src/evd/tuning/          runtime and compile-time tuning policy
src/evd/diagnostics/     opt-in verification and profiling helpers
```

## Requirements

- CMake 3.24 or newer;
- a C++17 compiler and CUDA compiler;
- MPI development headers and runtime;
- CUDA Runtime, cuBLAS, cuSOLVER, and cuRAND;
- cuSOLVERMp and cuBLASMp;
- NCCL and NVSHMEM;
- Intel MKL ILP64 with the GNU threading layer;
- one visible NVIDIA GPU per MPI rank.

CUDA, MPI, NCCL, NVSHMEM, cuSOLVERMp, cuBLASMp, and MKL must be ABI-compatible.
Multi-node jobs additionally require a working NCCL transport configuration.

## Native build

All generated files stay under `build/`, which is ignored by Git.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_CUDA_COMPILER=nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCUSOLVERMP_ROOT=/path/to/cusolvermp \
  -DCUBLASMP_ROOT=/path/to/cublasmp \
  -DNCCL_ROOT=/path/to/nccl \
  -DNVSHMEM_ROOT=/path/to/nvshmem \
  -DMKL_ROOT=/path/to/mkl

cmake --build build -j 16
```

`CUDA_MATH_ROOT`, `CUDA_COMM_ROOT`, and `MKLROOT` may be exported instead.
Build and installed targets receive runpaths for the dependency locations
found by CMake. Those paths must be available on every compute node.

Do not add an entire Conda `lib` directory to `LD_LIBRARY_PATH` while using a
system `mpirun`; doing so can switch the process to a different Conda
`libmpi`. Build targets already record the precise directory of the selected
CUDA numerical libraries. If the current shell is contaminated, launch with
`env -u LD_LIBRARY_PATH mpirun ...`.

Install the library and its CMake package with:

```bash
cmake --install build --prefix /path/to/install
```

A downstream CMake project can then use:

```cmake
find_package(GEVD4ISC26 CONFIG REQUIRED)
target_link_libraries(my_solver PRIVATE GEVD4ISC26::gevd4isc26)
```

If CUDA is installed outside a standard toolkit location, also pass
`-DCUDAToolkit_ROOT=/path/to/cuda` when configuring the downstream project.

## Executables

| Executable | Scope |
|---|---|
| `gevd_solve` | Complete `A,B -> Q,Lambda` solve using binary matrix files |
| `gevd_validate` | Complete GEVD with exact eigenvalues and full validation |
| `gevd_benchmark` | Complete scalable GEVD with distributed generation and sampled validation |
| `evd_benchmark` | `C -> Y,Lambda` standard-EVD benchmark for SBR/BC optimization |
| `d_and_c_benchmark` | CPU tridiagonal D&C benchmark with no CUDA/MPI dependency |

### Full correctness demo

```bash
mpirun -np 1 ./build/bin/gevd_validate \
  --n 256 --matrix-block 32 \
  --evd-block 32 --evd-panel 256 \
  --repeat 2 --timing
```

The demo checks the maximum eigenvalue error, the relative residual

```text
||A Q - B Q Lambda||_F / ||B Q Lambda||_F
```

and the normalized `B`-orthogonality error

```text
||Q^T B Q - I||_F / sqrt(n).
```

### Scalable complete-GEVD benchmark

This benchmark generates both `A` and SPD `B` directly in each rank's local
block-cyclic shard. Its reported `complete_gevd_seconds` covers the full
`solve()` call, including Cholesky, reduction, EVD, and generalized
backtransform.

```bash
OMP_NUM_THREADS=128 MKL_NUM_THREADS=128 MKL_DYNAMIC=FALSE \
OMP_PROC_BIND=FALSE \
mpirun -np 4 --bind-to none ./build/bin/gevd_benchmark \
  --n 32000 --grid-rows 1 --matrix-block 1024 \
  --evd-block 32 --evd-panel 1024 \
  --validation-vectors 3 --timing
```

`--matrix-block` controls the outer cuSOLVERMp/cuBLASMp 2D block-cyclic
layout; it is independent of the standard-EVD kernel's `--evd-block`. On H100
with cuBLASMp 0.5, smaller `matrix-block` values make the backtransform TRSM
dominated by panel scheduling and communication synchronization. The
benchmark therefore uses 1024, validated for the dual-GPU `n=32000` case.

Use `--direct-backtransform` for the lower-memory path,
`--stage-pipeline` for SBR-to-BC overlap, and `--repeat N` to exercise factor
reuse.

Do not combine `OMP_PLACES=cores` with `OMP_PROC_BIND=close` for these GPU
benchmarks. GNU OpenMP first pins the main thread to one physical core, and
the CUDA, NCCL, and CPU D&C worker threads created later inherit that
overly narrow CPU affinity. The `OMP_PROC_BIND=FALSE` setting above still
respects the CPU set assigned to the process by the job scheduler or MPI; it
only prevents the process's threads from collapsing onto one OpenMP place.

Some NCCL, cuBLASMp, or NVSHMEM initialization paths further narrow the
calling thread to the GPU-local NUMA node. On every rank, the solver
dynamically snapshots the original CPU mask granted by the launcher/cgroup
and restores the calling thread after communication-resource initialization
and immediately before CPU D&C starts. No CPU IDs or node counts are
hard-coded, so this also covers multi-node, multi-GPU jobs with heterogeneous
CPU topologies. Vendor communication proxy threads retain their GPU-local
placement.

The tridiagonal D&C solve currently runs on the CPU of global rank zero. If
MPI or the scheduler assigns rank zero a narrow cpuset, the solver respects
that allocation and never expands beyond the job boundary. Large jobs should
therefore allocate enough CPU cores to rank zero.

### Standard-EVD stage benchmark

`evd_benchmark` starts from a symmetric standard problem. It measures SBR/BC
pipeline changes rather than complete GEVD time.

```bash
OMP_NUM_THREADS=128 MKL_NUM_THREADS=128 MKL_DYNAMIC=FALSE \
OMP_PROC_BIND=FALSE \
mpirun -np 2 --bind-to none ./build/bin/evd_benchmark \
  --n 32000 --evd-block 32 --evd-panel 1024 \
  --mode both --validation-vectors 3
```

`--mode both` runs baseline ordering and stage-pipeline ordering, compares all
eigenvalues, validates sampled residuals/orthogonality, and reports both the
SBR+BC critical-path ratio and the complete EVD-stage ratio.

### Standalone CPU D&C benchmark

`src/d_and_c/` is an independently configurable C++17/MKL subproject with no CUDA,
MPI, or NCCL dependency. The complete EVD engine starts it through
`AsyncTridiagonalSolve`, retaining overlap between CPU D&C and the GPU
eigenvector backtransforms. Build and test only this module with:

```bash
cmake -S src/d_and_c -B build-d-and-c \
  -DCMAKE_BUILD_TYPE=Release \
  -DMKL_ROOT=/path/to/mkl
cmake --build build-d-and-c -j

OMP_NUM_THREADS=128 MKL_NUM_THREADS=128 MKL_DYNAMIC=FALSE \
OMP_PROC_BIND=FALSE \
./build-d-and-c/bin/d_and_c_benchmark \
  --n 32000 --repeat 2 --validation-vectors 3
```

`prepare_seconds` covers host eigenvector-buffer acquisition, first touch,
and worker launch. `solve_seconds` measures only `DSTEDC`; `wall_seconds`
covers both. The complete project also exposes the focused
`gevd4isc26_d_and_c` and `d_and_c_benchmark` targets. See
[`src/d_and_c/README.md`](src/d_and_c/README.md) for tuning controls.

### Matrix-file solve

`A.bin` and `B.bin` contain `n*n` native doubles in column-major order. Rank
zero performs full-file I/O and distributes the matrices.

```bash
mpirun -np 4 ./build/bin/gevd_solve \
  --n 32000 \
  --a A.bin --b B.bin \
  --q Q.bin --lambda Lambda.bin \
  --grid-rows 1 \
  --row-block 32 --col-block 32 \
  --evd-block 32 --evd-panel 1024 \
  --lambda-format diagonal --timing
```

`Q.bin` is a full column-major matrix. `Lambda.bin` stores its `n` diagonal
values by default; `--lambda-format dense` writes an explicit diagonal matrix.

## Library API

```cpp
#include <gevd4isc26/gevd.hpp>

gevd4isc26::ProcessGrid grid{1, mpi_size, 0, 0};
gevd4isc26::MatrixDistribution layout(n, block, block, grid, mpi_rank);

gevd4isc26::DistributedMatrix A(layout);
gevd4isc26::DistributedMatrix B(layout);
// Fill this rank's local column-major block-cyclic shards.

gevd4isc26::SolverOptions options;
options.evd_block_size = 32;
options.evd_panel_size = 1024;
options.stage_pipeline = gevd4isc26::StagePipelineMode::Automatic;

gevd4isc26::GeneralizedEigensolver solver(MPI_COMM_WORLD, grid, options);
auto result = solver.solve(A.view(), B.view());

const gevd4isc26::DistributedMatrix& Q = result.Q;
const std::vector<double>& lambda = result.Lambda.diagonal();
```

`Q` has the same distribution as `A` and `B`. The eigenvalue diagonal is
replicated on every rank.

## SBR-to-BC stage pipeline

The optional pipeline overlaps a dependency-safe BC prefix on rank zero with
the remaining SBR suffix:

1. SBR advances to an `evd_panel`-aligned release frontier.
2. Rank zero starts a bounded BC sweep batch.
3. Other ranks finish the SBR suffix on NCCL suffix communicators.
4. MPI joins the stages; the completed band suffix and BC-modified prefix are
   merged.
5. Existing distributed BC ownership resumes from the saved sweep progress.

The path does not assume peer memory or a single node. MPI controls the global
join and NCCL uses global ranks, so the same control flow supports multi-node,
multi-GPU execution.

Enable automatic planning through `StagePipelineMode::Automatic` or
`--stage-pipeline`. `stage_pipeline_sweeps` and
`--stage-pipeline-sweeps` are expert overrides; zero retains automatic
selection.

## Standard-EVD constraints

Let `b = evd_block`, `nb = evd_panel`, and `p` be the MPI rank count:

```text
p is a power of two (1, 2, 4, 8, 16, ...)
n % b == 0
nb % b == 0
nb / b is a power of two
n / b >= p
min(n, nb) <= floor((n / b) / p) * b
```

The solver validates these conditions before launching the EVD kernels.

## Tuning and diagnostics

BC-back provides these compile-time tuning options:

- `EVD_BC_BACK_U_COUNT`;
- `EVD_BC_BACK_MAX_WARP_COUNT`;
- `EVD_BC_BACK_U_COL_EXTERN_COUNT`.

Their defaults are 8, 24, and 90. One shared configuration header defines the
reflector layout for both the BC producer and backtransform consumer, ensuring
a consistent packed layout.
`EVD_BC_BACK_U_COUNT` must be a multiple of four that divides 32.

Fine-grained diagnostics are off by default:

- `EVD_DEBUG_STAGE_PROGRESS=1`;
- `EVD_PRINT_STAGE_STATS=1`;
- `EVD_PROFILE_PANEL_QR=1`;
- `EVD_PROFILE_BC_BACK=1`.

These modes may synchronize kernels and should not be used for production
timing. `--timing` prints the regular complete GEVD timing tree.
