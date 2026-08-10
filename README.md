# GEVD4ISC26

**中文** | [English](README_EN.md)

GEVD4ISC26 是一个独立的、分布式的实对称-正定广义特征值求解器，用于求解

```text
A Q = B Q Lambda
Q^T B Q = I
```

输入为对称矩阵 `A` 和对称正定矩阵 `B`，输出为分布式特征向量矩阵 `Q`
以及在所有进程上复制的特征值矩阵 `Lambda` 的对角线。每个 MPI 进程控制
一张 NVIDIA GPU，矩阵在进程间采用二维块循环分布。

求解器覆盖 Cholesky 分解、广义问题标准化、分布式标准特征值分解和广义
特征向量回代的完整数值流程，并提供 C++ 库接口、矩阵文件求解程序、正确性
验证程序和性能基准。

## 来源与致谢

本仓库 fork 自 [hansheng1001/EVD4SC2025](https://github.com/hansheng1001/EVD4SC2025)
（SC'25 论文 *Rethinking Back Transformation in 2-Stage Eigenvalue Decomposition on
Heterogeneous Architectures* 的开源实现）。我们在其基础上复现了该团队的后续论文
[*Pipelined Dense Symmetric Eigenvalue Decomposition on Multi-GPU
Architectures*](https://arxiv.org/abs/2511.16174)（Wang et al., arXiv:2511.16174）
中的流水线两阶段 EVD 算法，并将其扩展为完整的分布式广义特征值求解器，
用于 ISC26 学生集群竞赛的 DFTB+ 赛题。

**衷心感谢 EVD4SC2025 团队（Hansheng Wang 等）的出色工作与开源贡献——
正是在他们的算法与代码基础上，我们在 ISC26 中取得了 DFTB+ 赛题第一名的成绩。**

EVD 引擎部分沿用上游的 MIT 许可证（见 [`src/evd/LICENSE`](src/evd/LICENSE)）。

## 数值流程

`GeneralizedEigensolver::solve(A, B)` 依次执行以下阶段：

1. Cholesky 分解：`B = L L^T`。
2. 广义问题标准化：`C = L^{-1} A L^{-T}`。
3. 将公开 API 使用的二维块循环布局重分布为 EVD 引擎使用的块对齐列布局。
4. 求解标准对称特征值问题 `C Y = Y Lambda`：
   - 稠密矩阵到带状矩阵约化（SBR）；
   - 带状矩阵到三对角矩阵约化（BC）；
   - 三对角矩阵分治求解；
   - SBR 特征向量回代；
   - BC 特征向量回代；
   - 特征向量合成 GEMM。
5. 将 `Y` 重分布回公开 API 的矩阵布局。
6. 广义特征向量回代：`Q = L^{-T}Y`。

默认回代路径在 EVD 特征向量回代期间并行准备 `L^{-T}`。内存较小时，可通过
`BacktransformMethod::TriangularSolve` 选择直接求解 `L^T Q = Y` 的路径。

同一个 `GeneralizedEigensolver` 对象在连续使用相同 `B` 时，可以由所有进程
共同决定复用分布式 Cholesky 因子和逆矩阵。主机端输入矩阵不会被修改，求解器
只读取 `A` 和 `B` 的下三角部分。

## 仓库结构

```text
.
├── app/
│   ├── common/              公共问题生成与结果验证
│   ├── gevd_solve.cpp       二进制矩阵文件求解接口
│   ├── gevd_validate.cpp        小规模完整 GEVD 正确性验证
│   ├── gevd_benchmark.cpp   可扩展的完整 GEVD 基准
│   └── evd_benchmark.cu     标准 EVD 性能与调优基准
├── include/gevd4isc26/
│   ├── gevd.hpp             公共 API 统一入口
│   ├── solver.hpp           GEVD 求解器和结果接口
│   ├── distributed_matrix.hpp
│   └── matrix_io.hpp
├── src/
│   ├── gevd/                可读的完整 GEVD 流程编排
│   ├── evd/                 优化后的分布式标准 EVD 引擎
│   ├── d_and_c/             独立可构建的 CPU 三对角分治库与基准
│   ├── distributed/         进程网格、矩阵布局和分布式上下文
│   ├── runtime/             MPI/GPU 资源、描述符和内存管理
│   ├── diagnostics/         计时与集合式正确性检查
│   ├── io/                  矩阵文件接口实现
│   └── cmake/               依赖发现与安装包配置
├── CMakeLists.txt
├── README.md
└── README_EN.md
```

GEVD 层按数值阶段组织：

```text
src/gevd/pipeline.cu             A,B -> Q,Lambda 的阶段顺序
src/gevd/workspace.*             单次求解所需的分布式存储
src/gevd/cholesky.*              B = L L^T
src/gevd/standard_reduction.*    C = L^-1 A L^-T
src/evd/adapter.*                GEVD 与 EVD 之间的布局转换边界
src/gevd/backtransform.*         Q = L^-T Y 或 L^T Q = Y
src/gevd/factor_cache.*          B 相关因子的集合式复用
src/d_and_c/d_and_c_solver.*             CPU 三对角 D&C、异步任务和主机缓存
```

EVD 引擎按职责拆分：

```text
src/evd/engine/          阶段驱动和共享类型
src/evd/stages/          SBR、BC、三对角提取和特征向量回代
src/evd/communication/   MPI/NCCL 布局转换与带状矩阵通信
src/evd/kernels/         CUDA 数值内核
src/evd/runtime/         EVD 资源和错误处理
src/evd/tuning/          运行时与编译期调优策略
src/evd/diagnostics/     可选的正确性检查和插桩分析
```

## 依赖要求

- CMake 3.24 或更高版本；
- 支持 C++17 的 C++ 编译器和 CUDA 编译器；
- MPI 开发头文件和运行时；
- CUDA Runtime、cuBLAS、cuSOLVER 和 cuRAND；
- cuSOLVERMp 和 cuBLASMp；
- NCCL 和 NVSHMEM；
- Intel MKL ILP64，使用 GNU 线程层；
- 每个 MPI 进程可见一张 NVIDIA GPU。

CUDA、MPI、NCCL、NVSHMEM、cuSOLVERMp、cuBLASMp 和 MKL 必须在 ABI 上兼容。
多机运行还需要正确配置 NCCL 网络传输。

## 原生构建

所有生成文件均放在 `build/` 下，该目录已被 Git 忽略。

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

也可以通过环境变量 `CUDA_MATH_ROOT`、`CUDA_COMM_ROOT` 和 `MKLROOT`
提供依赖根目录。构建及安装目标都会记录 CMake 找到的依赖路径，这些路径
必须在每个计算节点上可用。

不要在使用系统 `mpirun` 时把整个 Conda `lib` 目录加入
`LD_LIBRARY_PATH`，否则可能把程序切换到 Conda 中另一个版本的
`libmpi`。构建目标已经记录所选 CUDA 数值库的精确目录；若当前 shell
中存在污染，可用 `env -u LD_LIBRARY_PATH mpirun ...` 启动。

安装库、可执行程序和 CMake package：

```bash
cmake --install build --prefix /path/to/install
```

下游 CMake 项目可以这样使用：

```cmake
find_package(GEVD4ISC26 CONFIG REQUIRED)
target_link_libraries(my_solver PRIVATE GEVD4ISC26::gevd4isc26)
```

如果 CUDA 安装在非标准路径，下游项目配置时还需要传入
`-DCUDAToolkit_ROOT=/path/to/cuda`。

## 可执行程序

| 可执行程序 | 用途 |
|---|---|
| `gevd_solve` | 从二进制矩阵文件执行完整的 `A,B -> Q,Lambda` 求解 |
| `gevd_validate` | 使用精确特征值进行完整 GEVD 正确性验证 |
| `gevd_benchmark` | 分布式生成问题并采样验证的完整 GEVD 基准 |
| `evd_benchmark` | 用于分析 SBR/BC 优化的 `C -> Y,Lambda` 标准 EVD 基准 |
| `d_and_c_benchmark` | 不依赖 CUDA/MPI 的 CPU 三对角 D&C 基准 |

### 完整正确性验证

```bash
mpirun -np 1 ./build/bin/gevd_validate \
  --n 256 --matrix-block 32 \
  --evd-block 32 --evd-panel 256 \
  --repeat 2 --timing
```

该程序检查最大特征值误差、相对残差

```text
||A Q - B Q Lambda||_F / ||B Q Lambda||_F
```

以及归一化的 `B`-正交误差

```text
||Q^T B Q - I||_F / sqrt(n).
```

### 可扩展的完整 GEVD 基准

该基准直接在每个进程的局部块循环分片中生成 `A` 和对称正定矩阵 `B`。
输出的 `complete_gevd_seconds` 覆盖完整的 `solve()` 调用，包括 Cholesky
分解、广义问题标准化、EVD 和广义特征向量回代。

```bash
OMP_NUM_THREADS=128 MKL_NUM_THREADS=128 MKL_DYNAMIC=FALSE \
OMP_PROC_BIND=FALSE \
mpirun -np 4 --bind-to none ./build/bin/gevd_benchmark \
  --n 32000 --grid-rows 1 --matrix-block 1024 \
  --evd-block 32 --evd-panel 1024 \
  --validation-vectors 3 --timing
```

`--matrix-block` 控制外层 cuSOLVERMp/cuBLASMp 的二维块循环分布，与
标准 EVD 内核的 `--evd-block` 不同。在 H100 和 cuBLASMp 0.5 上，
较小的 `matrix-block` 会使回代 TRSM 严重受面板调度和通信同步开销限制；
1024 是该基准在 `n=32000` 双卡配置上经过验证的默认值。

使用 `--direct-backtransform` 可选择低内存回代路径，使用
`--stage-pipeline` 可开启 SBR 到 BC 的流水线，使用 `--repeat N`
可验证因子缓存。

不要为这些 GPU 基准同时设置 `OMP_PLACES=cores` 和
`OMP_PROC_BIND=close`。GNU OpenMP 会先把主线程绑定到一个物理核，
随后创建的 CUDA、NCCL 和 CPU D&C 工作线程会继承过窄的 CPU
亲和性。上面的 `OMP_PROC_BIND=FALSE` 仍会遵守作业调度器或 MPI
为进程分配的 CPU 集合，只是不再把进程内部线程缩到单个 OpenMP
place。

部分 NCCL、cuBLASMp 或 NVSHMEM 初始化路径会把调用线程进一步收窄到
GPU 所在的 NUMA 节点。求解器会在每个 rank 上动态保存 launcher/cgroup
授予的原始 CPU mask，并在通信资源初始化之后以及启动 CPU D&C 之前恢复
当前调用线程。这个过程不写死 CPU 编号或节点数量，因此同样适用于多机、
多卡和节点 CPU 拓扑不一致的作业；第三方通信 proxy 线程仍保留其
GPU-local 绑定。

三对角 D&C 目前由全局 rank 0 的 CPU 执行。若 MPI 或作业调度器只给
rank 0 分配很窄的 cpuset，求解器会严格遵守该集合，不会越过作业资源边界；
大规模任务应为 rank 0 分配足够的 CPU 核。

### 标准 EVD 阶段基准

`evd_benchmark` 从一个对称标准特征值问题开始，用于测量 SBR/BC
流水线变化，而不是测量完整 GEVD 时间。

```bash
OMP_NUM_THREADS=128 MKL_NUM_THREADS=128 MKL_DYNAMIC=FALSE \
OMP_PROC_BIND=FALSE \
mpirun -np 2 --bind-to none ./build/bin/evd_benchmark \
  --n 32000 --evd-block 32 --evd-panel 1024 \
  --mode both --validation-vectors 3
```

`--mode both` 依次运行基线顺序和阶段流水线顺序，比较全部特征值，采样验证
残差和正交性，并分别报告 SBR+BC 关键路径加速比和完整 EVD 阶段加速比。

### 独立 CPU D&C 基准

`src/d_and_c/` 是可独立配置的 C++17/MKL 子工程，不依赖 CUDA、MPI 或 NCCL。
完整 EVD 引擎通过 `AsyncTridiagonalSolve` 启动它，使 CPU D&C 与 GPU
特征向量回代继续重叠。只编译和测试这个模块：

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

输出中的 `prepare_seconds` 是特征向量主机缓冲区获取、首次触页和线程启动
时间，`solve_seconds` 只测量 `DSTEDC`，`wall_seconds` 覆盖两者。完整工程
也可直接构建 `gevd4isc26_d_and_c` 和 `d_and_c_benchmark` 两个目标。
更多调优开关见 [`src/d_and_c/README.md`](src/d_and_c/README.md)。

### 矩阵文件求解

`A.bin` 和 `B.bin` 均包含 `n*n` 个按列主序存储的本机 IEEE-754
双精度数。只有 0 号进程读写完整文件，随后由程序完成矩阵分发和收集。

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

`Q.bin` 是完整的列主序矩阵。默认情况下，`Lambda.bin` 只保存 `n` 个
对角元素；`--lambda-format dense` 会写出显式的 `n*n` 对角矩阵。

## 库接口

```cpp
#include <gevd4isc26/gevd.hpp>

gevd4isc26::ProcessGrid grid{1, mpi_size, 0, 0};
gevd4isc26::MatrixDistribution layout(n, block, block, grid, mpi_rank);

gevd4isc26::DistributedMatrix A(layout);
gevd4isc26::DistributedMatrix B(layout);
// 填充当前进程持有的列主序块循环分片。

gevd4isc26::SolverOptions options;
options.evd_block_size = 32;
options.evd_panel_size = 1024;
options.stage_pipeline = gevd4isc26::StagePipelineMode::Automatic;

gevd4isc26::GeneralizedEigensolver solver(MPI_COMM_WORLD, grid, options);
auto result = solver.solve(A.view(), B.view());

const gevd4isc26::DistributedMatrix& Q = result.Q;
const std::vector<double>& lambda = result.Lambda.diagonal();
```

`Q` 的分布方式与 `A`、`B` 相同，特征值对角线在每个 MPI 进程上均有一份。

## SBR 到 BC 的阶段流水线

可选的阶段流水线在 0 号进程上执行满足依赖关系的 BC 前缀，同时让其他进程
继续完成剩余的 SBR 后缀：

1. SBR 推进到与 `evd_panel` 对齐的释放前沿。
2. 0 号进程开始执行一个有界的 BC sweep 批次。
3. 其他进程通过 NCCL 后缀通信器完成 SBR 后缀。
4. MPI 完成全局汇合，并合并完整的带状矩阵后缀与 BC 已修改的前缀。
5. 原有的分布式 BC 所有权划分从已保存的 sweep 进度继续执行。

该路径不依赖 GPU peer memory，也不假设所有进程位于同一台机器。MPI 控制
全局汇合，NCCL 使用全局进程编号，因此同一套控制流程支持多机多卡运行。

可以通过 `StagePipelineMode::Automatic` 或 `--stage-pipeline` 开启自动
规划。`stage_pipeline_sweeps` 和 `--stage-pipeline-sweeps` 是专家调优
选项，取值为零时由程序自动选择。

## 标准 EVD 约束

令 `b = evd_block`、`nb = evd_panel`，MPI 进程数为 `p`：

```text
p is a power of two (1, 2, 4, 8, 16, ...)
n % b == 0
nb % b == 0
nb / b is a power of two
n / b >= p
min(n, nb) <= floor((n / b) / p) * b
```

求解器会在启动 EVD CUDA 内核之前检查这些约束。

## 调优与诊断

BC 回代提供以下编译期调优选项：

- `EVD_BC_BACK_U_COUNT`；
- `EVD_BC_BACK_MAX_WARP_COUNT`；
- `EVD_BC_BACK_U_COL_EXTERN_COUNT`。

默认值分别为 8、24 和 90。BC reflector 的生产端和回代消费端使用同一个
配置头文件，以保证 packed 布局一致。
`EVD_BC_BACK_U_COUNT` 必须是 4 的倍数并且能够整除 32。

细粒度诊断默认关闭：

- `EVD_DEBUG_STAGE_PROGRESS=1`；
- `EVD_PRINT_STAGE_STATS=1`；
- `EVD_PROFILE_PANEL_QR=1`；
- `EVD_PROFILE_BC_BACK=1`。

这些模式可能同步 CUDA 内核，不应当用于正式性能测试。常规的 `--timing`
会输出完整 GEVD 流程的统一计时树。
