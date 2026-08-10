#include "distributed_evd.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define CUBLAS_CHECK(call) checkCublas((call), #call, __FILE__, __LINE__)
#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)

void checkCuda(cudaError_t status, const char* call, const char* file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(file) + ':' + std::to_string(line) + " " +
                             call + ": " + cudaGetErrorString(status));
  }
}

void checkCublas(cublasStatus_t status, const char* call, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(file) + ':' + std::to_string(line) + " " +
                             call + ": cuBLAS status " + std::to_string(status));
  }
}

void checkMpi(int status, const char* call, const char* file, int line) {
  if (status != MPI_SUCCESS) {
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(std::string(file) + ':' + std::to_string(line) + " " +
                             call + ": " + std::string(message, length));
  }
}

struct Options {
  long n = 32000;
  int blockSize = 32;
  int panelSize = 1024;
  std::string mode = "both";
  int pipelineSweeps = 0;
  int validationVectors = 3;
};

long parseLong(const char* text, const char* option) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0') {
    throw std::invalid_argument(std::string("invalid value for ") + option);
  }
  return value;
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() {
      if (++index >= argc) {
        throw std::invalid_argument("missing value after " + argument);
      }
      return argv[index];
    };
    if (argument == "--n") {
      options.n = parseLong(value(), "--n");
    } else if (argument == "--evd-block") {
      options.blockSize = static_cast<int>(parseLong(value(), "--evd-block"));
    } else if (argument == "--evd-panel") {
      options.panelSize = static_cast<int>(parseLong(value(), "--evd-panel"));
    } else if (argument == "--mode") {
      options.mode = value();
    } else if (argument == "--pipeline-sweeps") {
      options.pipelineSweeps =
          static_cast<int>(parseLong(value(), "--pipeline-sweeps"));
    } else if (argument == "--validation-vectors") {
      options.validationVectors =
          static_cast<int>(parseLong(value(), "--validation-vectors"));
    } else if (argument == "--help" || argument == "-h") {
      options.mode = "help";
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (options.mode != "baseline" && options.mode != "pipeline" &&
      options.mode != "both" && options.mode != "help") {
    throw std::invalid_argument("--mode must be baseline, pipeline, or both");
  }
  if (options.n <= 0 || options.blockSize <= 0 || options.panelSize <= 0 ||
      options.validationVectors <= 0) {
    throw std::invalid_argument("matrix and tuning values must be positive");
  }
  return options;
}

struct ColumnDistribution {
  std::vector<long> counts;
  std::vector<long> offsets;
};

ColumnDistribution makeColumnDistribution(long n, int blockSize, int ranks) {
  if (n % blockSize != 0) {
    throw std::invalid_argument("n must be divisible by the EVD block size");
  }
  const long blocks = n / blockSize;
  ColumnDistribution result;
  result.counts.assign(static_cast<size_t>(ranks), 0);
  result.offsets.assign(static_cast<size_t>(ranks), 0);
  for (int rank = 0; rank < ranks; ++rank) {
    result.counts[static_cast<size_t>(rank)] =
        (blocks / ranks + (rank < blocks % ranks ? 1 : 0)) * blockSize;
    if (rank > 0) {
      result.offsets[static_cast<size_t>(rank)] =
          result.offsets[static_cast<size_t>(rank - 1)] +
          result.counts[static_cast<size_t>(rank - 1)];
    }
  }
  return result;
}

__device__ std::uint64_t mixBits(std::uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__global__ void generateSymmetricMatrix(double* matrix,
                                        long n,
                                        long columnStart,
                                        long localColumns) {
  const long row = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long localColumn =
      static_cast<long>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= n || localColumn >= localColumns) {
    return;
  }
  const long column = columnStart + localColumn;
  double value = 0.0;
  if (row == column) {
    value = 4.0 + static_cast<double>(row) / static_cast<double>(n);
  } else {
    const std::uint64_t low = static_cast<std::uint64_t>(min(row, column));
    const std::uint64_t high = static_cast<std::uint64_t>(max(row, column));
    const std::uint64_t bits =
        mixBits((low + 1) * 0x9e3779b97f4a7c15ULL ^
                (high + 1) * 0xd1b54a32d192ed03ULL);
    const double uniform =
        static_cast<double>(bits >> 11) * 0x1.0p-53 - 0.5;
    value = 0.2 * uniform / sqrt(static_cast<double>(n));
  }
  matrix[row + localColumn * n] = value;
}

void generateMatrix(double* matrix,
                    long n,
                    long columnStart,
                    long localColumns) {
  const dim3 threads(32, 8);
  const dim3 blocks((n + threads.x - 1) / threads.x,
                    (localColumns + threads.y - 1) / threads.y);
  generateSymmetricMatrix<<<blocks, threads>>>(matrix, n, columnStart, localColumns);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

int ownerOfColumn(long column, const ColumnDistribution& distribution) {
  for (int rank = 0; rank < static_cast<int>(distribution.counts.size()); ++rank) {
    if (column >= distribution.offsets[static_cast<size_t>(rank)] &&
        column < distribution.offsets[static_cast<size_t>(rank)] +
                     distribution.counts[static_cast<size_t>(rank)]) {
      return rank;
    }
  }
  return static_cast<int>(distribution.counts.size()) - 1;
}

std::vector<long> validationIndices(long n, int count) {
  std::vector<long> indices;
  indices.reserve(static_cast<size_t>(count));
  for (int index = 0; index < count; ++index) {
    const long column = count == 1
                            ? n / 2
                            : (static_cast<long long>(index) * (n - 1)) /
                                  (count - 1);
    if (indices.empty() || indices.back() != column) {
      indices.push_back(column);
    }
  }
  return indices;
}

std::vector<double> fetchEigenvector(const double* eigenvectors,
                                     long n,
                                     long columnStart,
                                     long localColumns,
                                     long globalColumn,
                                     int owner,
                                     int rank,
                                     MPI_Comm communicator) {
  std::vector<double> vector(static_cast<size_t>(n), 0.0);
  if (rank == owner) {
    const long localColumn = globalColumn - columnStart;
    if (localColumn < 0 || localColumn >= localColumns) {
      throw std::runtime_error("eigenvector owner does not contain requested column");
    }
    CUDA_CHECK(cudaMemcpy(vector.data(),
                          eigenvectors + localColumn * n,
                          sizeof(double) * vector.size(),
                          cudaMemcpyDeviceToHost));
  }
  MPI_CHECK(MPI_Bcast(vector.data(),
                      static_cast<int>(vector.size()),
                      MPI_DOUBLE,
                      owner,
                      communicator));
  return vector;
}

double dot(const std::vector<double>& left, const std::vector<double>& right) {
  long double sum = 0.0;
  for (size_t index = 0; index < left.size(); ++index) {
    sum += static_cast<long double>(left[index]) * right[index];
  }
  return static_cast<double>(sum);
}

struct Validation {
  double maximumResidual = 0.0;
  double maximumOrthogonalityError = 0.0;
};

Validation validateEigenpairs(const double* eigenvectors,
                              const std::vector<double>& eigenvalues,
                              long n,
                              long columnStart,
                              long localColumns,
                              const ColumnDistribution& distribution,
                              int validationVectorCount,
                              int rank,
                              MPI_Comm communicator) {
  double* original = nullptr;
  double* localX = nullptr;
  double* localY = nullptr;
  CUDA_CHECK(cudaMalloc(&original,
                        sizeof(double) * static_cast<size_t>(n) * localColumns));
  CUDA_CHECK(cudaMalloc(&localX, sizeof(double) * static_cast<size_t>(localColumns)));
  CUDA_CHECK(cudaMalloc(&localY, sizeof(double) * static_cast<size_t>(n)));
  generateMatrix(original, n, columnStart, localColumns);

  cublasHandle_t blas = nullptr;
  CUBLAS_CHECK(cublasCreate(&blas));
  const double one = 1.0;
  const double zero = 0.0;
  const std::vector<long> indices = validationIndices(n, validationVectorCount);
  std::vector<std::vector<double>> vectors;
  vectors.reserve(indices.size());
  Validation validation;
  std::vector<double> partial(static_cast<size_t>(n));
  std::vector<double> product(static_cast<size_t>(n));

  for (long eigenvectorIndex : indices) {
    const int owner = ownerOfColumn(eigenvectorIndex, distribution);
    std::vector<double> vector = fetchEigenvector(eigenvectors,
                                                  n,
                                                  columnStart,
                                                  localColumns,
                                                  eigenvectorIndex,
                                                  owner,
                                                  rank,
                                                  communicator);
    CUDA_CHECK(cudaMemcpy(localX,
                          vector.data() + columnStart,
                          sizeof(double) * static_cast<size_t>(localColumns),
                          cudaMemcpyHostToDevice));
    CUBLAS_CHECK(cublasDgemv(blas,
                             CUBLAS_OP_N,
                             static_cast<int>(n),
                             static_cast<int>(localColumns),
                             &one,
                             original,
                             static_cast<int>(n),
                             localX,
                             1,
                             &zero,
                             localY,
                             1));
    CUDA_CHECK(cudaMemcpy(partial.data(),
                          localY,
                          sizeof(double) * partial.size(),
                          cudaMemcpyDeviceToHost));
    MPI_CHECK(MPI_Allreduce(partial.data(),
                            product.data(),
                            static_cast<int>(product.size()),
                            MPI_DOUBLE,
                            MPI_SUM,
                            communicator));

    long double residualSquared = 0.0;
    long double scaleSquared = 0.0;
    const double eigenvalue = eigenvalues[static_cast<size_t>(eigenvectorIndex)];
    for (long row = 0; row < n; ++row) {
      const double expected = eigenvalue * vector[static_cast<size_t>(row)];
      const double difference = product[static_cast<size_t>(row)] - expected;
      residualSquared += static_cast<long double>(difference) * difference;
      scaleSquared += static_cast<long double>(product[static_cast<size_t>(row)]) *
                      product[static_cast<size_t>(row)];
    }
    validation.maximumResidual = std::max(
        validation.maximumResidual,
        std::sqrt(static_cast<double>(residualSquared /
                                      std::max<long double>(scaleSquared, 1.0e-300L))));
    vectors.push_back(std::move(vector));
  }

  for (size_t row = 0; row < vectors.size(); ++row) {
    for (size_t column = 0; column < vectors.size(); ++column) {
      const double expected = row == column ? 1.0 : 0.0;
      validation.maximumOrthogonalityError = std::max(
          validation.maximumOrthogonalityError,
          std::abs(dot(vectors[row], vectors[column]) - expected));
    }
  }

  CUBLAS_CHECK(cublasDestroy(blas));
  CUDA_CHECK(cudaFree(localY));
  CUDA_CHECK(cudaFree(localX));
  CUDA_CHECK(cudaFree(original));
  return validation;
}

struct RunResult {
  double seconds = 0.0;
  gevd4isc26_standard_evd_timings_t stageTimings{};
  Validation validation;
  std::vector<double> eigenvalues;
};

RunResult runMode(bool pipeline,
                  const Options& options,
                  const ColumnDistribution& distribution,
                  int rank,
                  MPI_Comm communicator) {
  const long localColumns = distribution.counts[static_cast<size_t>(rank)];
  const long columnStart = distribution.offsets[static_cast<size_t>(rank)];
  double* matrix = nullptr;
  CUDA_CHECK(cudaMalloc(&matrix,
                        sizeof(double) * static_cast<size_t>(options.n) * localColumns));
  generateMatrix(matrix, options.n, columnStart, localColumns);
  RunResult result;
  result.eigenvalues.resize(static_cast<size_t>(options.n));

  MPI_CHECK(MPI_Barrier(communicator));
  const double start = MPI_Wtime();
  gevd4isc26_standard_evd_options_t stageOptions{};
  stageOptions.block_size = options.blockSize;
  stageOptions.panel_size = options.panelSize;
  stageOptions.enable_sbr_bc_pipeline = pipeline ? 1 : 0;
  stageOptions.pipeline_sweeps = options.pipelineSweeps;
  stageOptions.print_progress = 1;
  const int status = gevd4isc26_symmetric_evd_device(
      options.n,
      matrix,
      options.n,
      localColumns,
      result.eigenvalues.data(),
      matrix,
      options.n,
      localColumns,
      communicator,
      &stageOptions,
      nullptr,
      nullptr,
      &result.stageTimings);
  CUDA_CHECK(cudaDeviceSynchronize());
  MPI_CHECK(MPI_Barrier(communicator));
  const double localSeconds = MPI_Wtime() - start;
  MPI_CHECK(MPI_Allreduce(&localSeconds,
                          &result.seconds,
                          1,
                          MPI_DOUBLE,
                          MPI_MAX,
                          communicator));
  if (status != 0) {
    throw std::runtime_error("distributed EVD engine returned status " +
                             std::to_string(status));
  }

  result.validation = validateEigenpairs(matrix,
                                         result.eigenvalues,
                                         options.n,
                                         columnStart,
                                         localColumns,
                                         distribution,
                                         options.validationVectors,
                                         rank,
                                         communicator);
  CUDA_CHECK(cudaFree(matrix));
  return result;
}

void printResult(const char* label, const RunResult& result, int rank) {
  if (rank != 0) {
    return;
  }
  long double eigenvalueSum = 0.0;
  for (double value : result.eigenvalues) {
    eigenvalueSum += value;
  }
  std::cout << std::setprecision(12)
            << "EVD benchmark mode=" << label
            << " seconds=" << result.seconds
            << " sampled_residual=" << result.validation.maximumResidual
            << " sampled_orthogonality="
            << result.validation.maximumOrthogonalityError
            << " eigenvalue_sum=" << static_cast<double>(eigenvalueSum)
            << std::endl;
  const auto& timing = result.stageTimings;
  auto printStage = [](const char* stage, double milliseconds) {
    std::cout << "    " << std::left << std::setw(48) << stage << std::right
              << std::setw(14) << milliseconds / 1000.0 << " s\n";
  };
  std::cout << "  standard-stage timing (maximum across MPI ranks)\n";
  printStage("initialize distributed resources", timing.initialization_ms);
  printStage("complete symmetry / prepare workspace", timing.matrix_setup_ms);
  printStage("dense -> band reduction (SBR)", timing.dense_to_band_ms);
  printStage("band -> tridiagonal reduction (BC)",
             timing.band_to_tridiagonal_ms);
  printStage("SBR+BC critical path", timing.reduction_critical_path_ms);
  printStage("tridiagonal eigensolve", timing.tridiagonal_eigensolve_ms);
  printStage("apply SBR reflectors (SBR back)",
             timing.dense_backtransform_ms);
  printStage("apply BC reflectors (BC back)",
             timing.band_backtransform_ms);
  printStage("compose standard eigenvectors (GEMM)",
             timing.eigenvector_composition_ms);
  printStage("overlapped numerical critical path",
             timing.overlapped_critical_path_ms);
  printStage("numerical-core wall", timing.numerical_wall_ms);
  printStage("release distributed resources", timing.cleanup_ms);
  printStage("complete standard-stage engine call", timing.engine_call_ms);
}

int run(int argc, char** argv) {
  const Options options = parseOptions(argc, argv);
  int rank = 0;
  int ranks = 1;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
  if (options.mode == "help") {
    if (rank == 0) {
      std::cout << "Usage: evd_benchmark [--n N] [--evd-block B] "
                   "[--evd-panel NB] [--mode baseline|pipeline|both] "
                   "[--pipeline-sweeps S] [--validation-vectors K]\n";
    }
    return 0;
  }
  if (ranks != 1 && ranks != 2 && ranks != 4 && ranks != 8) {
    throw std::invalid_argument(
        "the distributed EVD engine supports 1, 2, 4, or 8 MPI ranks");
  }

  MPI_Comm localCommunicator = MPI_COMM_NULL;
  MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD,
                                MPI_COMM_TYPE_SHARED,
                                0,
                                MPI_INFO_NULL,
                                &localCommunicator));
  int localRank = 0;
  MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));
  MPI_CHECK(MPI_Comm_free(&localCommunicator));
  int devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&devices));
  if (devices <= 0) {
    throw std::runtime_error("no CUDA devices are visible");
  }
  CUDA_CHECK(cudaSetDevice(localRank % devices));

  const ColumnDistribution distribution =
      makeColumnDistribution(options.n, options.blockSize, ranks);
  if (rank == 0) {
    std::cout << "Distributed EVD benchmark n=" << options.n
              << " ranks=" << ranks
              << " b=" << options.blockSize
              << " nb=" << options.panelSize
              << std::endl;
  }

  RunResult baseline;
  RunResult pipeline;
  if (options.mode == "baseline" || options.mode == "both") {
    baseline = runMode(false, options, distribution, rank, MPI_COMM_WORLD);
    printResult("baseline", baseline, rank);
  }
  if (options.mode == "pipeline" || options.mode == "both") {
    pipeline = runMode(true, options, distribution, rank, MPI_COMM_WORLD);
    printResult("pipeline", pipeline, rank);
  }
  if (options.mode == "both") {
    double maximumEigenvalueDifference = 0.0;
    for (size_t index = 0; index < baseline.eigenvalues.size(); ++index) {
      maximumEigenvalueDifference = std::max(
          maximumEigenvalueDifference,
          std::abs(baseline.eigenvalues[index] - pipeline.eigenvalues[index]));
    }
    if (rank == 0) {
      std::cout << "standard_stage_speedup=" << baseline.seconds / pipeline.seconds
                << " max_eigenvalue_difference=" << maximumEigenvalueDifference
                << std::endl;
    }
    if (maximumEigenvalueDifference > 1.0e-8 ||
        pipeline.validation.maximumResidual > 1.0e-8 ||
        pipeline.validation.maximumOrthogonalityError > 1.0e-8) {
      return 2;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int result = 0;
  try {
    result = run(argc, argv);
  } catch (const std::exception& error) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "[rank " << rank << "] " << error.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    result = 1;
  }
  MPI_Finalize();
  return result;
}
