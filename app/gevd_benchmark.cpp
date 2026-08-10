#include "common/arguments.hpp"
#include "common/distributed_problem.hpp"
#include "gevd4isc26/solver.hpp"

#include <mpi.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace {

struct BenchmarkOptions {
  std::int64_t order = 4096;
  int grid_rows = 1;
  std::int64_t matrix_block = 1024;
  int evd_block = 32;
  int evd_panel = 1024;
  int repeats = 1;
  int validation_vectors = 3;
  int pipeline_sweeps = 0;
  bool stage_pipeline = false;
  bool direct_backtransform = false;
  bool timing = true;
};

void printUsage() {
  std::cout
      << "Usage: gevd_benchmark [--n N] [--grid-rows R] "
         "[--matrix-block B]\n"
         "                      [--evd-block B] [--evd-panel NB] "
         "[--repeat K]\n"
         "                      [--validation-vectors K] "
         "[--stage-pipeline]\n"
         "                      [--stage-pipeline-sweeps S] "
         "[--direct-backtransform]\n"
         "                      [--no-timing]\n";
}

[[nodiscard]] BenchmarkOptions parseOptions(int argc, char** argv) {
  BenchmarkOptions options;
  options.order =
      gevd4isc26::app::integerOption(argc, argv, "--n", options.order);
  options.grid_rows = static_cast<int>(gevd4isc26::app::integerOption(
      argc, argv, "--grid-rows", options.grid_rows));
  options.matrix_block = gevd4isc26::app::integerOption(
      argc, argv, "--matrix-block", options.matrix_block);
  options.evd_block = static_cast<int>(gevd4isc26::app::integerOption(
      argc, argv, "--evd-block", options.evd_block));
  options.evd_panel = static_cast<int>(gevd4isc26::app::integerOption(
      argc, argv, "--evd-panel", options.evd_panel));
  options.repeats = static_cast<int>(gevd4isc26::app::integerOption(
      argc, argv, "--repeat", options.repeats));
  options.validation_vectors =
      static_cast<int>(gevd4isc26::app::integerOption(
          argc, argv, "--validation-vectors", options.validation_vectors));
  options.pipeline_sweeps =
      static_cast<int>(gevd4isc26::app::integerOption(
          argc, argv, "--stage-pipeline-sweeps", 0));
  options.stage_pipeline =
      gevd4isc26::app::hasFlag(argc, argv, "--stage-pipeline");
  options.direct_backtransform =
      gevd4isc26::app::hasFlag(argc, argv, "--direct-backtransform");
  options.timing =
      !gevd4isc26::app::hasFlag(argc, argv, "--no-timing");
  if (options.order <= 0 || options.grid_rows <= 0 ||
      options.matrix_block <= 0 || options.evd_block <= 0 ||
      options.evd_panel <= 0 || options.repeats <= 0 ||
      options.validation_vectors <= 0 || options.pipeline_sweeps < 0) {
    throw std::invalid_argument(
        "matrix, grid, EVD, repeat, and validation values must be positive");
  }
  return options;
}

int run(int argc, char** argv) {
  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (gevd4isc26::app::hasFlag(argc, argv, "--help") ||
      gevd4isc26::app::hasFlag(argc, argv, "-h")) {
    if (rank == 0) {
      printUsage();
    }
    return 0;
  }

  const BenchmarkOptions command = parseOptions(argc, argv);
  if (ranks % command.grid_rows != 0) {
    throw std::invalid_argument(
        "rank count must be divisible by --grid-rows");
  }

  const gevd4isc26::ProcessGrid grid{
      command.grid_rows, ranks / command.grid_rows, 0, 0};
  const gevd4isc26::MatrixDistribution distribution(
      command.order, command.matrix_block, command.matrix_block, grid, rank);
  gevd4isc26::DistributedMatrix A(distribution);
  gevd4isc26::DistributedMatrix B(distribution);
  gevd4isc26::app::generateDistributedProblem(A, B);

  gevd4isc26::SolverOptions solver_options;
  solver_options.evd_block_size = command.evd_block;
  solver_options.evd_panel_size = command.evd_panel;
  solver_options.stage_pipeline =
      command.stage_pipeline ? gevd4isc26::StagePipelineMode::Automatic
                             : gevd4isc26::StagePipelineMode::Disabled;
  solver_options.stage_pipeline_sweeps = command.pipeline_sweeps;
  solver_options.backtransform =
      command.direct_backtransform
          ? gevd4isc26::BacktransformMethod::TriangularSolve
          : gevd4isc26::BacktransformMethod::OverlappedInverse;
  solver_options.print_timing = command.timing;

  if (rank == 0) {
    std::cout << "Complete GEVD benchmark n=" << command.order
              << " ranks=" << ranks << " grid=" << grid.rows << 'x'
              << grid.columns << " matrix_block=" << command.matrix_block
              << " evd_block=" << command.evd_block
              << " evd_panel=" << command.evd_panel << std::endl;
  }
  gevd4isc26::GeneralizedEigensolver solver(
      MPI_COMM_WORLD, grid, solver_options);
  std::optional<gevd4isc26::GeneralizedEigenResult> result;
  for (int iteration = 0; iteration < command.repeats; ++iteration) {
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    result.emplace(solver.solve(A.view(), B.view()));
    const double local_seconds = MPI_Wtime() - start;
    double maximum_seconds = 0.0;
    MPI_Allreduce(&local_seconds, &maximum_seconds, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << std::setprecision(12)
                << "complete_gevd_seconds=" << maximum_seconds
                << " iteration=" << iteration + 1 << std::endl;
    }
  }

  const gevd4isc26::app::SampledValidation validation =
      gevd4isc26::app::validateDistributedResult(
          A.view(), B.view(), *result, command.validation_vectors,
          MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "sampled_relative_residual="
              << validation.maximum_relative_residual
              << " sampled_B_orthogonality="
              << validation.maximum_B_orthogonality_error << std::endl;
  }
  return validation.passed() ? 0 : 2;
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
