#include "common/arguments.hpp"
#include "gevd4isc26/matrix_io.hpp"
#include "gevd4isc26/solver.hpp"

#include <mpi.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CommandLine {
  std::int64_t order = 0;
  std::string path_A;
  std::string path_B;
  std::string path_Q;
  std::string path_Lambda;
  int grid_rows = 1;
  int grid_columns = 0;
  std::int64_t row_block_size = 32;
  std::int64_t column_block_size = 32;
  int evd_block_size = 32;
  int evd_panel_size = 1024;
  int first_gpu = 0;
  int stage_pipeline_sweeps = 0;
  bool direct_backtransform = false;
  bool stage_pipeline = false;
  bool timing = false;
  bool help = false;
  gevd4isc26::LambdaFileFormat lambda_format =
      gevd4isc26::LambdaFileFormat::Diagonal;
};

[[nodiscard]] std::string usageText() {
  return
      "Usage: gevd_solve --n N --a A.bin --b B.bin --q Q.bin --lambda Lambda.bin\n"
      "                  [--grid-rows R] [--grid-cols C]\n"
      "                  [--row-block MB] [--col-block NB]\n"
      "                  [--evd-block B] [--evd-panel NB]\n"
      "                  [--first-gpu ID] [--direct-backtransform]\n"
      "                  [--stage-pipeline [--stage-pipeline-sweeps S]]\n"
      "                  [--lambda-format diagonal|dense] [--timing]\n"
      "A.bin and B.bin are raw column-major n-by-n double matrices.\n";
}

[[noreturn]] void usageError(const std::string& message) {
  throw std::invalid_argument(message + "\n" + usageText());
}

[[nodiscard]] CommandLine parseArguments(int argc, char** argv) {
  CommandLine options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() -> std::string {
      if (++index >= argc) {
        usageError("missing value after " + argument);
      }
      return argv[index];
    };

    if (argument == "--n") {
      options.order = gevd4isc26::app::parseInteger(value(), "--n");
    } else if (argument == "--a") {
      options.path_A = value();
    } else if (argument == "--b") {
      options.path_B = value();
    } else if (argument == "--q") {
      options.path_Q = value();
    } else if (argument == "--lambda") {
      options.path_Lambda = value();
    } else if (argument == "--grid-rows") {
      options.grid_rows = static_cast<int>(
          gevd4isc26::app::parseInteger(value(), "--grid-rows"));
    } else if (argument == "--grid-cols") {
      options.grid_columns = static_cast<int>(
          gevd4isc26::app::parseInteger(value(), "--grid-cols"));
    } else if (argument == "--row-block") {
      options.row_block_size =
          gevd4isc26::app::parseInteger(value(), "--row-block");
    } else if (argument == "--col-block") {
      options.column_block_size =
          gevd4isc26::app::parseInteger(value(), "--col-block");
    } else if (argument == "--evd-block") {
      options.evd_block_size = static_cast<int>(
          gevd4isc26::app::parseInteger(value(), "--evd-block"));
    } else if (argument == "--evd-panel") {
      options.evd_panel_size = static_cast<int>(
          gevd4isc26::app::parseInteger(value(), "--evd-panel"));
    } else if (argument == "--first-gpu") {
      options.first_gpu = static_cast<int>(
          gevd4isc26::app::parseInteger(value(), "--first-gpu"));
    } else if (argument == "--stage-pipeline") {
      options.stage_pipeline = true;
    } else if (argument == "--stage-pipeline-sweeps") {
      options.stage_pipeline_sweeps =
          static_cast<int>(gevd4isc26::app::parseInteger(
              value(), "--stage-pipeline-sweeps"));
    } else if (argument == "--direct-backtransform") {
      options.direct_backtransform = true;
    } else if (argument == "--lambda-format") {
      const std::string format = value();
      if (format == "diagonal") {
        options.lambda_format = gevd4isc26::LambdaFileFormat::Diagonal;
      } else if (format == "dense") {
        options.lambda_format = gevd4isc26::LambdaFileFormat::Dense;
      } else {
        usageError("--lambda-format must be diagonal or dense");
      }
    } else if (argument == "--timing") {
      options.timing = true;
    } else if (argument == "--help" || argument == "-h") {
      options.help = true;
    } else {
      usageError("unknown option: " + argument);
    }
  }

  if (!options.help &&
      (options.order <= 0 || options.path_A.empty() || options.path_B.empty() ||
       options.path_Q.empty() || options.path_Lambda.empty())) {
    usageError("--n, --a, --b, --q, and --lambda are required");
  }
  return options;
}

void broadcastError(std::string* error, int rank, MPI_Comm communicator) {
  int length = rank == 0 ? static_cast<int>(error->size()) : 0;
  MPI_Bcast(&length, 1, MPI_INT, 0, communicator);
  if (rank != 0) {
    error->assign(static_cast<std::size_t>(length), '\0');
  }
  if (length != 0) {
    MPI_Bcast(error->data(), length, MPI_CHAR, 0, communicator);
    throw std::runtime_error(*error);
  }
}

[[nodiscard]] gevd4isc26::DistributedMatrix readAndScatter(
    const std::string& path,
    const gevd4isc26::MatrixDistribution& distribution,
    int rank,
    MPI_Comm communicator) {
  std::vector<double> full_matrix;
  std::string input_error;
  if (rank == 0) {
    try {
      full_matrix = gevd4isc26::readRawDenseMatrix(path, distribution.order());
    } catch (const std::exception& error) {
      input_error = error.what();
    }
  }
  broadcastError(&input_error, rank, communicator);
  return gevd4isc26::scatterDenseMatrix(full_matrix, distribution, communicator);
}

int run(int argc, char** argv) {
  const CommandLine command = parseArguments(argc, argv);
  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (command.help) {
    if (rank == 0) {
      std::cout << usageText();
    }
    return 0;
  }

  int grid_columns = command.grid_columns;
  if (grid_columns == 0) {
    if (command.grid_rows <= 0 || ranks % command.grid_rows != 0) {
      usageError("MPI rank count must be divisible by --grid-rows");
    }
    grid_columns = ranks / command.grid_rows;
  }
  const gevd4isc26::ProcessGrid grid{command.grid_rows, grid_columns, 0, 0};
  if (grid.size() != ranks) {
    usageError("--grid-rows * --grid-cols must equal the MPI rank count");
  }
  const gevd4isc26::MatrixDistribution distribution(
      command.order, command.row_block_size, command.column_block_size, grid, rank);

  // Read and release one full root-side matrix at a time. This halves the
  // command-line path's peak staging memory compared with loading both first.
  gevd4isc26::DistributedMatrix A =
      readAndScatter(command.path_A, distribution, rank, MPI_COMM_WORLD);
  gevd4isc26::DistributedMatrix B =
      readAndScatter(command.path_B, distribution, rank, MPI_COMM_WORLD);

  gevd4isc26::SolverOptions solver_options;
  solver_options.evd_block_size = command.evd_block_size;
  solver_options.evd_panel_size = command.evd_panel_size;
  solver_options.first_gpu = command.first_gpu;
  if (command.stage_pipeline) {
    solver_options.stage_pipeline = gevd4isc26::StagePipelineMode::Automatic;
    solver_options.stage_pipeline_sweeps = command.stage_pipeline_sweeps;
  }
  if (command.direct_backtransform) {
    solver_options.backtransform =
        gevd4isc26::BacktransformMethod::TriangularSolve;
  }
  solver_options.print_timing = command.timing;
  gevd4isc26::GeneralizedEigensolver solver(MPI_COMM_WORLD, grid, solver_options);
  gevd4isc26::GeneralizedEigenResult result = solver.solve(A.view(), B.view());
  std::vector<double> full_Q =
      gevd4isc26::gatherDenseMatrix(result.Q.view(), MPI_COMM_WORLD);

  std::string output_error;
  if (rank == 0) {
    try {
      gevd4isc26::writeRawDenseMatrix(command.path_Q, full_Q);
      gevd4isc26::writeLambda(command.path_Lambda, result.Lambda, command.lambda_format);
      std::cout << "Solved A Q = B Q Lambda with n=" << command.order << ", grid="
                << grid.rows << 'x' << grid.columns << "\nQ: " << command.path_Q
                << "\nLambda: " << command.path_Lambda << std::endl;
    } catch (const std::exception& error) {
      output_error = error.what();
    }
  }
  broadcastError(&output_error, rank, MPI_COMM_WORLD);
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
