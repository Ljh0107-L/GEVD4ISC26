#include "d_and_c_solver.hpp"

#include <algorithm>
#include <chrono>
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

struct Options {
  std::int64_t order = 32000;
  int repeats = 1;
  int validation_vectors = 3;
  bool validate = true;
  bool help = false;
};

std::int64_t parseInteger(const char* text, const char* option) {
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (end == text || *end != '\0') {
    throw std::invalid_argument(std::string("invalid value for ") + option);
  }
  return static_cast<std::int64_t>(value);
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
      options.order = parseInteger(value(), "--n");
    } else if (argument == "--repeat") {
      options.repeats = static_cast<int>(parseInteger(value(), "--repeat"));
    } else if (argument == "--validation-vectors") {
      options.validation_vectors =
          static_cast<int>(parseInteger(value(), "--validation-vectors"));
    } else if (argument == "--no-validation") {
      options.validate = false;
    } else if (argument == "--help" || argument == "-h") {
      options.help = true;
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (options.order <= 0 || options.repeats <= 0 ||
      options.validation_vectors <= 0) {
    throw std::invalid_argument(
        "--n, --repeat, and --validation-vectors must be positive");
  }
  return options;
}

void generateTridiagonal(std::int64_t order,
                         std::vector<double>* diagonal,
                         std::vector<double>* off_diagonal) {
  diagonal->resize(static_cast<std::size_t>(order));
  off_diagonal->resize(static_cast<std::size_t>(std::max<std::int64_t>(
      order - 1, 0)));
  for (std::int64_t row = 0; row < order; ++row) {
    const double position =
        static_cast<double>(row + 1) / static_cast<double>(order + 1);
    (*diagonal)[static_cast<std::size_t>(row)] =
        4.0 + position + 0.05 * std::sin(17.0 * position);
    if (row + 1 < order) {
      (*off_diagonal)[static_cast<std::size_t>(row)] =
          0.20 + 0.02 * std::cos(11.0 * position);
    }
  }
}

std::vector<std::int64_t> validationIndices(std::int64_t order, int count) {
  std::vector<std::int64_t> indices;
  indices.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    const std::int64_t column =
        count == 1
            ? order / 2
            : static_cast<std::int64_t>(
                  static_cast<long double>(index) * (order - 1) /
                  static_cast<long double>(count - 1));
    if (indices.empty() || indices.back() != column) {
      indices.push_back(column);
    }
  }
  return indices;
}

struct Validation {
  double maximum_residual = 0.0;
  double maximum_orthogonality_error = 0.0;
  double relative_trace_error = 0.0;
};

Validation validate(const std::vector<double>& original_diagonal,
                    const std::vector<double>& original_off_diagonal,
                    const std::vector<double>& eigenvalues,
                    const double* eigenvectors,
                    std::int64_t order,
                    int vector_count) {
  const std::vector<std::int64_t> indices =
      validationIndices(order, vector_count);
  Validation result;
  for (std::int64_t column : indices) {
    const double* vector =
        eigenvectors + static_cast<std::size_t>(column * order);
    long double residual_squared = 0.0;
    long double product_squared = 0.0;
    for (std::int64_t row = 0; row < order; ++row) {
      long double product =
          original_diagonal[static_cast<std::size_t>(row)] * vector[row];
      if (row > 0) {
        product +=
            original_off_diagonal[static_cast<std::size_t>(row - 1)] *
            vector[row - 1];
      }
      if (row + 1 < order) {
        product += original_off_diagonal[static_cast<std::size_t>(row)] *
                   vector[row + 1];
      }
      const long double difference =
          product -
          eigenvalues[static_cast<std::size_t>(column)] * vector[row];
      residual_squared += difference * difference;
      product_squared += product * product;
    }
    result.maximum_residual = std::max(
        result.maximum_residual,
        std::sqrt(static_cast<double>(
            residual_squared /
            std::max(product_squared, static_cast<long double>(1.0e-300)))));
  }

  for (std::int64_t left_column : indices) {
    const double* left =
        eigenvectors + static_cast<std::size_t>(left_column * order);
    for (std::int64_t right_column : indices) {
      const double* right =
          eigenvectors + static_cast<std::size_t>(right_column * order);
      long double dot = 0.0;
      for (std::int64_t row = 0; row < order; ++row) {
        dot += static_cast<long double>(left[row]) * right[row];
      }
      const long double expected =
          left_column == right_column ? 1.0L : 0.0L;
      result.maximum_orthogonality_error = std::max(
          result.maximum_orthogonality_error,
          static_cast<double>(std::abs(dot - expected)));
    }
  }

  long double input_trace = 0.0;
  long double eigenvalue_trace = 0.0;
  for (std::size_t index = 0; index < eigenvalues.size(); ++index) {
    input_trace += original_diagonal[index];
    eigenvalue_trace += eigenvalues[index];
  }
  result.relative_trace_error = static_cast<double>(
      std::abs(input_trace - eigenvalue_trace) /
      std::max(std::abs(input_trace), static_cast<long double>(1.0e-300)));
  return result;
}

int run(int argc, char** argv) {
  const Options options = parseOptions(argc, argv);
  if (options.help) {
    std::cout
        << "Usage: d_and_c_benchmark [--n N] [--repeat R] "
           "[--validation-vectors K] [--no-validation]\n";
    return 0;
  }

  const long double output_bytes =
      static_cast<long double>(options.order) * options.order * sizeof(double);
  std::cout << std::setprecision(12)
            << "D&C CPU benchmark n=" << options.order
            << " repeats=" << options.repeats
            << " eigenvector_gib="
            << static_cast<double>(output_bytes / (1024.0L * 1024.0L * 1024.0L))
            << '\n';

  std::vector<double> original_diagonal;
  std::vector<double> original_off_diagonal;
  generateTridiagonal(
      options.order, &original_diagonal, &original_off_diagonal);

  int status = 0;
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    gevd4isc26::evd::d_and_c::AsyncTridiagonalSolve solve;
    const auto wall_start = std::chrono::steady_clock::now();
    const auto prepare_start = wall_start;
    const int start_info =
        solve.start(options.order,
                    original_diagonal,
                    original_off_diagonal);
    const auto prepare_stop = std::chrono::steady_clock::now();
    const int info = start_info == 0 ? solve.wait() : start_info;
    const auto wall_stop = std::chrono::steady_clock::now();

    const double prepare_seconds =
        std::chrono::duration<double>(prepare_stop - prepare_start).count();
    const double wall_seconds =
        std::chrono::duration<double>(wall_stop - wall_start).count();
    std::cout << "run=" << repeat + 1
              << " prepare_seconds=" << prepare_seconds
              << " solve_seconds=" << solve.solveMilliseconds() / 1000.0
              << " wall_seconds=" << wall_seconds
              << " info=" << info;
    if (info != 0) {
      std::cout << '\n';
      return info > 0 ? info : 1;
    }

    if (options.validate) {
      const Validation validation =
          validate(original_diagonal,
                   original_off_diagonal,
                   solve.eigenvalues(),
                   solve.eigenvectors(),
                   options.order,
                   options.validation_vectors);
      std::cout
          << " sampled_residual=" << validation.maximum_residual
          << " sampled_orthogonality="
          << validation.maximum_orthogonality_error
          << " relative_trace_error=" << validation.relative_trace_error;
      if (validation.maximum_residual > 1.0e-10 ||
          validation.maximum_orthogonality_error > 1.0e-10 ||
          validation.relative_trace_error > 1.0e-12) {
        status = 2;
      }
    }
    std::cout << '\n';
  }
  return status;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
