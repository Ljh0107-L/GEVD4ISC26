#pragma once

#include "gevd4isc26/distributed_matrix.hpp"
#include "gevd4isc26/solver.hpp"

#include <mpi.h>

#include <string>
#include <vector>

namespace gevd4isc26 {

// Root owns a full column-major matrix; the return value is this rank's
// block-cyclic shard. On non-root ranks full_matrix must be empty.
[[nodiscard]] DistributedMatrix scatterDenseMatrix(const std::vector<double>& full_matrix,
                                                    const MatrixDistribution& local_distribution,
                                                    MPI_Comm communicator,
                                                    int root = 0);

// The inverse operation. Only root receives a non-empty vector.
[[nodiscard]] std::vector<double> gatherDenseMatrix(const DistributedMatrixView& local_matrix,
                                                    MPI_Comm communicator,
                                                    int root = 0);

// Raw dense files contain exactly n*n IEEE-754 doubles in column-major order.
[[nodiscard]] std::vector<double> readRawDenseMatrix(const std::string& path,
                                                     std::int64_t order);
void writeRawDenseMatrix(const std::string& path, const std::vector<double>& matrix);

enum class LambdaFileFormat {
  // Store only n diagonal values. This is the scalable default.
  Diagonal,
  // Store an explicit n-by-n column-major matrix, including all zeros.
  Dense,
};

void writeLambda(const std::string& path,
                 const DiagonalMatrix& lambda,
                 LambdaFileFormat format);

}  // namespace gevd4isc26
