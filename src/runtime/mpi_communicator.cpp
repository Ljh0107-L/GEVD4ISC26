#include "runtime/mpi_communicator.hpp"

#include "runtime/error.hpp"

#include <stdexcept>
#include <utility>

namespace gevd4isc26::detail {

MpiCommunicator::MpiCommunicator(MPI_Comm communicator) {
  duplicate(communicator);
}

MpiCommunicator::~MpiCommunicator() {
  reset();
}

MpiCommunicator::MpiCommunicator(MpiCommunicator&& other) noexcept
    : communicator_(std::exchange(other.communicator_, MPI_COMM_NULL)),
      rank_(std::exchange(other.rank_, 0)),
      size_(std::exchange(other.size_, 1)) {}

MpiCommunicator& MpiCommunicator::operator=(
    MpiCommunicator&& other) noexcept {
  if (this != &other) {
    reset();
    communicator_ = std::exchange(other.communicator_, MPI_COMM_NULL);
    rank_ = std::exchange(other.rank_, 0);
    size_ = std::exchange(other.size_, 1);
  }
  return *this;
}

void MpiCommunicator::duplicate(MPI_Comm communicator) {
  reset();
  int initialized = 0;
  GEVD_MPI(MPI_Initialized(&initialized));
  if (initialized == 0) {
    throw std::runtime_error(
        "MPI must be initialized before constructing the GEVD solver");
  }
  if (communicator == MPI_COMM_NULL) {
    throw std::invalid_argument("GEVD communicator must not be MPI_COMM_NULL");
  }
  GEVD_MPI(MPI_Comm_dup(communicator, &communicator_));
  GEVD_MPI(MPI_Comm_set_errhandler(communicator_, MPI_ERRORS_RETURN));
  GEVD_MPI(MPI_Comm_rank(communicator_, &rank_));
  GEVD_MPI(MPI_Comm_size(communicator_, &size_));
}

void MpiCommunicator::reset() noexcept {
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (communicator_ != MPI_COMM_NULL && finalized == 0) {
    MPI_Comm_free(&communicator_);
  }
  communicator_ = MPI_COMM_NULL;
  rank_ = 0;
  size_ = 1;
}

}  // namespace gevd4isc26::detail
