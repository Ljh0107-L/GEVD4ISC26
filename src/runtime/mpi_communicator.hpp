#pragma once

#include <mpi.h>

namespace gevd4isc26::detail {

class MpiCommunicator {
public:
  MpiCommunicator() = default;
  explicit MpiCommunicator(MPI_Comm communicator);
  ~MpiCommunicator();

  MpiCommunicator(const MpiCommunicator&) = delete;
  MpiCommunicator& operator=(const MpiCommunicator&) = delete;

  MpiCommunicator(MpiCommunicator&& other) noexcept;
  MpiCommunicator& operator=(MpiCommunicator&& other) noexcept;

  void duplicate(MPI_Comm communicator);
  void reset() noexcept;

  [[nodiscard]] MPI_Comm get() const noexcept { return communicator_; }
  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int size() const noexcept { return size_; }

private:
  MPI_Comm communicator_ = MPI_COMM_NULL;
  int rank_ = 0;
  int size_ = 1;
};

}  // namespace gevd4isc26::detail
