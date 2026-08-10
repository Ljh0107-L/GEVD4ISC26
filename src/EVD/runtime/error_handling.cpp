#include "runtime/error_handling.hpp"

#include <cstdlib>

namespace gevd4isc26::evd {
namespace {

thread_local MPI_Comm error_communicator = MPI_COMM_WORLD;

}  // namespace

MPI_Comm activeErrorCommunicator() noexcept {
  return error_communicator;
}

MPI_Comm setActiveErrorCommunicator(MPI_Comm communicator) noexcept {
  const MPI_Comm previous = error_communicator;
  error_communicator = communicator;
  return previous;
}

[[noreturn]] void abortActiveCommunicator(int code) noexcept {
  MPI_Abort(error_communicator, code);
  std::abort();
}

}  // namespace gevd4isc26::evd
