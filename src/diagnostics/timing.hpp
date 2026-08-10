#pragma once

#include "distributed/context.hpp"
#include "evd/adapter.hpp"

namespace gevd4isc26::detail {

// Local wall times for one complete solve of A Q = B Q Lambda.  Reporting
// performs one collective maximum so timing itself does not add a barrier per
// numerical stage.
struct GevdPipelineTimings {
  double total_seconds = 0.0;
  double resource_setup_seconds = 0.0;
  double input_transfer_seconds = 0.0;
  double cholesky_seconds = 0.0;
  double reduction_to_standard_seconds = 0.0;
  StandardEigensolverTimings standard_eigensolver;
  double inverse_preparation_seconds = 0.0;
  double generalized_backtransform_seconds = 0.0;
  double output_transfer_seconds = 0.0;

  bool reused_cholesky = false;
  bool reused_inverse = false;
  bool used_overlapped_inverse = false;
};

void printGevdPipelineTimings(const DistributedGpuContext& context,
                              const GevdPipelineTimings& local);

}  // namespace gevd4isc26::detail
