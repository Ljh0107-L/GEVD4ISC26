#pragma once

#include "distributed/context.hpp"

namespace gevd4isc26::detail {

[[nodiscard]] int readNumericalInfo(const DistributedGpuContext& context,
                                    const int* device_info);

void checkNumericalInfo(const DistributedGpuContext& context,
                        int local_info,
                        const char* stage);

}  // namespace gevd4isc26::detail
