#pragma once

// These values are compile-time tuning parameters because they determine
// CUDA register and shared-memory shapes. CMake may override them for a
// particular GPU, but every BC producer and consumer includes this header so
// the packed-reflector layout cannot silently diverge.
#ifndef EVD_BC_BACK_ROWS_PER_THREAD
#define EVD_BC_BACK_ROWS_PER_THREAD 8
#endif

#ifndef EVD_BC_BACK_MAX_WARP_GROUPS
#define EVD_BC_BACK_MAX_WARP_GROUPS 24
#endif

#ifndef EVD_BC_BACK_COLUMN_TILE
#define EVD_BC_BACK_COLUMN_TILE 90
#endif

namespace gevd4isc26::evd {

inline constexpr int kBcBackRowsPerThread =
    EVD_BC_BACK_ROWS_PER_THREAD;
inline constexpr int kBcBackSweepRows =
    kBcBackRowsPerThread * 32;
inline constexpr int kBcBackMaxWarpGroups =
    EVD_BC_BACK_MAX_WARP_GROUPS;
inline constexpr int kBcBackColumnTile =
    EVD_BC_BACK_COLUMN_TILE;
inline constexpr int kBcBackReductionWidth =
    32 / kBcBackRowsPerThread;

static_assert(kBcBackRowsPerThread > 0);
static_assert(32 % kBcBackRowsPerThread == 0);
static_assert(kBcBackRowsPerThread % 4 == 0,
              "BC backtransform vector loads require a multiple of four");
static_assert(kBcBackMaxWarpGroups > 0);
static_assert(kBcBackColumnTile > 0);

}  // namespace gevd4isc26::evd
