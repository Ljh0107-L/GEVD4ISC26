#pragma once

#include "engine/evd_types.hpp"

#include <algorithm>

namespace gevd4isc26::evd {

// Environment-controlled implementation choices used by the top-level EVD
// driver. Loading them once makes the executed path explicit and keeps the
// mathematical stage orchestration free of scattered getenv calls.
struct EvdDriverConfig {
  int back_balance_percent = 0;
  bool align_backtransform_to_block = true;

  bool stage_pipeline = false;
  bool sbr_double_block = true;
  bool sbr_double_block_back = true;
  bool sbr_band_pipeline = false;
  bool sbr_pipeline_early_return = false;
  bool sbr_suffix_communicator = false;
  bool bc_tile_suffix_broadcast = false;
  bool backtransform_barriers = true;
  bool final_gemm_start_barrier = true;

  int bc_range_mode = 1;
  int bc_sync_mode = 1;
  bool bc_neighbor_handoff = false;
  bool bc_local_pipeline = false;
  bool bc_segmented_full_band_sync = true;
  bool bc_segmented_brown_only = false;
  bool bc_segmented_rolling = false;
  bool bc_segmented_block_local = false;
  bool bc_segmented_tile_wave = false;
  bool bc_extract_last_owner = false;
  long bc_brown_columns = 0;

  long final_gemm_tile_columns = 512;

  [[nodiscard]] static EvdDriverConfig load(bool use_stage_pipeline,
                                            long order,
                                            long block_size) {
    EvdDriverConfig config;
    config.back_balance_percent =
        envIntOrDefault("EVD_BACK_BALANCE_PCT", 0);
    config.align_backtransform_to_block =
        envIntOrDefault("EVD_BACK_BALANCE_ALIGN_B", 1) != 0;

    config.stage_pipeline = use_stage_pipeline;
    config.sbr_double_block =
        envIntOrDefault("EVD_SBR_DOUBLE_BLOCK", 1) != 0;
    config.sbr_double_block_back =
        config.sbr_double_block &&
        envIntOrDefault("EVD_SBR_DOUBLE_BLOCK_BACK", 1) != 0;
    config.sbr_band_pipeline =
        config.sbr_double_block &&
        envIntOrDefault("EVD_SBR_BAND_PIPELINE",
                        use_stage_pipeline ? 1 : 0) != 0;
    config.sbr_pipeline_early_return =
        config.sbr_double_block && use_stage_pipeline &&
        envIntOrDefault("EVD_SBR_PIPELINE_EARLY_RETURN",
                        use_stage_pipeline ? 1 : 0) != 0;
    config.sbr_suffix_communicator =
        envIntOrDefault("EVD_SBR_SUFFIX_COMM",
                        use_stage_pipeline ? 1 : 0) != 0;
    config.bc_tile_suffix_broadcast =
        envIntOrDefault("EVD_BC_SEGMENTED_TILE_SUFFIX_BCAST", 0) != 0;
    config.backtransform_barriers =
        envIntOrDefault("EVD_BACK_PIPELINE_BARRIERS",
                        use_stage_pipeline ? 0 : 1) != 0;
    config.final_gemm_start_barrier =
        envIntOrDefault("EVD_FINAL_GEMM_START_BARRIER",
                        use_stage_pipeline ? 0 : 1) != 0;

    config.bc_range_mode = envIntOrDefault("EVD_BC_RANGE", 1);
    config.bc_sync_mode = envIntOrDefault("EVD_BC_SYNC_MODE", 1);
    config.bc_neighbor_handoff =
        envIntOrDefault("EVD_BC_BAND_NEIGHBOR", 0) != 0;
    config.bc_local_pipeline =
        envIntOrDefault("EVD_BC_PIPELINE_LOCAL", 0) != 0;
    config.bc_segmented_full_band_sync =
        envIntOrDefault("EVD_BC_SEGMENTED_FULL_BCAST", 1) != 0;
    config.bc_segmented_brown_only =
        envIntOrDefault("EVD_BC_SEGMENTED_BROWN_ONLY", 0) != 0;
    config.bc_segmented_rolling =
        envIntOrDefault("EVD_BC_SEGMENTED_ROLLING", 0) != 0;
    config.bc_segmented_block_local =
        envIntOrDefault("EVD_BC_SEGMENTED_BLOCK_LOCAL", 0) != 0;
    config.bc_segmented_tile_wave =
        envIntOrDefault("EVD_BC_SEGMENTED_TILE_WAVE", 0) != 0;
    config.bc_extract_last_owner =
        envIntOrDefault("EVD_BC_EXTRACT_LAST_OWNER", 0) != 0;
    config.bc_brown_columns = static_cast<long>(
        envIntOrDefault("EVD_BC_BROWN_COLS",
                        static_cast<int>(4 * block_size)));

    config.final_gemm_tile_columns = std::min<long>(
        order, static_cast<long>(
                   envIntOrDefault("EVD_FINAL_TILE_COLS", 512)));
    return config;
  }

  [[nodiscard]] bool initializeSuffixCommunicators() const noexcept {
    return sbr_suffix_communicator || bc_tile_suffix_broadcast;
  }
};

}  // namespace gevd4isc26::evd
