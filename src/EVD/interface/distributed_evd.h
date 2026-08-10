#pragma once

#include <mpi.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// This narrow C interface isolates the optimized distributed EVD engine from
// the readable GEVD orchestration code. Callers see the kernel implementation
// only as the standard symmetric eigensolver stage C Y = Y Lambda.

typedef int (*gevd4isc26_concurrent_work_t)(void* user_data);

typedef struct {
  int block_size;
  int panel_size;
  int enable_sbr_bc_pipeline;
  int pipeline_sweeps;
  int print_progress;
} gevd4isc26_standard_evd_options_t;

// All fields are maximum wall times across MPI ranks, in milliseconds.
// Dense-to-band and band-to-tridiagonal are traditionally called SBR and BC.
typedef struct {
  double initialization_ms;
  double matrix_setup_ms;
  double dense_to_band_ms;
  double band_to_tridiagonal_ms;
  double reduction_critical_path_ms;
  double tridiagonal_eigensolve_ms;
  double dense_backtransform_ms;
  double band_backtransform_ms;
  double eigenvector_composition_ms;
  double overlapped_critical_path_ms;
  double numerical_wall_ms;
  double cleanup_ms;
  double engine_call_ms;
} gevd4isc26_standard_evd_timings_t;

// Solves the standard symmetric problem C Y = Y Lambda.  C and Y use the
// engine's block-aligned column distribution.  eigenvalues is replicated on
// every MPI rank before this function returns.
int gevd4isc26_symmetric_evd_device(
    int64_t order,
    double* matrix_C,
    int64_t leading_dimension_C,
    int64_t local_columns_C,
    double* eigenvalues,
    double* eigenvectors_Y,
    int64_t leading_dimension_Y,
    int64_t local_columns_Y,
    MPI_Comm communicator,
    const gevd4isc26_standard_evd_options_t* options,
    gevd4isc26_concurrent_work_t concurrent_work,
    void* concurrent_work_data,
    gevd4isc26_standard_evd_timings_t* timings);

#ifdef __cplusplus
}
#endif
