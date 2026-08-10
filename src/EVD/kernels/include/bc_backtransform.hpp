#pragma once

#include <cuda_runtime.h>

long bcBacktransformPackedElementCount(long order);

int packBcBacktransformReflectors(
    double* packed_reflectors,
    const double* reflectors,
    long reflector_leading_dimension,
    long order,
    int band_width,
    cudaStream_t stream = nullptr);

int applyPackedBcBacktransform(
    double* eigenvectors,
    long eigenvector_leading_dimension,
    long local_eigenvector_columns,
    const double* packed_reflectors,
    const long* packed_reflector_offsets,
    long packed_reflector_offset_count,
    long order,
    int band_width,
    cudaStream_t stream = nullptr);

// Non-packed variants are retained for kernel experiments and comparison
// runs. The distributed production path uses applyPackedBcBacktransform().
int applyLocalBcBacktransform(
    double* eigenvectors,
    long eigenvector_leading_dimension,
    long local_eigenvector_columns,
    double* reflectors,
    long reflector_leading_dimension,
    long order,
    int band_width,
    cudaStream_t stream = nullptr);

int applyBcBacktransform(
    double* eigenvectors,
    long eigenvector_leading_dimension,
    double* reflectors,
    long reflector_leading_dimension,
    long order,
    int band_width,
    cudaStream_t stream = nullptr);
