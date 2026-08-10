#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

namespace evd_redist {

inline long ceilDiv(long a, long b)
{
  return (a + b - 1) / b;
}

inline size_t tileBytesFromEnv()
{
  const char *value = std::getenv("EVD_REDIST_TILE_MB");
  long mb = value != nullptr ? std::atol(value) : 128;
  if (mb <= 0) {
    mb = 128;
  }
  return static_cast<size_t>(mb) * 1024u * 1024u;
}

__host__ __device__ inline long localToGlobalIndex(long localIndex,
                                                    long block,
                                                    int procCoord,
                                                    int srcProc,
                                                    int nprocs)
{
  const long localBlock = localIndex / block;
  const long inBlock = localIndex % block;
  const long firstOwnedBlock = (procCoord - srcProc + nprocs) % nprocs;
  const long globalBlock = localBlock * nprocs + firstOwnedBlock;
  return globalBlock * block + inBlock;
}

__host__ __device__ inline long globalToLocalIndex(long globalIndex,
                                                    long block,
                                                    int procCoord,
                                                    int srcProc,
                                                    int nprocs)
{
  const long globalBlock = globalIndex / block;
  const long inBlock = globalIndex % block;
  const long firstOwnedBlock = (procCoord - srcProc + nprocs) % nprocs;
  return ((globalBlock - firstOwnedBlock) / nprocs) * block + inBlock;
}

inline bool ownsGlobalIndex(long globalIndex, long block, int procCoord, int srcProc, int nprocs)
{
  const long globalBlock = globalIndex / block;
  return (srcProc + globalBlock) % nprocs == procCoord;
}

inline long numroc(long n, long block, int procCoord, int srcProc, int nprocs)
{
  long count = 0;
  const long nblocks = ceilDiv(n, block);
  for (long blk = 0; blk < nblocks; ++blk) {
    if ((srcProc + blk) % nprocs == procCoord) {
      const long begin = blk * block;
      count += std::min(block, n - begin);
    }
  }
  return count;
}

inline long countOwnedInRange(long start,
                              long cols,
                              long block,
                              int procCoord,
                              int srcProc,
                              int nprocs)
{
  if (cols <= 0) {
    return 0;
  }
  const long end = start + cols;
  long count = 0;
  const long firstBlock = start / block;
  const long lastBlock = (end - 1) / block;
  for (long blk = firstBlock; blk <= lastBlock; ++blk) {
    if ((srcProc + blk) % nprocs != procCoord) {
      continue;
    }
    const long begin = std::max(start, blk * block);
    const long finish = std::min(end, (blk + 1) * block);
    count += std::max<long>(0, finish - begin);
  }
  return count;
}

inline long firstLocalIndexInRange(long start,
                                   long cols,
                                   long block,
                                   int procCoord,
                                   int srcProc,
                                   int nprocs)
{
  if (cols <= 0) {
    return 0;
  }
  const long end = start + cols;
  const long firstBlock = start / block;
  const long lastBlock = (end - 1) / block;
  for (long blk = firstBlock; blk <= lastBlock; ++blk) {
    if ((srcProc + blk) % nprocs != procCoord) {
      continue;
    }
    const long global = std::max(start, blk * block);
    return globalToLocalIndex(global, block, procCoord, srcProc, nprocs);
  }
  return 0;
}

inline void buildBlockAlignedColumnDistribution(long n,
                                                long block,
                                                int ranks,
                                                std::vector<long> *counts,
                                                std::vector<long> *displs)
{
  const long totalBlocks = n / block;
  counts->assign(ranks, 0);
  for (int r = 0; r < ranks; ++r) {
    const long blocks = totalBlocks / ranks + (r < static_cast<int>(totalBlocks % ranks) ? 1 : 0);
    (*counts)[r] = blocks * block;
  }
  displs->assign(ranks, 0);
  for (int r = 1; r < ranks; ++r) {
    (*displs)[r] = (*displs)[r - 1] + (*counts)[r - 1];
  }
}

inline int checkCuda(cudaError_t status, const char *what)
{
  if (status == cudaSuccess) {
    return 0;
  }
  std::cerr << "GEVD standard-stage redistribution CUDA error in " << what << ": "
            << cudaGetErrorString(status) << std::endl;
  return 1;
}

inline int checkNccl(ncclResult_t status, const char *what)
{
  if (status == ncclSuccess) {
    return 0;
  }
  std::cerr << "GEVD standard-stage redistribution NCCL error in " << what << ": "
            << ncclGetErrorString(status) << std::endl;
  return 1;
}

inline int checkMpi(int status, const char *what)
{
  if (status == MPI_SUCCESS) {
    return 0;
  }
  char errstr[MPI_MAX_ERROR_STRING];
  int errlen = 0;
  MPI_Error_string(status, errstr, &errlen);
  std::cerr << "GEVD standard-stage redistribution MPI error in " << what << ": "
            << std::string(errstr, errlen) << std::endl;
  return 1;
}

static __global__ void packBlacsTileKernel(const double *src,
                                           long lld,
                                           long localRows,
                                           long localColStart,
                                           long cols,
                                           double *dst)
{
  const long idx = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
  const long total = localRows * cols;
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (long pos = idx; pos < total; pos += stride) {
    const long row = pos % localRows;
    const long col = pos / localRows;
    dst[pos] = src[row + (localColStart + col) * lld];
  }
}

static __global__ void unpackBlacsToColumnBlockKernel(const double *src,
                                                      long srcLocalRows,
                                                      long srcLocalColStart,
                                                      long srcMyrow,
                                                      long srcMycol,
                                                      long mb,
                                                      long nb,
                                                      long rsrc,
                                                      long csrc,
                                                      long nprow,
                                                      long npcol,
                                                      double *dst,
                                                      long n,
                                                      long dstColStart,
                                                      long cols)
{
  const long idx = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
  const long total = srcLocalRows * cols;
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (long pos = idx; pos < total; pos += stride) {
    const long localRow = pos % srcLocalRows;
    const long packedCol = pos / srcLocalRows;
    const long globalRow = localToGlobalIndex(localRow, mb, srcMyrow, rsrc, nprow);
    const long globalCol =
        localToGlobalIndex(srcLocalColStart + packedCol, nb, srcMycol, csrc, npcol);
    const long dstLocalCol = globalCol - dstColStart;
    if (globalRow < n && dstLocalCol >= 0) {
      dst[globalRow + dstLocalCol * n] = src[pos];
    }
  }
}

static __global__ void packColumnBlockToBlacsKernel(const double *src,
                                                    long n,
                                                    long srcColStart,
                                                    long dstLocalRows,
                                                    long dstLocalColStart,
                                                    long dstMyrow,
                                                    long dstMycol,
                                                    long mb,
                                                    long nb,
                                                    long rsrc,
                                                    long csrc,
                                                    long nprow,
                                                    long npcol,
                                                    double *dst,
                                                    long cols)
{
  const long idx = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
  const long total = dstLocalRows * cols;
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (long pos = idx; pos < total; pos += stride) {
    const long dstLocalRow = pos % dstLocalRows;
    const long packedCol = pos / dstLocalRows;
    const long globalRow = localToGlobalIndex(dstLocalRow, mb, dstMyrow, rsrc, nprow);
    const long globalCol =
        localToGlobalIndex(dstLocalColStart + packedCol, nb, dstMycol, csrc, npcol);
    const long srcLocalCol = globalCol - srcColStart;
    if (globalRow < n && srcLocalCol >= 0) {
      dst[pos] = src[globalRow + srcLocalCol * n];
    }
  }
}

static __global__ void unpackColumnBlockToBlacsKernel(const double *src,
                                                      long dstLocalRows,
                                                      long dstLocalColStart,
                                                      long cols,
                                                      double *dst,
                                                      long lld)
{
  const long idx = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
  const long total = dstLocalRows * cols;
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (long pos = idx; pos < total; pos += stride) {
    const long row = pos % dstLocalRows;
    const long col = pos / dstLocalRows;
    dst[row + (dstLocalColStart + col) * lld] = src[pos];
  }
}

static __global__ void symmetrizeColumnBlockDiagonalKernel(double *a,
                                                           long n,
                                                           long colStart,
                                                           long localCols)
{
  const long localRow = blockIdx.x * blockDim.x + threadIdx.x;
  const long localCol = blockIdx.y * blockDim.y + threadIdx.y;
  if (localRow >= localCols || localCol >= localCols) {
    return;
  }
  const long globalRow = colStart + localRow;
  const long globalCol = colStart + localCol;
  if (globalRow < globalCol) {
    a[globalRow + localCol * n] = a[globalCol + localRow * n];
  }
}

static __global__ void packLowerOffdiagKernel(const double *src,
                                              long n,
                                              long lowColStart,
                                              long lowCols,
                                              long highRowStart,
                                              long highRows,
                                              double *dst)
{
  const long idx = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
  const long total = lowCols * highRows;
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (long pos = idx; pos < total; pos += stride) {
    const long highRow = pos % highRows;
    const long lowCol = pos / highRows;
    dst[pos] = src[highRowStart + highRow + lowCol * n];
  }
}

static __global__ void unpackUpperOffdiagKernel(const double *src,
                                                long lowColStart,
                                                long lowCols,
                                                long highColStart,
                                                long highColsTile,
                                                long dstColStart,
                                                double *dst,
                                                long n)
{
  const long idx = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
  const long total = lowCols * highColsTile;
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (long pos = idx; pos < total; pos += stride) {
    const long highCol = pos % highColsTile;
    const long lowRow = pos / highColsTile;
    const long localCol = (highColStart + highCol) - dstColStart;
    dst[lowColStart + lowRow + localCol * n] = src[pos];
  }
}

inline int launch1d(long elems, cudaStream_t /*stream*/,
                    void (*unused)() = nullptr)
{
  (void)unused;
  const int block = 256;
  return static_cast<int>(std::min<long>(65535, std::max<long>(1, ceilDiv(elems, block))));
}

inline int blacsToColumnBlock(long n,
                              long mb,
                              long nb,
                              int rsrc,
                              int csrc,
                              int nprow,
                              int npcol,
                              int rank,
                              int size,
                              const double *dBlacs,
                              long lld,
                              double *dColumn,
                              long columnBlockSize,
                              ncclComm_t nccl,
                              cudaStream_t stream)
{
  if (size != nprow * npcol || n % columnBlockSize != 0) {
    std::cerr << "Invalid BLACS to column-block redistribution shape" << std::endl;
    return 1;
  }

  std::vector<long> colCounts;
  std::vector<long> colDispls;
  buildBlockAlignedColumnDistribution(n, columnBlockSize, size, &colCounts, &colDispls);

  std::vector<long> rankLocalRows(size);
  for (int r = 0; r < size; ++r) {
    rankLocalRows[r] = numroc(n, mb, r / npcol, rsrc, nprow);
  }

  size_t tileElems = tileBytesFromEnv() / sizeof(double);
  tileElems = std::max(tileElems, static_cast<size_t>(n));
  const long tileCols = std::max<long>(1, static_cast<long>(tileElems / static_cast<size_t>(n)));
  double *dSend = nullptr;
  double *dRecv = nullptr;
  if (checkCuda(cudaMalloc(&dSend, tileElems * sizeof(double)), "cudaMalloc redist send") != 0 ||
      checkCuda(cudaMalloc(&dRecv, tileElems * sizeof(double)), "cudaMalloc redist recv") != 0) {
    cudaFree(dSend);
    cudaFree(dRecv);
    return 1;
  }

  const int selfCol = rank % npcol;
  const long selfRows = rankLocalRows[rank];
  const int block = 256;

  for (int dst = 0; dst < size; ++dst) {
    const long dstStart = colDispls[dst];
    const long dstCols = colCounts[dst];
    for (long off = 0; off < dstCols; off += tileCols) {
      const long tileStart = dstStart + off;
      const long curCols = std::min(tileCols, dstCols - off);
      std::vector<size_t> recvOffsets(size, 0);
      std::vector<size_t> recvCounts(size, 0);
      size_t totalRecv = 0;
      for (int src = 0; src < size; ++src) {
        const int srcCol = src % npcol;
        const long ownedCols = countOwnedInRange(tileStart, curCols, nb, srcCol, csrc, npcol);
        recvOffsets[src] = totalRecv;
        recvCounts[src] = static_cast<size_t>(rankLocalRows[src]) * static_cast<size_t>(ownedCols);
        totalRecv += recvCounts[src];
      }
      if (totalRecv > tileElems) {
        std::cerr << "BLACS to column-block tile buffer is too small" << std::endl;
        cudaFree(dSend);
        cudaFree(dRecv);
        return 1;
      }

      const long selfOwnedCols = countOwnedInRange(tileStart, curCols, nb, selfCol, csrc, npcol);
      const size_t selfCount = static_cast<size_t>(selfRows) * static_cast<size_t>(selfOwnedCols);
      if (selfCount > 0) {
        const long selfLocalColStart =
            firstLocalIndexInRange(tileStart, curCols, nb, selfCol, csrc, npcol);
        double *packDst = rank == dst ? dRecv + recvOffsets[rank] : dSend;
        packBlacsTileKernel<<<launch1d(static_cast<long>(selfCount), stream), block, 0, stream>>>(
            dBlacs, lld, selfRows, selfLocalColStart, selfOwnedCols, packDst);
        if (checkCuda(cudaGetLastError(), "packBlacsTileKernel") != 0) {
          cudaFree(dSend);
          cudaFree(dRecv);
          return 1;
        }
      }

      if (checkNccl(ncclGroupStart(), "ncclGroupStart blacsToColumnBlock") != 0) {
        cudaFree(dSend);
        cudaFree(dRecv);
        return 1;
      }
      if (rank == dst) {
        for (int src = 0; src < size; ++src) {
          if (src == rank || recvCounts[src] == 0) {
            continue;
          }
          if (checkNccl(ncclRecv(dRecv + recvOffsets[src], recvCounts[src], ncclDouble, src, nccl,
                                 stream),
                        "ncclRecv blacsToColumnBlock") != 0) {
            cudaFree(dSend);
            cudaFree(dRecv);
            return 1;
          }
        }
      } else if (selfCount > 0) {
        if (checkNccl(ncclSend(dSend, selfCount, ncclDouble, dst, nccl, stream),
                      "ncclSend blacsToColumnBlock") != 0) {
          cudaFree(dSend);
          cudaFree(dRecv);
          return 1;
        }
      }
      if (checkNccl(ncclGroupEnd(), "ncclGroupEnd blacsToColumnBlock") != 0 ||
          checkCuda(cudaStreamSynchronize(stream), "sync blacsToColumnBlock") != 0) {
        cudaFree(dSend);
        cudaFree(dRecv);
        return 1;
      }

      if (rank == dst) {
        for (int src = 0; src < size; ++src) {
          if (recvCounts[src] == 0) {
            continue;
          }
          const int srcRow = src / npcol;
          const int srcCol = src % npcol;
          const long srcOwnedCols =
              countOwnedInRange(tileStart, curCols, nb, srcCol, csrc, npcol);
          const long srcLocalColStart =
              firstLocalIndexInRange(tileStart, curCols, nb, srcCol, csrc, npcol);
          unpackBlacsToColumnBlockKernel<<<launch1d(static_cast<long>(recvCounts[src]), stream),
                                            block, 0, stream>>>(
              dRecv + recvOffsets[src], rankLocalRows[src], srcLocalColStart, srcRow, srcCol, mb,
              nb, rsrc, csrc, nprow, npcol, dColumn, n, dstStart, srcOwnedCols);
          if (checkCuda(cudaGetLastError(), "unpackBlacsToColumnBlockKernel") != 0) {
            cudaFree(dSend);
            cudaFree(dRecv);
            return 1;
          }
        }
        if (checkCuda(cudaStreamSynchronize(stream), "sync unpack blacsToColumnBlock") != 0) {
          cudaFree(dSend);
          cudaFree(dRecv);
          return 1;
        }
      }
    }
  }

  cudaFree(dSend);
  cudaFree(dRecv);
  return 0;
}

inline int columnBlockToBlacs(long n,
                              long mb,
                              long nb,
                              int rsrc,
                              int csrc,
                              int nprow,
                              int npcol,
                              int rank,
                              int size,
                              const double *dColumn,
                              long columnBlockSize,
                              double *dBlacs,
                              long lld,
                              ncclComm_t nccl,
                              cudaStream_t stream)
{
  if (size != nprow * npcol || n % columnBlockSize != 0) {
    std::cerr << "Invalid column-block to BLACS redistribution shape" << std::endl;
    return 1;
  }

  std::vector<long> colCounts;
  std::vector<long> colDispls;
  buildBlockAlignedColumnDistribution(n, columnBlockSize, size, &colCounts, &colDispls);

  std::vector<long> rankLocalRows(size);
  for (int r = 0; r < size; ++r) {
    rankLocalRows[r] = numroc(n, mb, r / npcol, rsrc, nprow);
  }

  size_t tileElems = tileBytesFromEnv() / sizeof(double);
  tileElems = std::max(tileElems, static_cast<size_t>(n));
  const long tileCols = std::max<long>(1, static_cast<long>(tileElems / static_cast<size_t>(n)));
  double *dSend = nullptr;
  double *dRecv = nullptr;
  if (checkCuda(cudaMalloc(&dSend, tileElems * sizeof(double)), "cudaMalloc redist send") != 0 ||
      checkCuda(cudaMalloc(&dRecv, tileElems * sizeof(double)), "cudaMalloc redist recv") != 0) {
    cudaFree(dSend);
    cudaFree(dRecv);
    return 1;
  }

  const long selfColStart = colDispls[rank];
  const long selfColEnd = selfColStart + colCounts[rank];
  const int block = 256;

  for (int dst = 0; dst < size; ++dst) {
    const int dstRow = dst / npcol;
    const int dstCol = dst % npcol;
    const long dstRows = rankLocalRows[dst];
    for (long tileStart = 0; tileStart < n; tileStart += tileCols) {
      const long curCols = std::min(tileCols, n - tileStart);
      std::vector<size_t> recvOffsets(size, 0);
      std::vector<size_t> recvCounts(size, 0);
      size_t totalRecv = 0;
      for (int src = 0; src < size; ++src) {
        const long srcBegin = std::max(tileStart, colDispls[src]);
        const long srcEnd = std::min(tileStart + curCols, colDispls[src] + colCounts[src]);
        const long srcCols = std::max<long>(0, srcEnd - srcBegin);
        const long ownedCols = countOwnedInRange(srcBegin, srcCols, nb, dstCol, csrc, npcol);
        recvOffsets[src] = totalRecv;
        recvCounts[src] = static_cast<size_t>(dstRows) * static_cast<size_t>(ownedCols);
        totalRecv += recvCounts[src];
      }
      if (totalRecv > tileElems) {
        std::cerr << "Column-block to BLACS tile buffer is too small" << std::endl;
        cudaFree(dSend);
        cudaFree(dRecv);
        return 1;
      }

      const long selfBegin = std::max(tileStart, selfColStart);
      const long selfEnd = std::min(tileStart + curCols, selfColEnd);
      const long selfColsInTile = std::max<long>(0, selfEnd - selfBegin);
      const long selfOwnedCols = countOwnedInRange(selfBegin, selfColsInTile, nb, dstCol, csrc,
                                                   npcol);
      const size_t selfCount = static_cast<size_t>(dstRows) * static_cast<size_t>(selfOwnedCols);
      if (selfCount > 0) {
        const long dstLocalColStart =
            firstLocalIndexInRange(selfBegin, selfColsInTile, nb, dstCol, csrc, npcol);
        double *packDst = rank == dst ? dRecv + recvOffsets[rank] : dSend;
        packColumnBlockToBlacsKernel<<<launch1d(static_cast<long>(selfCount), stream), block, 0,
                                       stream>>>(
            dColumn, n, selfColStart, dstRows, dstLocalColStart, dstRow, dstCol, mb, nb, rsrc,
            csrc, nprow, npcol, packDst, selfOwnedCols);
        if (checkCuda(cudaGetLastError(), "packColumnBlockToBlacsKernel") != 0) {
          cudaFree(dSend);
          cudaFree(dRecv);
          return 1;
        }
      }

      if (checkNccl(ncclGroupStart(), "ncclGroupStart columnBlockToBlacs") != 0) {
        cudaFree(dSend);
        cudaFree(dRecv);
        return 1;
      }
      if (rank == dst) {
        for (int src = 0; src < size; ++src) {
          if (src == rank || recvCounts[src] == 0) {
            continue;
          }
          if (checkNccl(ncclRecv(dRecv + recvOffsets[src], recvCounts[src], ncclDouble, src, nccl,
                                 stream),
                        "ncclRecv columnBlockToBlacs") != 0) {
            cudaFree(dSend);
            cudaFree(dRecv);
            return 1;
          }
        }
      } else if (selfCount > 0) {
        if (checkNccl(ncclSend(dSend, selfCount, ncclDouble, dst, nccl, stream),
                      "ncclSend columnBlockToBlacs") != 0) {
          cudaFree(dSend);
          cudaFree(dRecv);
          return 1;
        }
      }
      if (checkNccl(ncclGroupEnd(), "ncclGroupEnd columnBlockToBlacs") != 0 ||
          checkCuda(cudaStreamSynchronize(stream), "sync columnBlockToBlacs") != 0) {
        cudaFree(dSend);
        cudaFree(dRecv);
        return 1;
      }

      if (rank == dst) {
        for (int src = 0; src < size; ++src) {
          if (recvCounts[src] == 0) {
            continue;
          }
          const long srcBegin = std::max(tileStart, colDispls[src]);
          const long srcEnd = std::min(tileStart + curCols, colDispls[src] + colCounts[src]);
          const long srcCols = std::max<long>(0, srcEnd - srcBegin);
          const long srcOwnedCols = countOwnedInRange(srcBegin, srcCols, nb, dstCol, csrc, npcol);
          const long dstLocalColStart =
              firstLocalIndexInRange(srcBegin, srcCols, nb, dstCol, csrc, npcol);
          unpackColumnBlockToBlacsKernel<<<launch1d(static_cast<long>(recvCounts[src]), stream),
                                            block, 0, stream>>>(
              dRecv + recvOffsets[src], dstRows, dstLocalColStart, srcOwnedCols, dBlacs, lld);
          if (checkCuda(cudaGetLastError(), "unpackColumnBlockToBlacsKernel") != 0) {
            cudaFree(dSend);
            cudaFree(dRecv);
            return 1;
          }
        }
        if (checkCuda(cudaStreamSynchronize(stream), "sync unpack columnBlockToBlacs") != 0) {
          cudaFree(dSend);
          cudaFree(dRecv);
          return 1;
        }
      }
    }
  }

  cudaFree(dSend);
  cudaFree(dRecv);
  return 0;
}

inline int symmetrizeColumnBlockLowerToUpper(long n,
                                             double *dColumn,
                                             int rank,
                                             int size,
                                             const std::vector<long> &counts,
                                             const std::vector<long> &displs,
                                             ncclComm_t nccl,
                                             cudaStream_t stream)
{
  dim3 diagBlock(16, 16);
  dim3 diagGrid(ceilDiv(counts[rank], diagBlock.x), ceilDiv(counts[rank], diagBlock.y));
  symmetrizeColumnBlockDiagonalKernel<<<diagGrid, diagBlock, 0, stream>>>(
      dColumn, n, displs[rank], counts[rank]);
  if (checkCuda(cudaGetLastError(), "symmetrizeColumnBlockDiagonalKernel") != 0 ||
      checkCuda(cudaStreamSynchronize(stream), "sync diagonal symmetrize") != 0) {
    return 1;
  }

  size_t tileElems = tileBytesFromEnv() / sizeof(double);
  tileElems = std::max(tileElems, static_cast<size_t>(n));
  double *dBuf = nullptr;
  if (checkCuda(cudaMalloc(&dBuf, tileElems * sizeof(double)), "cudaMalloc sym buffer") != 0) {
    return 1;
  }

  const int block = 256;
  for (int lo = 0; lo < size; ++lo) {
    const long loCols = counts[lo];
    const long loStart = displs[lo];
    if (loCols <= 0) {
      continue;
    }
    const long maxHighTileCols =
        std::max<long>(1, static_cast<long>(tileElems / static_cast<size_t>(loCols)));
    for (int hi = lo + 1; hi < size; ++hi) {
      const long hiStart = displs[hi];
      const long hiCols = counts[hi];
      for (long off = 0; off < hiCols; off += maxHighTileCols) {
        const long curHighCols = std::min(maxHighTileCols, hiCols - off);
        const size_t count = static_cast<size_t>(loCols) * static_cast<size_t>(curHighCols);
        if (rank == lo) {
          packLowerOffdiagKernel<<<launch1d(static_cast<long>(count), stream), block, 0, stream>>>(
              dColumn, n, loStart, loCols, hiStart + off, curHighCols, dBuf);
          if (checkCuda(cudaGetLastError(), "packLowerOffdiagKernel") != 0) {
            cudaFree(dBuf);
            return 1;
          }
        }

        if (checkNccl(ncclGroupStart(), "ncclGroupStart symmetrize") != 0) {
          cudaFree(dBuf);
          return 1;
        }
        if (rank == lo) {
          if (checkNccl(ncclSend(dBuf, count, ncclDouble, hi, nccl, stream),
                        "ncclSend symmetrize") != 0) {
            cudaFree(dBuf);
            return 1;
          }
        } else if (rank == hi) {
          if (checkNccl(ncclRecv(dBuf, count, ncclDouble, lo, nccl, stream),
                        "ncclRecv symmetrize") != 0) {
            cudaFree(dBuf);
            return 1;
          }
        }
        if (checkNccl(ncclGroupEnd(), "ncclGroupEnd symmetrize") != 0 ||
            checkCuda(cudaStreamSynchronize(stream), "sync symmetrize sendrecv") != 0) {
          cudaFree(dBuf);
          return 1;
        }

        if (rank == hi) {
          unpackUpperOffdiagKernel<<<launch1d(static_cast<long>(count), stream), block, 0,
                                    stream>>>(dBuf, loStart, loCols, hiStart + off, curHighCols,
                                              hiStart, dColumn, n);
          if (checkCuda(cudaGetLastError(), "unpackUpperOffdiagKernel") != 0 ||
              checkCuda(cudaStreamSynchronize(stream), "sync unpack symmetrize") != 0) {
            cudaFree(dBuf);
            return 1;
          }
        }
      }
    }
  }

  cudaFree(dBuf);
  return 0;
}

}  // namespace evd_redist
