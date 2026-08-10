

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "runtime/cuda_check.hpp"
#include "tuning/bc_backtransform_config.hpp"

using gevd4isc26::evd::kBcBackColumnTile;
using gevd4isc26::evd::kBcBackMaxWarpGroups;
using gevd4isc26::evd::kBcBackReductionWidth;
using gevd4isc26::evd::kBcBackRowsPerThread;
using gevd4isc26::evd::kBcBackSweepRows;

#if CUDART_VERSION >= 13000
using LegacyAlignedDouble4 = double4_16a;
#else
using LegacyAlignedDouble4 = double4;
#endif

namespace {

bool profileBcBackKernel()
{
  const char *value = std::getenv("EVD_PROFILE_BC_BACK");
  return value != nullptr && std::atoi(value) != 0;
}

} // namespace

static __inline__ __device__ double warpReduceSum(double val, int ThreadCount = 32)
{
  for (int mask = ThreadCount / 2; mask > 0; mask /= 2)
  {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

extern __shared__ double dynamicSharedMemory[];

static long packedBcReflectorColumns(long n)
{
  int sweepCount = (n - 1 - 1 + (kBcBackSweepRows - 1)) / (kBcBackSweepRows);
  int lastSweepUCount = n - ((sweepCount - 1) * kBcBackSweepRows + 1) - 1;
  long countU = 0;
  for (int i = 0, tmp = lastSweepUCount; i < sweepCount; ++i, tmp += kBcBackSweepRows)
  {
    int padded = (tmp + kBcBackColumnTile - 1) / kBcBackColumnTile * kBcBackColumnTile;
    countU += padded;
  }
  return countU;
}

long bcBacktransformPackedElementCount(long n)
{
  return packedBcReflectorColumns(n) * kBcBackSweepRows;
}

static void buildBcBacktransformSweepOffsets(long n, std::vector<long> *offsets)
{
  int sweepCount = (n - 1 - 1 + (kBcBackSweepRows - 1)) / (kBcBackSweepRows);
  int lastSweepUCount = n - ((sweepCount - 1) * kBcBackSweepRows + 1) - 1;
  offsets->assign(sweepCount, 0);
  long offset = 0;
  for (int i = 0, tmp = lastSweepUCount; i < sweepCount; ++i, tmp += kBcBackSweepRows)
  {
    (*offsets)[i] = offset;
    int padded = (tmp + kBcBackColumnTile - 1) / kBcBackColumnTile * kBcBackColumnTile;
    offset += static_cast<long>(padded) * kBcBackSweepRows;
  }
}

__global__ void packBcBacktransformReflectorsKernel(long n,
                                          int sweepCount,
                                          int lastSweepUCount,
                                          const long *sweepOffsets,
                                          const double *U,
                                          long ldU,
                                          double *packedU)
{
  long lane = blockIdx.x * blockDim.x + threadIdx.x;
  long packedCol = blockIdx.y * blockDim.y + threadIdx.y;
  int sweepIndex = blockIdx.z;
  if (lane >= kBcBackSweepRows || sweepIndex >= sweepCount) {
    return;
  }

  long totalU = lastSweepUCount + static_cast<long>(sweepIndex) * kBcBackSweepRows;
  long paddedU = (totalU + kBcBackColumnTile - 1) / kBcBackColumnTile * kBcBackColumnTile;
  if (packedCol >= paddedU) {
    return;
  }

  long base = sweepOffsets[sweepIndex] + packedCol * kBcBackSweepRows;
  double value = 0.0;
  if (packedCol < totalU) {
    long sweepBaseRow = (static_cast<long>(sweepCount) - sweepIndex - 1) * kBcBackSweepRows;
    long row = sweepBaseRow + 1 + packedCol + lane;
    if (row < n) {
      value = U[packedCol * ldU + row];
    }
  }
  packedU[base + lane] = value;
}

__global__ void applyPackedBcBacktransformKernel(int n,
                                                              int perBlockN,
                                                              int largeBlockNum,
                                                              int sweepCount,
                                                              int lastSweepUCount,
                                                              const long *sweepOffsets,
                                                              const double *packedU,
                                                              double *dQ,
                                                              long ldQ)
{
  double* sU2 = dynamicSharedMemory;

  __shared__ double stailQ[kBcBackMaxWarpGroups*kBcBackColumnTile];
  __shared__ double stailQW[kBcBackMaxWarpGroups*kBcBackColumnTile];
  __shared__ double sTData[kBcBackMaxWarpGroups*32];

  double rQ[kBcBackRowsPerThread];
  LegacyAlignedDouble4 *rQ4 = (LegacyAlignedDouble4*)rQ;

  int bInx = blockIdx.x;
  if(bInx < largeBlockNum)
  {
    perBlockN += 1;
    dQ = dQ + bInx * perBlockN * ldQ;
  }else
  {
    dQ = dQ + (bInx *perBlockN + largeBlockNum) * ldQ;
  }

  int i = threadIdx.x;
  int j = threadIdx.y;

  for (int sweepIndex = 0; sweepIndex < sweepCount; sweepIndex++)
  {
    long sweepBaseRow = (sweepCount - sweepIndex - 1) * kBcBackSweepRows;
    int totalU = lastSweepUCount + sweepIndex * kBcBackSweepRows;
    long indexU = 0;

    for (; totalU > 0;)
    {
      __syncthreads();

      #pragma unroll
      for(int k =j; k<kBcBackColumnTile;k += kBcBackMaxWarpGroups)
      {
        #pragma unroll
        for(int t =0; t<kBcBackRowsPerThread;t++)
        {
          sU2[k*kBcBackSweepRows+i + t *32] = 0.0;
          if(k < totalU)
          {
            long packedBase = sweepOffsets[sweepIndex] + (indexU + k) * kBcBackSweepRows;
            sU2[k*kBcBackSweepRows+i + t *32] = packedU[packedBase + i*kBcBackRowsPerThread + t];
          }
        }
      }

      __syncthreads();

      for (int k = j; k < perBlockN; k += kBcBackMaxWarpGroups)
      {
        LegacyAlignedDouble4 *tmpDQ4 =
            (LegacyAlignedDouble4*)(dQ+k * ldQ + sweepBaseRow);
        #pragma unroll
        for (int t = 0; t < kBcBackRowsPerThread/4; t++)
        {
          rQ4[t] = tmpDQ4[i*kBcBackRowsPerThread/4 + t];
        }

        __syncwarp();

        #pragma unroll
        for (int t = i; t < kBcBackColumnTile; t +=32)
        {
          stailQ[j*kBcBackColumnTile + t] = dQ[k * ldQ + sweepBaseRow + kBcBackSweepRows + t];
        }

        __syncwarp();

        int h = 0;
        #pragma unroll
        for (; h < kBcBackColumnTile; h++)
        {
          if (0 != i)
          {
            sTData[j*32 + i] = rQ[0];
          }else
          {
            stailQW[j*kBcBackColumnTile+h] = rQ[0];
          }

          __syncwarp();

          #pragma unroll
          for (int t = 0; t < kBcBackRowsPerThread-1; t++)
          {
            rQ[t] = rQ[t+1];
          }

          if(31 != i)
          {
            rQ[kBcBackRowsPerThread-1] = sTData[j*32 + i+1];
          }else{
            rQ[kBcBackRowsPerThread-1] = stailQ[j*kBcBackColumnTile+h];
          }
          __syncwarp();

          double nux = 0.0;
          #pragma unroll
          for (int t = 0; t < kBcBackRowsPerThread; t++)
          {
            nux += sU2[h*kBcBackSweepRows+i + t * 32] * rQ[t];
          }

          nux = warpReduceSum(nux, kBcBackReductionWidth);

          #pragma unroll
          for (int t = 0; t < kBcBackRowsPerThread; t++)
          {
            rQ[t] -= nux * sU2[h*kBcBackSweepRows+i+t * 32];
          }
        }

        #pragma unroll
        for (int t = i; t < kBcBackColumnTile; t +=32)
        {
          dQ[k * ldQ + sweepBaseRow + t] = stailQW[j*kBcBackColumnTile + t];
        }

        tmpDQ4 =
            (LegacyAlignedDouble4*)(dQ+k * ldQ + sweepBaseRow + h);
        #pragma unroll
        for (int t = 0; t < kBcBackRowsPerThread/4; t++)
        {
          tmpDQ4[i*kBcBackRowsPerThread/4 + t] = rQ4[t];
        }
      }

      indexU += kBcBackColumnTile;
      totalU -= kBcBackColumnTile;
      sweepBaseRow += kBcBackColumnTile;

      __syncthreads();
    }
  }
}

int packBcBacktransformReflectors(double *packedU,
                                    const double *U,
                                    long ldU,
                                    long n,
                                    int b,
                                    cudaStream_t stream)
{
  (void)b;
  int sweepCount = (n - 1 - 1 + (kBcBackSweepRows - 1)) / (kBcBackSweepRows);
  int lastSweepUCount = n - ((sweepCount - 1) * kBcBackSweepRows + 1) - 1;
  std::vector<long> hOffsets;
  buildBcBacktransformSweepOffsets(n, &hOffsets);

  long *dOffsets = nullptr;
  cudaError_t err = cudaMalloc(&dOffsets, sizeof(long) * hOffsets.size());
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc sweep offsets failed: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }
  err = cudaMemcpyAsync(dOffsets, hOffsets.data(), sizeof(long) * hOffsets.size(), cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpy sweep offsets failed: " << cudaGetErrorString(err) << std::endl;
    cudaFree(dOffsets);
    return -1;
  }

  long maxPaddedU = 0;
  for (int i = 0, tmp = lastSweepUCount; i < sweepCount; ++i, tmp += kBcBackSweepRows) {
    maxPaddedU = std::max<long>(maxPaddedU, (tmp + kBcBackColumnTile - 1) / kBcBackColumnTile * kBcBackColumnTile);
  }

  dim3 block(32, 8, 1);
  dim3 grid((kBcBackSweepRows + block.x - 1) / block.x,
            (maxPaddedU + block.y - 1) / block.y,
            sweepCount);
  packBcBacktransformReflectorsKernel<<<grid, block, 0, stream>>>(n,
                                                       sweepCount,
                                                       lastSweepUCount,
                                                       dOffsets,
                                                       U,
                                                       ldU,
                                                       packedU);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "packBcBacktransformReflectorsKernel failed: " << cudaGetErrorString(err) << std::endl;
    cudaFree(dOffsets);
    return -1;
  }
  err = cudaStreamSynchronize(stream);
  cudaFree(dOffsets);
  if (err != cudaSuccess) {
    std::cerr << "packBcBacktransformReflectorsKernel sync failed: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }
  return 0;
}

int applyPackedBcBacktransform(double *Q,
                                          long ldQ,
                                          long qCols,
                                          const double *packedU,
                                          const long *packedUOffsets,
                                          long packedUOffsetCount,
                                          long n,
                                          int b,
                                          cudaStream_t stream)
{
  (void)b;
  if (qCols <= 0)
  {
    return 0;
  }

  int sweepCount = (n - 1 - 1 + (kBcBackSweepRows - 1)) / (kBcBackSweepRows);
  int lastSweepUCount = n - ((sweepCount-1)*kBcBackSweepRows+1)-1;
  if (packedUOffsets == nullptr || packedUOffsetCount != sweepCount) {
    std::cerr << "invalid packed-U offset table: expected " << sweepCount
              << " entries, got " << packedUOffsetCount << std::endl;
    return -1;
  }
  const long *dOffsets = packedUOffsets;

  ssize_t shareDyMem = kBcBackColumnTile* kBcBackSweepRows *8;
  cudaFuncSetAttribute(applyPackedBcBacktransformKernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shareDyMem);

  dim3 dimBlock(32, kBcBackMaxWarpGroups, 1);
  int dev = 0;
  cudaGetDevice(&dev);
  int numBlocksPerSm = 0;
  int numThreads = 32 * kBcBackMaxWarpGroups;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm,
      applyPackedBcBacktransformKernel,
      numThreads,
      shareDyMem);
  if (err != cudaSuccess)
  {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  int blockNum = numBlocksPerSm * deviceProp.multiProcessorCount;
  int perBlockN = qCols / blockNum;
  int largeBlockNum = qCols % blockNum;

  const bool profileKernel = profileBcBackKernel();
  if (profileKernel) {
    startTimer();
  }
  void *kernelArgs[] = {(void *)&n,
                        (void *)&perBlockN,
                        (void *)&largeBlockNum,
                        (void *)&sweepCount,
                        (void *)&lastSweepUCount,
                        (void *)&dOffsets,
                        (void *)&packedU,
                        (void *)&Q,
                        (void *)&ldQ};

  dim3 dimGrid(blockNum, 1, 1);
  cudaLaunchCooperativeKernel((void *)applyPackedBcBacktransformKernel,
                              dimGrid,
                              dimBlock,
                              kernelArgs,
                              shareDyMem,
                              stream);
  float kernelMilliseconds = 0.0F;
  if (profileKernel) {
    kernelMilliseconds = stopTimer();
  }
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "packed BC back kernel failed: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  if (profileKernel) {
    printf("packed BC Back %ldx%ld local_cols=%ld u_count=%d max_warp=%d u_col=%d takes %lf ms, tflops is %lf\n",
           n,
           n,
           qCols,
           kBcBackRowsPerThread,
           kBcBackMaxWarpGroups,
           kBcBackColumnTile,
           kernelMilliseconds,
           (2.0 * n * n * qCols ) / (kernelMilliseconds * 1e9));
  }

  return 0;
}
__global__ void applyBcBacktransformKernel(int n,
                                            int perBlockN,
                                            int largeBlockNum,
                                            int sweepCount,
                                            int lastSweepUCount,
                                            double *dCU,
                                            long ldCU,
                                            double *dQ,
                                            long ldQ)
{
  double* sU2 = dynamicSharedMemory;

  __shared__ double stailQ[kBcBackMaxWarpGroups*kBcBackColumnTile];

  __shared__ double stailQW[kBcBackMaxWarpGroups*kBcBackColumnTile];


  __shared__ double sTData[kBcBackMaxWarpGroups*32];

  double rQ[kBcBackRowsPerThread]; 

  LegacyAlignedDouble4 *rQ4 = (LegacyAlignedDouble4*)rQ;

  int bInx = blockIdx.x;
  if(bInx < largeBlockNum)
  {
    perBlockN += 1;
    dQ = dQ + bInx * perBlockN * ldQ;
  }else
  {
    dQ = dQ + (bInx *perBlockN + largeBlockNum) * ldQ;
  }

  int i = threadIdx.x;
  int j = threadIdx.y;


  int sweepIndex;

  int totalU; 

  long sweepBaseRow;
  long indexU = 0;

  #pragma unroll
  for (sweepIndex = 0; sweepIndex < sweepCount; sweepIndex++)
  {
    sweepBaseRow = (sweepCount - sweepIndex - 1) * kBcBackSweepRows;
    // totalU = (kBcBackSweepRows - 2) + sweepIndex * kBcBackSweepRows; 

    totalU = lastSweepUCount + sweepIndex * kBcBackSweepRows; 

    indexU = 0; 

    #pragma unroll
    for (;totalU > 0;)
    {
      __syncthreads();


      #pragma unroll
      for(int k =j; k<kBcBackColumnTile;k += kBcBackMaxWarpGroups)
      {
        #pragma unroll
        for(int t =0; t<kBcBackRowsPerThread;t++)
        {
          sU2[k*kBcBackSweepRows+i + t *32] = 0.0;
          if(k < totalU)
          {
            sU2[k*kBcBackSweepRows+i + t *32] = dCU[(indexU+k)*ldCU + sweepBaseRow + 1 + k + i*kBcBackRowsPerThread +t];
            // sU2[k*kBcBackSweepRows+i + t *32] = dCU[(indexU+k)*kBcBackSweepRows + i +t*32];
          }
        }
      }

      __syncthreads();

      for (int k = j; k < perBlockN; k += kBcBackMaxWarpGroups)
      {

        LegacyAlignedDouble4 *tmpDQ4 =
            (LegacyAlignedDouble4*)(dQ+k * ldQ + sweepBaseRow);
        #pragma unroll
        for (int t = 0; t < kBcBackRowsPerThread/4; t++)
        {
          rQ4[t] = tmpDQ4[i*kBcBackRowsPerThread/4 + t];
        }

        __syncwarp();

        #pragma unroll
        for (int t = i; t < kBcBackColumnTile; t +=32)
        {
          stailQ[j*kBcBackColumnTile + t] = dQ[k * ldQ + sweepBaseRow + kBcBackSweepRows + t];
        }
        

        __syncwarp();

        int h = 0;
        #pragma unroll
        for (; h < kBcBackColumnTile; h++)
        {

          if (0 != i)
          {
            sTData[j*32 + i] = rQ[0];
          }else
          {
            stailQW[j*kBcBackColumnTile+h] = rQ[0];
          }

          __syncwarp();

          #pragma unroll
          for (int t = 0; t < kBcBackRowsPerThread-1; t++)
          {
            rQ[t] = rQ[t+1];
          }

          if(31 != i)
          {
            rQ[kBcBackRowsPerThread-1] = sTData[j*32 + i+1];
          }else{

            rQ[kBcBackRowsPerThread-1] = stailQ[j*kBcBackColumnTile+h];
          }
          __syncwarp();

          double nux = 0.0;

          #pragma unroll
          for (int t = 0; t < kBcBackRowsPerThread; t++)
          {
            nux += sU2[h*kBcBackSweepRows+i + t * 32] * rQ[t];
          }

          nux = warpReduceSum(nux, kBcBackReductionWidth);

          #pragma unroll
          for (int t = 0; t < kBcBackRowsPerThread; t++)
          {
            rQ[t] -= nux * sU2[h*kBcBackSweepRows+i+t * 32];
          }

        }

        #pragma unroll
        for (int t = i; t < kBcBackColumnTile; t +=32)
        {
          // stailQ[j*kBcBackColumnTile + t] = dQ[k * ldQ + sweepBaseRow + kBcBackSweepRows + t];
          dQ[k * ldQ + sweepBaseRow + t] = stailQW[j*kBcBackColumnTile + t];
        }

        
        tmpDQ4 =
            (LegacyAlignedDouble4*)(dQ+k * ldQ + sweepBaseRow + h);
        #pragma unroll
        for (int t = 0; t < kBcBackRowsPerThread/4; t++)
        {
          tmpDQ4[i*kBcBackRowsPerThread/4 + t] = rQ4[t];
        }
      }

      indexU += kBcBackColumnTile;
      totalU -= kBcBackColumnTile;

      sweepBaseRow += kBcBackColumnTile;

      __syncthreads();
    }


  }
}

int applyLocalBcBacktransform(double *Q,
                              long ldQ,
                              long qCols,
                              double *U,
                              long ldU,
                              long n,
                              int b,
                              cudaStream_t stream)
{
  (void)b;
  if (qCols <= 0)
  {
    return 0;
  }

  int sweepCount = (n - 1 - 1 + (kBcBackSweepRows - 1)) / (kBcBackSweepRows);

  int lastSweepUCount = n - ((sweepCount-1)*kBcBackSweepRows+1)-1;

  const std::size_t sharedMemoryBytes =
      static_cast<std::size_t>(kBcBackColumnTile) * kBcBackSweepRows *
      sizeof(double);
  cudaFuncSetAttribute(applyBcBacktransformKernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sharedMemoryBytes);


  dim3 dimBlock(32, kBcBackMaxWarpGroups, 1);
  int dev                = 0;
  cudaGetDevice(&dev);


  int numBlocksPerSm = 0;
  int numThreads = 32 * kBcBackMaxWarpGroups;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                                  applyBcBacktransformKernel,
                                                                  numThreads,
                                                                  sharedMemoryBytes);
  if (err != cudaSuccess)
  {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }
 
  int blockNum = numBlocksPerSm * deviceProp.multiProcessorCount;

  int perBlockN = qCols / blockNum;
  int largeBlockNum = qCols % blockNum;


  const bool profileKernel = profileBcBackKernel();
  if (profileKernel) {
    startTimer();
  }
  void *kernelArgs[] = {(void *)&n,
                        (void *)&perBlockN,
                        (void *)&largeBlockNum,
                        (void *)&sweepCount,
                        (void *)&lastSweepUCount,
                        (void *)&U,
                        (void *)&ldU,
                        (void *)&Q,
                        (void *)&ldQ};

  dim3 dimGrid(blockNum, 1, 1);
  cudaLaunchCooperativeKernel((void *)applyBcBacktransformKernel,
                              dimGrid,
                              dimBlock,
                              kernelArgs,
                              sharedMemoryBytes,
                              stream);

  if (profileKernel) {
    const float elapsedMilliseconds = stopTimer();
    printf("global BC Back %ldx%ld local_cols=%ld takes %lf ms, tflops is %lf\n",
           n,
           n,
           qCols,
           elapsedMilliseconds,
           (2.0 * n * n * qCols ) / (elapsedMilliseconds * 1e9));
  }

  return 0;
}

int applyBcBacktransform(double *Q, long ldQ, double *U, long ldU, long n, int b, cudaStream_t stream)
{
  return applyLocalBcBacktransform(Q, ldQ, n, U, ldU, n, b, stream);
}
