
#pragma once

#include <cooperative_groups.h>

#include "tuning/bc_backtransform_config.hpp"

namespace cg = cooperative_groups;

using gevd4isc26::evd::kBcBackSweepRows;

static __inline__ __device__ double warpAllReduceSum(double val)
{
  for (int mask = warpSize / 2; mask > 0; mask /= 2)
  {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}


__device__ int bcStopFlag = 0;
__device__ int bcActiveBlockCount = 0;

template <int BandWidth, bool UseGlobalStop>
__device__ void chaseBulgesKernelBody(int n,
                                                int b,
                                                double *subA,
                                                int ldSubA,
                                                double *dU,
                                                long ldU,
                                                double *packedU,
                                                const long *packedUOffsets,
                                                int packedUSweepCount,
                                                int packedN,
                                                int packedSweepOffset,
                                                int sweepStart,
                                                int sweepEnd,
                                                int segmentRowStart,
                                                int segmentRowEnd,
                                                int segmentStopSweep,
                                                int blockNum,
                                                int *com)
{
  auto grid  = cg::this_grid();

  int Nx = blockDim.x;
  int Ny = blockDim.y;

  if (BandWidth != b)
  {
    return;
  }

  int warpGroupThdCount = Ny / 2;

  int bInx = sweepStart + grid.block_rank();

  int i = threadIdx.x;
  int j = threadIdx.y;

  __shared__ double u[BandWidth];

  __shared__ double S1[BandWidth * BandWidth]; 
  __shared__ double S2[BandWidth * BandWidth]; 
  int ldSS = b;

  double nu;
  double utx;

  const bool singleStepRows = (segmentRowStart == -3);
  const bool slantedDependencyRows = (segmentRowStart == -6);
  const bool multiStepRows = (segmentRowStart == -4) || slantedDependencyRows;
  const bool flatMultiStepRows = (segmentRowStart == -5);
  const bool multiStepLikeRows = multiStepRows || flatMultiStepRows;
  // -2 resumes a full-range launch from com[] but has no segment boundary.
  // Keeping it separate avoids testing the segment frontier in every chasing
  // step after a stage-pipeline hand-off.
  const bool resumeFullRange = (segmentRowStart == -2);
  const bool forceSegmentRows =
      singleStepRows || multiStepRows || flatMultiStepRows;
  const bool useSegmentRows =
      forceSegmentRows || (segmentRowStart >= 0) || (segmentRowEnd < n);
  const bool useSlantedSegmentEnd =
      (!singleStepRows) && (!flatMultiStepRows) && (segmentRowStart >= 0 || multiStepRows);
  const int boundedSegmentEnd = max(0, min(segmentRowEnd, n));

  for (; 0 == bcStopFlag; bInx += blockNum)
  {
    int opColB1;
    int opRowB1;

    double *uB = (dU != nullptr) ? (dU + bInx * ldU) : nullptr;
    const int stopSweep = min(sweepEnd, n - 2) - 1;
    const int slantStopSweep =
        (segmentStopSweep >= 0) ? min(segmentStopSweep, n - 3) : stopSweep;
    int sweepSegmentEnd = boundedSegmentEnd;
    int dependencySegmentEnd = boundedSegmentEnd;
    if (useSegmentRows && useSlantedSegmentEnd)
    {
      const int sweepLag = max(0, slantStopSweep - bInx);
      const long slantedEnd = static_cast<long>(boundedSegmentEnd) +
                              static_cast<long>(2 * b) * static_cast<long>(sweepLag);
      sweepSegmentEnd = static_cast<int>(min(static_cast<long>(n), slantedEnd));
      dependencySegmentEnd = sweepSegmentEnd;
      if (slantedDependencyRows)
      {
        const long previousSlantedEnd = slantedEnd + static_cast<long>(2 * b);
        dependencySegmentEnd =
            static_cast<int>(min(static_cast<long>(n), previousSlantedEnd));
      }
    }

    long opRow = bInx + 1;
    bool skipCurrentSweep = false;
    bool skipRequestsStop = false;
    const bool inSweepRange = (bInx < sweepEnd) && (bInx < n - 2);
    if ((resumeFullRange || useSegmentRows) && inSweepRange)
    {
      int resumeRow = com[bInx];
      if (resumeRow >= n + 3 * b)
      {
        skipCurrentSweep = true;
        skipRequestsStop = (stopSweep == bInx);
      }
      else
      {
        if (resumeRow > opRow)
        {
          opRow = resumeRow;
        }
        if (useSegmentRows && opRow >= sweepSegmentEnd)
        {
          skipCurrentSweep = true;
          skipRequestsStop = (stopSweep == bInx);
        }
      }
    }

    int rowB1 = skipCurrentSweep ? 0 : min(b, (int)(n - opRow));
    int colB1 = 1;
    if (!skipCurrentSweep && opRow > bInx + 1)
    {
      colB1 = min(b, max(1, (int)(opRow - bInx - 1)));
    }

    double *B1 = skipCurrentSweep ? subA : (subA + colB1 + (opRow - colB1) * ldSubA);

    bool firstFlag = true;
    bool cycFlag   = true;
    bool completedSweep = false;

    int rowS;
    int colS;

    double *S; 

    while (cycFlag && 0 == bcStopFlag)
    {
      const bool activeSweep = (!skipCurrentSweep) && (bInx < sweepEnd) && (bInx < n - 2);
      const bool dependencyBlocked =
          activeSweep && (0 != bInx) && (opRow + 2 * b > com[bInx - 1]);
      bool pausedBySegmentDependency = false;
      if (multiStepRows)
      {
        if ((0 == i) && (0 == j) && (0 == grid.block_rank()))
        {
          bcActiveBlockCount = 0;
        }
        grid.sync();
      }
      if (singleStepRows && !activeSweep)
      {
        cycFlag = false;
      }
      if (skipCurrentSweep)
      {
        if ((!multiStepLikeRows) && skipRequestsStop && (0 == i) && (0 == j))
        {
          bcStopFlag = 1;
        }
        cycFlag = false;
      }
      if (useSegmentRows && dependencyBlocked && (!useSlantedSegmentEnd) &&
          com[bInx - 1] >= sweepSegmentEnd)
      {
        if ((0 == i) && (0 == j))
        {
          com[bInx] = opRow;
          if (!multiStepLikeRows)
          {
            bcStopFlag = 1;
          }
        }
        pausedBySegmentDependency = true;
        __syncthreads();
      }
      else if (useSegmentRows && dependencyBlocked && com[bInx - 1] >= dependencySegmentEnd)
      {
        if ((0 == i) && (0 == j))
        {
          com[bInx] = opRow;
          if ((!multiStepLikeRows) && stopSweep == bInx)
          {
            bcStopFlag = 1;
          }
        }
        pausedBySegmentDependency = true;
        __syncthreads();
      }
      else if (singleStepRows && dependencyBlocked)
      {
        if ((0 == i) && (0 == j))
        {
          com[bInx] = opRow;
        }
        pausedBySegmentDependency = true;
        __syncthreads();
      }

      if (pausedBySegmentDependency)
      {
        cycFlag = false;
      }

      const bool doActiveWork =
          activeSweep && (false == dependencyBlocked) && (false == pausedBySegmentDependency);
      if (multiStepRows)
      {
        if (doActiveWork && (0 == i) && (0 == j))
        {
          atomicAdd(&bcActiveBlockCount, 1);
        }
        grid.sync();
        if ((0 == i) && (0 == j) && (0 == grid.block_rank()) && bcActiveBlockCount == 0)
        {
          bcStopFlag = 1;
        }
        grid.sync();
        if (0 != bcStopFlag)
        {
          cycFlag = false;
        }
      }

      if (doActiveWork)
      {
        if (true == firstFlag)
        {
#pragma unroll
          for (opColB1 = j; opColB1 < colB1; opColB1 += Ny)
          {
#pragma unroll
            for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
            {
              S1[opRowB1 + opColB1 * ldSS] = B1[(opRowB1 - opColB1) + opColB1 * ldSubA];
            }
          }

          rowS = 0;
          colS = 0;
        }

        firstFlag = false;
        
        __syncthreads();


        if (0 != j)
        {

#pragma unroll
          for (int opColS = j - 1; opColS < colS; opColS += (Ny - 1))
          {
#pragma unroll
            for (int opRowS = i; (opColS <= opRowS) && (opRowS < rowS); opRowS += Nx)
            {
              S[(opRowS - opColS) + opColS * ldSubA] = S2[opRowS + opColS * ldSS];
            }
          }

          colS = rowS = rowB1;
          S           = subA + opRow * ldSubA;

#pragma unroll
          for (int opColS = j - 1; opColS < colS; opColS += (Ny - 1))
          {
#pragma unroll
            for (int opRowS = i; (opColS <= opRowS) && (opRowS < rowS); opRowS += Nx)
            {

              S2[opRowS + opColS * ldSS] = S[(opRowS - opColS) + opColS * ldSubA];

              S2[opColS + opRowS * ldSS] = S2[opRowS + opColS * ldSS];
            }
          }

        }
        else
        {

#pragma unroll
          for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
          {
            u[opRowB1] = S1[opRowB1];
          }

          __syncwarp();

          nu = 0.0;

#pragma unroll
          for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
          {
            //  u[opRowB1] = B1[opRowB1][0]
            nu += u[opRowB1] * u[opRowB1];
          }

          double norm_x_squre = warpAllReduceSum(nu);
          double norm_x       = sqrt(norm_x_squre);

          // A sweep over an already tridiagonal (or diagonal) matrix can see
          // an exactly zero reflector.  In that case the Householder
          // transformation is the identity: keep u at zero instead of
          // dividing by zero and contaminating the tridiagonal with NaNs.
          const bool nonzero_norm = norm_x != 0.0;
          double scale = nonzero_norm ? 1.0 / norm_x : 0.0;
#pragma unroll
          for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
          {
            //  u[opRowB1] = B1[opRowB1][0]
            u[opRowB1] *= scale;
          }

          __syncwarp();


          if (0 == i)
          {
            if (nonzero_norm)
            {
              double u1 = u[0];
              u[0] += (u1 >= 0) ? 1 : -1;
            }

          }

          __syncwarp();

          scale = nonzero_norm ? 1 / (sqrt(abs(u[0]))) : 0.0;



#pragma unroll
          for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
          {
            //  u[opRowB1] = B1[opRowB1][0]
            u[opRowB1] *= scale;
          }


#pragma unroll
          for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
          {
            //  u[opRowB1] = B1[opRowB1][0]
            long uRow = opRow + opRowB1;
            if (packedU != nullptr)
            {
              long globalSweep = static_cast<long>(packedSweepOffset) + static_cast<long>(bInx);
              long globalURow = static_cast<long>(packedSweepOffset) + uRow;
              long delta = globalURow - globalSweep - 1;
              if (delta >= 0)
              {
                long sweepBaseRow = (delta / kBcBackSweepRows) * kBcBackSweepRows;
                int sweepIndex = packedUSweepCount - 1 - static_cast<int>(sweepBaseRow / kBcBackSweepRows);
                long lane = delta - sweepBaseRow;
                long totalU = static_cast<long>(packedN) - sweepBaseRow - 2;
                if (sweepIndex >= 0 && sweepIndex < packedUSweepCount &&
                    lane < kBcBackSweepRows && globalSweep < totalU)
                {
                  packedU[packedUOffsets[sweepIndex] + globalSweep * kBcBackSweepRows + lane] = u[opRowB1];
                }
              }
            }
            else
            {
              uB[uRow] = u[opRowB1];
            }
          }

          colS = rowS = rowB1;
          S           = subA + opRow * ldSubA;
        }

        __syncthreads();


        __syncthreads();

#pragma unroll
        for (opColB1 = j; opColB1 < colB1; opColB1 += Ny)
        {
          nu = 0.0;
#pragma unroll
          for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
          {
            nu += u[opRowB1] * S1[opRowB1 + opColB1 * ldSS];
          }

          utx = warpAllReduceSum(nu);

#pragma unroll
          for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
          {
            S1[opRowB1 + opColB1 * ldSS] -= utx * u[opRowB1];
          }

          __syncwarp();
        }

        __syncthreads();


        if (j < warpGroupThdCount)
        {
#pragma unroll
          for (opColB1 = j; opColB1 < colB1; opColB1 += warpGroupThdCount)
          {
#pragma unroll
            for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
            {
              B1[(opRowB1 - opColB1) + opColB1 * ldSubA] = S1[opRowB1 + opColB1 * ldSS];

            }
          }

          opRow += rowB1;                   
          rowB1 = min(b, (int)(n - opRow)); 
          colB1 = colS;
          B1    = subA + colB1 + (opRow - colB1) * ldSubA;

#pragma unroll
          for (opColB1 = j; opColB1 < colB1; opColB1 += warpGroupThdCount)
          {
#pragma unroll
            for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
            {
              S1[opRowB1 + opColB1 * ldSS] = B1[(opRowB1 - opColB1) + opColB1 * ldSubA];
            }
          }

        }
        else
        {
#pragma unroll
          for (int opColS = j - warpGroupThdCount; opColS < colS; opColS += warpGroupThdCount)
          {
            nu = 0.0;
#pragma unroll
            for (int opRowS = i; opRowS < rowS; opRowS += Nx)
            {
              nu += u[opRowS] * S2[opRowS + opColS * ldSS];
            }

            utx = warpAllReduceSum(nu);

#pragma unroll
            for (int opRowS = i; opRowS < rowS; opRowS += Nx)
            {
              S2[opRowS + opColS * ldSS] -= utx * u[opRowS];
            }

            __syncwarp();
          }

          opRow += rowB1;                 
          rowB1 = min(b, (int)(n - opRow)); 
          colB1 = colS;
          B1    = subA + colB1 + (opRow - colB1) * ldSubA;

        }

        __syncthreads();
#pragma unroll
        for (int opRowS = j; opRowS < rowS; opRowS += Ny)
        {
          nu = 0.0;
#pragma unroll
          for (int opColS = i; opColS < colS; opColS += Nx)
          {
            nu += u[opColS] * S2[opRowS + opColS * ldSS];
          }

          utx = warpAllReduceSum(nu);

#pragma unroll
          for (int opColS = i; opColS < colS; opColS += Nx)
          {
            S2[opRowS + opColS * ldSS] -= utx * u[opColS];
          }

          __syncwarp();
        }

        __syncthreads();


#pragma unroll
        for (opRowB1 = j; opRowB1 < rowB1; opRowB1 += Ny)
        {
          nu = 0.0;
#pragma unroll
          for (opColB1 = i; opColB1 < colB1; opColB1 += Nx)
          {
            nu += u[opColB1] * S1[opRowB1 + opColB1 * ldSS];
          }


          utx = warpAllReduceSum(nu);

#pragma unroll
          for (opColB1 = i; opColB1 < colB1; opColB1 += Nx)
          {
            S1[opRowB1 + opColB1 * ldSS] -= utx * u[opColB1];
          }


          __syncwarp();
        }

        __syncthreads();


        bool pauseForSegment = false;
        if (singleStepRows)
        {
          pauseForSegment = true;
        }
        else if (useSegmentRows && rowB1 > 1 && opRow >= sweepSegmentEnd)
        {
          pauseForSegment = true;
        }

        if (rowB1 <= 1 || pauseForSegment)
        {

#pragma unroll
          for (int opColS = j; opColS < colS; opColS += Ny)
          {
#pragma unroll
            for (int opRowS = i; (opColS <= opRowS) && (opRowS < rowS); opRowS += Nx)
            {
              S[(opRowS - opColS) + opColS * ldSubA] = S2[opRowS + opColS * ldSS];
            }
          }

#pragma unroll
          for (int opColB1 = j; opColB1 < colB1; opColB1 += Ny)
          {
#pragma unroll
            for (int opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
            {
              B1[(opRowB1 - opColB1) + opColB1 * ldSubA] = S1[opRowB1 + opColB1 * ldSS];

            }
          }

          __syncthreads();

          completedSweep = (rowB1 <= 1);

          if ((!multiStepLikeRows) && (stopSweep == bInx) && 0 == threadIdx.x && 0 == threadIdx.y)
          {
            bcStopFlag = 1;
          }
          __syncthreads();

          cycFlag = false;
        }

        if ((0 == i) && (0 == j))
        {
          com[bInx] = opRow;

          if (false == cycFlag && completedSweep)
          {
            com[bInx] = n + 3 * b;
          }
        }

        __syncthreads();
      }

      grid.sync();
      if (singleStepRows && (0 == i) && (0 == j) && (0 == grid.block_rank()))
      {
        bcStopFlag = 1;
      }
      grid.sync();
    }
  }
}

template <int BandWidth>
__global__ void chaseBulgesKernel(int n,
                                           int b,
                                           double *subA,
                                           int ldSubA,
                                           double *dU,
                                           long ldU,
                                           int blockNum,
                                           int *com)
{
  chaseBulgesKernelBody<BandWidth, true>(n,
                                                   b,
                                                   subA,
                                                   ldSubA,
                                                   dU,
                                                   ldU,
                                                   nullptr,
                                                   nullptr,
                                                   0,
                                                   n,
                                                   0,
                                                   0,
                                                   n - 2,
                                                   -1,
                                                   n,
                                                   -1,
                                                   blockNum,
                                                   com);
}

template <int BandWidth>
__global__ void chaseBulgesPackedKernel(int n,
                                                   int b,
                                                   double *subA,
                                                   int ldSubA,
                                                   double *packedU,
                                                   const long *packedUOffsets,
                                                   int packedUSweepCount,
                                                   int blockNum,
                                                   int *com)
{
  chaseBulgesKernelBody<BandWidth, true>(n,
                                                   b,
                                                   subA,
                                                   ldSubA,
                                                   nullptr,
                                                   0,
                                                   packedU,
                                                   packedUOffsets,
                                                   packedUSweepCount,
                                                   n,
                                                   0,
                                                   0,
                                                   n - 2,
                                                   -1,
                                                   n,
                                                   -1,
                                                   blockNum,
                                                   com);
}

template <int BandWidth>
__global__ void chaseBulgesRangePackedKernel(int n,
                                                         int b,
                                                         double *subA,
                                                         int ldSubA,
                                                         double *packedU,
                                                         const long *packedUOffsets,
                                                         int packedUSweepCount,
                                                         int sweepStart,
                                                         int sweepEnd,
                                                         int blockNum,
                                                         int *com)
{
  chaseBulgesKernelBody<BandWidth, false>(n,
                                                    b,
                                                    subA,
                                                    ldSubA,
                                                    nullptr,
                                                    0,
                                                    packedU,
                                                    packedUOffsets,
                                                    packedUSweepCount,
                                                    n,
                                                    0,
                                                    sweepStart,
                                                    sweepEnd,
                                                    -1,
                                                    n,
                                                    -1,
                                                    blockNum,
                                                    com);
}

template <int BandWidth>
__global__ void chaseBulgesSegmentPackedKernel(int n,
                                                           int b,
                                                           double *subA,
                                                           int ldSubA,
                                                           double *packedU,
                                                           const long *packedUOffsets,
                                                           int packedUSweepCount,
                                                           int sweepStart,
                                                           int sweepEnd,
                                                           int segmentRowStart,
                                                           int segmentRowEnd,
                                                           int segmentStopSweep,
                                                           int blockNum,
                                                           int *com)
{
  chaseBulgesKernelBody<BandWidth, false>(n,
                                                    b,
                                                    subA,
                                                    ldSubA,
                                                    nullptr,
                                                    0,
                                                    packedU,
                                                    packedUOffsets,
                                                    packedUSweepCount,
                                                    n,
                                                    0,
                                                    sweepStart,
                                                    sweepEnd,
                                                    segmentRowStart,
                                                    segmentRowEnd,
                                                    segmentStopSweep,
                                                    blockNum,
                                                    com);
}

template <int BandWidth>
__global__ void chaseBulgesLocalPackedKernel(int localN,
                                                         int globalN,
                                                         int globalSweepOffset,
                                                         int b,
                                                         double *subA,
                                                         int ldSubA,
                                                         double *packedU,
                                                         const long *packedUOffsets,
                                                         int packedUSweepCount,
                                                         int sweepStart,
                                                         int sweepEnd,
                                                         int blockNum,
                                                         int *com)
{
  chaseBulgesKernelBody<BandWidth, false>(localN,
                                                    b,
                                                    subA,
                                                    ldSubA,
                                                    nullptr,
                                                    0,
                                                    packedU,
                                                    packedUOffsets,
                                                    packedUSweepCount,
                                                    globalN,
                                                    globalSweepOffset,
                                                    sweepStart,
                                                    sweepEnd,
                                                    -1,
                                                    localN,
                                                    -1,
                                                    blockNum,
                                                    com);
}
