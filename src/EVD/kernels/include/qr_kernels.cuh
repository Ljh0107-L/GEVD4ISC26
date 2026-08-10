#include <cuda_fp16.h>

#pragma once
template <typename T>
static __inline__ __device__ T warpAllReduceSum(T val)
{
  for (int mask = warpSize / 2; mask > 0; mask /= 2)
  {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template <long M, long N>
__global__ void householderQrKernel(long m, long n, half *A, long lda, half *R, long ldr)
{

  using T = half;

  long mm = min(m - blockIdx.x * M, M);

  if (0 >= mm)
  {
    return;
  }

  A = A + blockIdx.x * M;
  R = R + blockIdx.x * N;

  long nn = min(N, n);
  
  if(N != n)
  {
    return;
  }


  long i = threadIdx.x;
  long j = threadIdx.y;

  __shared__ T AA[M * N], RR[N];
  long ldaa = mm;

  for(long h = j; h < nn; h += blockDim.y)
  {
    for(long k = i; k < mm; k += blockDim.x)
    {
      AA[k + h * ldaa] = A[k + h * lda];
    }
  }

  __syncthreads();


  for (long cols = 0; cols < nn; cols++)
  {

    if (j == cols % blockDim.y)
    {

      T nu = 0.0;

#pragma unroll
      for (long k = i; k < mm; k += blockDim.x)
      {
        if( k >= cols)
        {
          nu += AA[k + cols*ldaa] * AA[k + cols*ldaa];
        }
      }

      T norm_x_squre = warpAllReduceSum(nu);
      T norm_x       = hsqrt(norm_x_squre);

      const bool nonzero_norm = norm_x != half(0);
      T scale = nonzero_norm ? __float2half(1.0) / norm_x : half(0);
#pragma unroll
      for (long k = i; k < mm; k += blockDim.x)
      {
        if( k >= cols)
        {
          AA[k + cols*ldaa] *=scale;
        }
      }


      if (0 == i)
      {
        if (nonzero_norm) {
          T u1 = AA[cols + cols * mm];
          AA[cols + cols * ldaa] += (u1 >= half(0)) ? 1 : -1;
          RR[cols] = (u1 >= half(0)) ? -norm_x : norm_x;
        } else {
          RR[cols] = half(0);
        }
      }

      __syncwarp();


      scale = nonzero_norm
                  ? half(1) / (hsqrt(__habs(AA[cols + cols * ldaa])))
                  : half(0);
#pragma unroll
      for(long k = i; k < mm; k += blockDim.x)
      {
        if(k >= cols)
        {
          AA[k + cols * ldaa] *= scale;
        }
      }
    }

    __syncthreads();

    #pragma unroll
    for(long h = j; h < nn; h += blockDim.y)
    {
      if(h > cols)
      {
        T nu = 0.0;
        #pragma unroll
        for(long k = i; k < mm; k += blockDim.x)
        {
          if(k >= cols)
          {
            nu += AA[k + cols * ldaa] * AA[k + h * ldaa];
          }
        }
        T utx = warpAllReduceSum(nu);

        #pragma unroll
        for(long k = i; k < mm; k += blockDim.x)
        {
          if(k >= cols)
          {
            AA[k + h * ldaa] -= utx*AA[k + cols * ldaa];
          }
        }
        // __syncwarp();
      }
    }

  }

  __syncthreads();

  #pragma unroll
  for(long h = j; h < nn; h += blockDim.y)
  {
    #pragma unroll
    for(long k = i; k < nn; k += blockDim.x)
    {
      if(k < h)
      {
        R[k + h*ldr] = AA[k + h*ldaa];
        AA[k + h*ldaa] = 0.0;
      }else if ( k > h)
      {
        R[k + h*ldr] = 0.0;
      }else
      {
        R[h + h*ldr] = RR[h];
      }
    }
  }


  #define MAX_THREAD_PROC_Q_ELEMENTS 8
  T q[MAX_THREAD_PROC_Q_ELEMENTS];

  #pragma unroll
  for(long h = j; h < nn; h += blockDim.y)
  {
    #pragma unroll
    for(long k = i, t = 0; k< mm; k += blockDim.x, ++t)
    {
      q[t] = 0.0;
      if(h == k)
      {
        q[t] = 1.0;
      }
    }
    __syncwarp();

    #pragma unroll
    for(long hh = h; hh >= 0; --hh)
    {
      T nu = 0.0;
      #pragma unroll
      for(long k = i, t=0; k < mm; k += blockDim.x, ++t)
      {
        nu += q[t] * AA[k + hh*ldaa];
      }

      T utx = warpAllReduceSum(nu);

      #pragma unroll
      for(long k = i, t = 0; k < mm; k += blockDim.x, ++t)
      {
        q[t] -= utx * AA[k + hh*ldaa];
      }
      // __syncwarp();
    }

    #pragma unroll
    for(long k = i, t=0; k < mm; k += blockDim.x, ++t)
    {
      A[k + h*lda] = q[t];
    }
    // __syncwarp();
  }

}




template <long M, long N>
__global__ void householderQrKernel(long m, long n, double *A, long lda, double *R, long ldr)
{

  using T = double;


  long mm = min(m - blockIdx.x * M, M);

  if (0 >= mm)
  {
    return;
  }

  A = A + blockIdx.x * M;
  R = R + blockIdx.x * N;


  long nn = min(N, n);
  
  if(N != n)
  {
    return;
  }


  long i = threadIdx.x;
  long j = threadIdx.y;


  __shared__ T AA[M * N], RR[N];
  long ldaa = mm;


  for(long h = j; h < nn; h += blockDim.y)
  {
    for(long k = i; k < mm; k += blockDim.x)
    {
      AA[k + h * ldaa] = A[k + h * lda];
    }
  }


  __syncthreads();

  for (long cols = 0; cols < nn; cols++)
  {


    if (j == cols % blockDim.y)
    {

      T nu = 0.0;

#pragma unroll
      for (long k = i; k < mm; k += blockDim.x)
      {
        if( k >= cols)
        {
          nu += AA[k + cols*ldaa] * AA[k + cols*ldaa];
        }
      }

      T norm_x_squre = warpAllReduceSum(nu);
      T norm_x       = sqrt(norm_x_squre);

      const bool nonzero_norm = norm_x != 0.0;
      T scale = nonzero_norm ? 1.0 / norm_x : 0.0;
#pragma unroll
      for (long k = i; k < mm; k += blockDim.x)
      {
        if( k >= cols)
        {
          AA[k + cols*ldaa] *=scale;
        }
      }


      if (0 == i)
      {
        if (nonzero_norm) {
          T u1 = AA[cols + cols * mm];
          AA[cols + cols * ldaa] += (u1 >= 0) ? 1 : -1;
          RR[cols] = (u1 >= 0) ? -norm_x : norm_x;
        } else {
          RR[cols] = 0.0;
        }
      }

      __syncwarp();

      scale = nonzero_norm ? 1 / (sqrt(abs(AA[cols + cols * ldaa]))) : 0.0;
#pragma unroll
      for(long k = i; k < mm; k += blockDim.x)
      {
        if(k >= cols)
        {
          AA[k + cols * ldaa] *= scale;
        }
      }
    }

    __syncthreads();

    #pragma unroll
    for(long h = j; h < nn; h += blockDim.y)
    {

      if(h > cols)
      {
        T nu = 0.0;
        #pragma unroll
        for(long k = i; k < mm; k += blockDim.x)
        {
          if(k >= cols)
          {
            nu += AA[k + cols * ldaa] * AA[k + h * ldaa];
          }
        }
        T utx = warpAllReduceSum(nu);

        #pragma unroll
        for(long k = i; k < mm; k += blockDim.x)
        {
          if(k >= cols)
          {
            AA[k + h * ldaa] -= utx*AA[k + cols * ldaa];
          }
        }

      }
    }

  }

  __syncthreads();

  #pragma unroll
  for(long h = j; h < nn; h += blockDim.y)
  {
    #pragma unroll
    for(long k = i; k < nn; k += blockDim.x)
    {
      if(k < h)
      {
        R[k + h*ldr] = AA[k + h*ldaa];
        AA[k + h*ldaa] = 0.0;
      }else if ( k > h)
      {
        R[k + h*ldr] = 0.0;
      }else
      {
        R[h + h*ldr] = RR[h];
      }
    }
  }

  #define MAX_THREAD_PROC_Q_ELEMENTS 8
  T q[MAX_THREAD_PROC_Q_ELEMENTS];

  #pragma unroll
  for(long h = j; h < nn; h += blockDim.y)
  {
    #pragma unroll
    for(long k = i, t = 0; k< mm; k += blockDim.x, ++t)
    {
      q[t] = 0.0;
      if(h == k)
      {
        q[t] = 1.0;
      }
    }
    __syncwarp();

    #pragma unroll
    for(long hh = h; hh >= 0; --hh)
    {
      T nu = 0.0;
      #pragma unroll
      for(long k = i, t=0; k < mm; k += blockDim.x, ++t)
      {
        nu += q[t] * AA[k + hh*ldaa];
      }

      T utx = warpAllReduceSum(nu);

      #pragma unroll
      for(long k = i, t = 0; k < mm; k += blockDim.x, ++t)
      {
        q[t] -= utx * AA[k + hh*ldaa];
      }
      // __syncwarp();
    }

    #pragma unroll
    for(long k = i, t=0; k < mm; k += blockDim.x, ++t)
    {
      A[k + h*lda] = q[t];
    }
    // __syncwarp();
  }

}

template <long M, long N>
__global__ void householderQrKernel(long m, long n, float *A, long lda, float *R, long ldr)
{

  using T = float;

  long mm = min(m - blockIdx.x * M, M);

  if (0 >= mm)
  {
    return;
  }

  A = A + blockIdx.x * M;
  R = R + blockIdx.x * N;

  long nn = min(N, n);
  

  if(N != n)
  {
    return;
  }


  long i = threadIdx.x;
  long j = threadIdx.y;


  __shared__ T AA[M * N], RR[N];
  long ldaa = mm;


  for(long h = j; h < nn; h += blockDim.y)
  {
    for(long k = i; k < mm; k += blockDim.x)
    {
      AA[k + h * ldaa] = A[k + h * lda];
    }
  }

  __syncthreads();


  for (long cols = 0; cols < nn; cols++)
  {


    if (j == cols % blockDim.y)
    {

      T nu = 0.0;

#pragma unroll
      for (long k = i; k < mm; k += blockDim.x)
      {
        if( k >= cols)
        {
          nu += AA[k + cols*ldaa] * AA[k + cols*ldaa];
        }
      }

      T norm_x_squre = warpAllReduceSum(nu);
      T norm_x       = sqrt(norm_x_squre);

      const bool nonzero_norm = norm_x != 0.0;
      T scale = nonzero_norm ? 1.0 / norm_x : 0.0;
#pragma unroll
      for (long k = i; k < mm; k += blockDim.x)
      {
        if( k >= cols)
        {
          AA[k + cols*ldaa] *=scale;
        }
      }


      if (0 == i)
      {
        if (nonzero_norm) {
          T u1 = AA[cols + cols * mm];
          AA[cols + cols * ldaa] += (u1 >= 0) ? 1 : -1;
          RR[cols] = (u1 >= 0) ? -norm_x : norm_x;
        } else {
          RR[cols] = 0.0;
        }
      }

      __syncwarp();

      scale = nonzero_norm ? 1 / (sqrt(abs(AA[cols + cols * ldaa]))) : 0.0;
#pragma unroll
      for(long k = i; k < mm; k += blockDim.x)
      {
        if(k >= cols)
        {
          AA[k + cols * ldaa] *= scale;
        }
      }
    }

    __syncthreads();

    #pragma unroll
    for(long h = j; h < nn; h += blockDim.y)
    {

      if(h > cols)
      {
        T nu = 0.0;
        #pragma unroll
        for(long k = i; k < mm; k += blockDim.x)
        {
          if(k >= cols)
          {
            nu += AA[k + cols * ldaa] * AA[k + h * ldaa];
          }
        }
        T utx = warpAllReduceSum(nu);

        #pragma unroll
        for(long k = i; k < mm; k += blockDim.x)
        {
          if(k >= cols)
          {
            AA[k + h * ldaa] -= utx*AA[k + cols * ldaa];
          }
        }

      }
    }

  }

  __syncthreads();

  #pragma unroll
  for(long h = j; h < nn; h += blockDim.y)
  {
    #pragma unroll
    for(long k = i; k < nn; k += blockDim.x)
    {
      if(k < h)
      {
        R[k + h*ldr] = AA[k + h*ldaa];
        AA[k + h*ldaa] = 0.0;
      }else if ( k > h)
      {
        R[k + h*ldr] = 0.0;
      }else
      {
        R[h + h*ldr] = RR[h];
      }
    }
  }

  #define MAX_THREAD_PROC_Q_ELEMENTS 8
  T q[MAX_THREAD_PROC_Q_ELEMENTS];

  #pragma unroll
  for(long h = j; h < nn; h += blockDim.y)
  {
    #pragma unroll
    for(long k = i, t = 0; k< mm; k += blockDim.x, ++t)
    {
      q[t] = 0.0;
      if(h == k)
      {
        q[t] = 1.0;
      }
    }
    __syncwarp();

    #pragma unroll
    for(long hh = h; hh >= 0; --hh)
    {
      T nu = 0.0;
      #pragma unroll
      for(long k = i, t=0; k < mm; k += blockDim.x, ++t)
      {
        nu += q[t] * AA[k + hh*ldaa];
      }

      T utx = warpAllReduceSum(nu);

      #pragma unroll
      for(long k = i, t = 0; k < mm; k += blockDim.x, ++t)
      {
        q[t] -= utx * AA[k + hh*ldaa];
      }

    }

    #pragma unroll
    for(long k = i, t=0; k < mm; k += blockDim.x, ++t)
    {
      A[k + h*lda] = q[t];
    }
    // __syncwarp();
  }

}
