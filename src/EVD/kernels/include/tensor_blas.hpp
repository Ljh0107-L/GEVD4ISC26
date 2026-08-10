#include <cuda_fp16.h>

template <unsigned Size>
__device__ inline void cp(void *const dst, const void *const src)
{
  static_assert(Size == 4 || Size == 8 || Size == 16, "Size must be one of 4, 8 and 16");
  if (Size == 4)
  {
    *(reinterpret_cast<uint32_t *>(dst)) = *(reinterpret_cast<const uint32_t *>(src));
  }
  else if (Size == 8)
  {
    *(reinterpret_cast<uint64_t *>(dst)) = *(reinterpret_cast<const uint64_t *>(src));
  }
  else
  {
    *(reinterpret_cast<ulong2 *>(dst)) = *(reinterpret_cast<const ulong2 *>(src));
  }
}

template <class IdxT, unsigned block_size, unsigned smem_len = block_size * 8>
__global__ void
s2h_swpipe(const IdxT m, const IdxT n, const float *const as, int ldas, __half *ah, int ldah)
{
  __shared__ float smem_f32[smem_len];
  __shared__ half smem_f16[smem_len];

  const auto in = blockIdx.x;

  for (unsigned i = 0; i < m; i += smem_len)
  {
    if (i + smem_len <= m)
    {
      // Load FP32 elements
      if (reinterpret_cast<long>(ah) % 16 == 0 && ldah % 4 == 0)
      {
        for (unsigned j = 0; j < smem_len; j += block_size * 4)
        {
          const auto smem_i = j + threadIdx.x * 4;
          if (smem_len < block_size * 4 && smem_i >= smem_len)
            break;
          const auto im = i + smem_i;
          cp<16>(&smem_f32[smem_i], &as[im + ldas * in]);
        }
        __syncthreads();
      }
      else if (reinterpret_cast<long>(ah) % 8 == 0 && ldah % 2 == 0)
      {
        for (unsigned j = 0; j < smem_len; j += block_size * 2)
        {
          const auto smem_i = j + threadIdx.x * 2;
          if (smem_len < block_size * 2 && smem_i >= smem_len)
            break;
          const auto im = i + smem_i;
          cp<8>(&smem_f32[smem_i], &as[im + ldas * in]);
        }
        __syncthreads();
      }
      else
      {
        for (unsigned j = 0; j < smem_len; j += block_size)
        {
          const auto smem_i = j + threadIdx.x;
          const auto im     = i + smem_i;
          cp<4>(&smem_f32[smem_i], &as[im + ldas * in]);
        }
      }
      // Convert to FP16
      for (unsigned j = 0; j < smem_len; j += block_size)
      {
        const auto smem_i = j + threadIdx.x;
        smem_f16[smem_i]  = __float2half(smem_f32[smem_i]);
      }
      // Store FP16s
      if (reinterpret_cast<long>(ah) % 16 == 0 && ldah % 8 == 0)
      {
        __syncthreads();
        for (unsigned j = 0; j < smem_len; j += block_size * 8)
        {
          const auto smem_i = j + threadIdx.x * 8;
          if (smem_len < block_size * 8 && smem_i >= smem_len)
            break;
          const auto im = i + smem_i;
          cp<16>(&ah[im + ldah * in], &smem_f16[smem_i]);
        }
      }
      else if (reinterpret_cast<long>(ah) % 8 == 0 && ldah % 4 == 0)
      {
        __syncthreads();
        for (unsigned j = 0; j < smem_len; j += block_size * 4)
        {
          const auto smem_i = j + threadIdx.x * 4;
          if (smem_len < block_size * 4 && smem_i >= smem_len)
            break;
          const auto im = i + smem_i;
          cp<8>(&ah[im + ldah * in], &smem_f16[smem_i]);
        }
      }
      else if (reinterpret_cast<long>(ah) % 4 == 0 && ldah % 2 == 0)
      {
        __syncthreads();
        for (unsigned j = 0; j < smem_len; j += block_size * 2)
        {
          const auto smem_i = j + threadIdx.x * 2;
          if (smem_len < block_size * 2 && smem_i >= smem_len)
            break;
          const auto im = i + smem_i;
          cp<4>(&ah[im + ldah * in], &smem_f16[smem_i]);
        }
      }
      else
      {
        for (unsigned j = 0; j < smem_len; j += block_size)
        {
          const auto smem_i  = j + threadIdx.x;
          const auto im      = i + smem_i;
          ah[im + ldah * in] = smem_f16[smem_i];
        }
      }
    }
    else
    {
      // Load FP32 elements
      unsigned j = 0;
      for (; j < smem_len; j += block_size)
      {
        const auto smem_i = j + threadIdx.x;
        const auto im     = i + smem_i;
        if (im < m)
        {
          smem_f32[smem_i] = as[im + ldas * in];
        }
        else
        {
          break;
        }
      }
      const unsigned max_j = j;

      // Convert to FP16
      for (unsigned j = 0; j < max_j; j += block_size)
      {
        const auto smem_i = j + threadIdx.x;
        smem_f16[smem_i]  = __float2half(smem_f32[smem_i]);
      }
      // Store FP16s
      for (unsigned j = 0; j < max_j; j += block_size)
      {
        const auto smem_i  = j + threadIdx.x;
        const auto im      = i + smem_i;
        ah[im + ldah * in] = smem_f16[smem_i];
      }
    }
  }
}
