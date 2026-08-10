struct CachedCudaBlock {
  void *ptr = nullptr;
  size_t bytes = 0;
  int device = -1;
  bool inUse = false;
};

bool cachedCudaAllocatorEnabled()
{
  return envIntOrDefault("EVD_CUDA_ALLOC_CACHE", 1) != 0;
}

std::vector<CachedCudaBlock> &cachedCudaBlocks()
{
  static std::vector<CachedCudaBlock> blocks;
  return blocks;
}

cudaError_t evdMallocRaw(void **ptr, size_t bytes)
{
  *ptr = nullptr;
  if (!cachedCudaAllocatorEnabled() || bytes == 0) {
    return cudaMalloc(ptr, bytes);
  }

  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  const size_t alignedBytes = (bytes + 255) & ~static_cast<size_t>(255);
  auto &blocks = cachedCudaBlocks();
  int best = -1;
  size_t bestBytes = std::numeric_limits<size_t>::max();
  const size_t maxWaste = std::max<size_t>(64ULL * 1024ULL * 1024ULL, alignedBytes / 4);
  for (int i = 0; i < static_cast<int>(blocks.size()); ++i) {
    const CachedCudaBlock &block = blocks[i];
    if (block.inUse || block.device != device || block.bytes < alignedBytes) {
      continue;
    }
    if (block.bytes - alignedBytes > maxWaste) {
      continue;
    }
    if (block.bytes < bestBytes) {
      best = i;
      bestBytes = block.bytes;
    }
  }
  if (best >= 0) {
    blocks[best].inUse = true;
    *ptr = blocks[best].ptr;
    return cudaSuccess;
  }

  void *raw = nullptr;
  status = cudaMalloc(&raw, alignedBytes);
  if (status != cudaSuccess) {
    return status;
  }
  blocks.push_back({raw, alignedBytes, device, true});
  *ptr = raw;
  return cudaSuccess;
}

cudaError_t evdFree(void *ptr)
{
  if (!cachedCudaAllocatorEnabled() || ptr == nullptr) {
    return cudaFree(ptr);
  }

  auto &blocks = cachedCudaBlocks();
  for (CachedCudaBlock &block : blocks) {
    if (block.ptr == ptr) {
      block.inUse = false;
      return cudaSuccess;
    }
  }
  return cudaFree(ptr);
}

template <typename T>
cudaError_t evdMalloc(T **ptr, size_t bytes)
{
  void *raw = nullptr;
  cudaError_t status = evdMallocRaw(&raw, bytes);
  if (status == cudaSuccess) {
    *ptr = static_cast<T *>(raw);
  }
  return status;
}
