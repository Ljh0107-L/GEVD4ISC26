void *checkedMallocHost(size_t bytes)
{
  void *ptr = nullptr;
  EVD_CUDA_CHECK(cudaMallocHost(&ptr, bytes));
  return ptr;
}

void checkedFreeHost(void *ptr)
{
  if (ptr != nullptr) {
    EVD_CUDA_CHECK(cudaFreeHost(ptr));
  }
}

void loadLocalColumnsFromBinary(const char *fileName,
                                double *dA,
                                long n,
                                long colStart,
                                long localCols,
                                int rank)
{
  const size_t expectedBytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(double);
  std::ifstream file(fileName, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "rank " << rank << " failed to open input matrix: " << fileName << std::endl;
    ::gevd4isc26::evd::abortActiveCommunicator(1);
  }
  const std::streamoff fileBytes = file.tellg();
  if (fileBytes != static_cast<std::streamoff>(expectedBytes)) {
    std::cerr << "rank " << rank << " input matrix size mismatch: expected "
              << expectedBytes << " bytes, got " << fileBytes << std::endl;
    ::gevd4isc26::evd::abortActiveCommunicator(1);
  }

  const size_t chunkDoubles = 16 * 1024 * 1024;
  double *host = static_cast<double *>(checkedMallocHost(chunkDoubles * sizeof(double)));
  const size_t totalDoubles = static_cast<size_t>(n) * static_cast<size_t>(localCols);
  size_t done = 0;
  const std::streamoff byteOffset =
      static_cast<std::streamoff>(colStart) * static_cast<std::streamoff>(n) *
      static_cast<std::streamoff>(sizeof(double));
  file.seekg(byteOffset, std::ios::beg);
  while (done < totalDoubles) {
    const size_t count = std::min(chunkDoubles, totalDoubles - done);
    const size_t bytes = count * sizeof(double);
    file.read(reinterpret_cast<char *>(host), static_cast<std::streamsize>(bytes));
    if (file.gcount() != static_cast<std::streamsize>(bytes)) {
      std::cerr << "rank " << rank << " short read from input matrix" << std::endl;
      ::gevd4isc26::evd::abortActiveCommunicator(1);
    }
    EVD_CUDA_CHECK(cudaMemcpy(dA + done, host, bytes, cudaMemcpyHostToDevice));
    done += count;
  }
  checkedFreeHost(host);
}

__global__ void symmetrizeLocalDiagonalBlock(double *A,
                                             long ldA,
                                             long colStart,
                                             long localCols)
{
  long rowLocal = blockIdx.x * blockDim.x + threadIdx.x;
  long colLocal = blockIdx.y * blockDim.y + threadIdx.y;
  if (rowLocal >= localCols || colLocal >= localCols) {
    return;
  }
  long globalRow = colStart + rowLocal;
  long globalCol = colStart + colLocal;
  if (globalRow < globalCol) {
    A[globalRow + colLocal * ldA] = A[globalCol + rowLocal * ldA];
  }
}

__global__ void unpackTransposedPeerBlock(const double *src,
                                          long srcRows,
                                          double *dst,
                                          long ldDst,
                                          long dstRowStart,
                                          long dstCols)
{
  long srcCol = blockIdx.x * blockDim.x + threadIdx.x;
  long srcRow = blockIdx.y * blockDim.y + threadIdx.y;
  if (srcRow < srcRows && srcCol < dstCols) {
    dst[dstRowStart + srcRow + srcCol * ldDst] = src[srcCol + srcRow * dstCols];
  }
}

void symmetrizeLowerToUpperDistributed(double *dA,
                                       long n,
                                       long localCols,
                                       const std::vector<long> &counts,
                                       const std::vector<long> &displs,
                                       int rank,
                                       MPI_Comm comm)
{
  dim3 block(16, 16);
  dim3 grid(ceilDiv(localCols, block.x), ceilDiv(localCols, block.y));
  symmetrizeLocalDiagonalBlock<<<grid, block>>>(dA, n, displs[rank], localCols);
  EVD_CUDA_CHECK(cudaGetLastError());
  EVD_CUDA_CHECK(cudaDeviceSynchronize());

  long maxRows = 0;
  long maxCols = 0;
  for (long c : counts) {
    maxRows = std::max(maxRows, c);
    maxCols = std::max(maxCols, c);
  }
  const size_t maxElems = static_cast<size_t>(maxRows) * static_cast<size_t>(maxCols);
  double *hostPeer = static_cast<double *>(checkedMallocHost(maxElems * sizeof(double)));
  double *dPeer = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dPeer, maxElems * sizeof(double)));

  for (int src = 0; src < static_cast<int>(counts.size()); ++src) {
    for (int dst = src + 1; dst < static_cast<int>(counts.size()); ++dst) {
      const long srcCols = counts[src];
      const long dstCols = counts[dst];
      const size_t elems = static_cast<size_t>(srcCols) * static_cast<size_t>(dstCols);
      const int tag = 17000 + src * 64 + dst;
      if (rank == src) {
        EVD_CUDA_CHECK(cudaMemcpy2D(hostPeer,
                           dstCols * sizeof(double),
                           dA + displs[dst],
                           n * sizeof(double),
                           dstCols * sizeof(double),
                           srcCols,
                           cudaMemcpyDeviceToHost));
        MPI_CHECK(MPI_Send(hostPeer, static_cast<int>(elems), MPI_DOUBLE, dst, tag, comm));
      } else if (rank == dst) {
        MPI_CHECK(MPI_Recv(hostPeer,
                           static_cast<int>(elems),
                           MPI_DOUBLE,
                           src,
                           tag,
                           comm,
                           MPI_STATUS_IGNORE));
        EVD_CUDA_CHECK(cudaMemcpy(dPeer, hostPeer, elems * sizeof(double), cudaMemcpyHostToDevice));
        dim3 tBlock(16, 16);
        dim3 tGrid(ceilDiv(localCols, tBlock.x), ceilDiv(counts[src], tBlock.y));
        unpackTransposedPeerBlock<<<tGrid, tBlock>>>(dPeer, counts[src], dA, n, displs[src], localCols);
        EVD_CUDA_CHECK(cudaGetLastError());
        EVD_CUDA_CHECK(cudaDeviceSynchronize());
      }
      MPI_CHECK(MPI_Barrier(comm));
    }
  }

  EVD_CUDA_CHECK(evdFree(dPeer));
  checkedFreeHost(hostPeer);
}

__global__ void gatherPanelRowsBT(const double *panel,
                                  long ldPanel,
                                  long panelRowStart,
                                  long activeGlobalStart,
                                  long activeCols,
                                  double *outBT,
                                  long ldOut)
{
  long r = blockIdx.x * blockDim.x + threadIdx.x;
  long c = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < ldOut && c < activeCols) {
    long globalRow = activeGlobalStart + c;
    outBT[r + c * ldOut] = panel[(globalRow - panelRowStart) + r * ldPanel];
  }
}

__global__ void packBandColumns(const double *A,
                                long ldA,
                                long n,
                                long b,
                                long colStart,
                                long localCols,
                                double *band,
                                long ldBand)
{
  long r = blockIdx.x * blockDim.x + threadIdx.x;
  long c = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < ldBand && c < localCols) {
    long globalCol = colStart + c;
    long globalRow = globalCol + r;
    band[r + c * ldBand] = (globalRow < n) ? A[globalRow + c * ldA] : 0.0;
  }
}

__global__ void setLocalIdentity(double *Q, long ldQ, long n, long colStart, long localCols)
{
  long row = blockIdx.x * blockDim.x + threadIdx.x;
  long col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < n && col < localCols) {
    Q[row + col * ldQ] = (row == colStart + col) ? 1.0 : 0.0;
  }
}
