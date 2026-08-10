// Included after the SBR runtime helpers so cleanup can also release optional
// suffix communicators. The class gives every early return the same resource
// lifecycle as the successful path.
class EvdRuntime {
public:
  explicit EvdRuntime(DistContext& context) : context_(context) {
    [[maybe_unused]]
    gevd4isc26::detail::ScopedCpuAffinityRestore preserve_cpu_affinity;
    ncclUniqueId id;
    if (context_.rank == 0) {
      NCCL_CHECK_LOCAL(ncclGetUniqueId(&id));
    }
    MPI_CHECK(MPI_Bcast(&id, static_cast<int>(sizeof(id)), MPI_BYTE, 0,
                        context_.comm));
    EVD_CUDA_CHECK(cudaStreamCreate(&context_.commStream));
    NCCL_CHECK_LOCAL(ncclCommInitRank(
        &context_.nccl, context_.size, id, context_.rank));
    CUSOLVER_CHECK_LOCAL(cusolverDnCreate(&cusolver_));
    CUBLAS_CHECK_LOCAL(cublasCreate(&cublas_));
  }

  ~EvdRuntime() {
    release();
  }

  EvdRuntime(const EvdRuntime&) = delete;
  EvdRuntime& operator=(const EvdRuntime&) = delete;

  [[nodiscard]] cusolverDnHandle_t cusolver() const noexcept {
    return cusolver_;
  }

  [[nodiscard]] cublasHandle_t cublas() const noexcept {
    return cublas_;
  }

  void release() noexcept {
    if (cublas_ != nullptr) {
      cublasDestroy(cublas_);
      cublas_ = nullptr;
    }
    if (cusolver_ != nullptr) {
      cusolverDnDestroy(cusolver_);
      cusolver_ = nullptr;
    }
    destroySbrSuffixNccls(&context_);
    if (context_.nccl != nullptr) {
      ncclCommDestroy(context_.nccl);
      context_.nccl = nullptr;
    }
    if (context_.commStream != nullptr) {
      cudaStreamDestroy(context_.commStream);
      context_.commStream = nullptr;
    }
  }

private:
  DistContext& context_;
  cusolverDnHandle_t cusolver_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
};
