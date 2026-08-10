  } else if (segmentedRowPipeline) {
    const long totalSweeps = std::max<long>(0, ctx.n - 2);
    long maxSegmentRetries =
        static_cast<long>(envIntOrDefault("EVD_BC_SEGMENTED_MAX_RETRIES", 4096));
    if (maxSegmentRetries < 1) {
      maxSegmentRetries = 1;
    }
    if (ctx.rank == 0) {
      std::cout << "BC segmented row-pipeline total_sweeps=" << totalSweeps
                << " full_band_sync=" << (segmentedRowPipelineFullBcast ? 1 : 0)
                << std::endl;
    }

    for (int owner = 0; owner < ctx.size; ++owner) {
      const long rowStart = ctx.displs[owner];
      const long rowEnd = (owner + 1 < ctx.size)
                              ? (ctx.displs[owner] + ctx.counts[owner])
                              : ctx.n;
      if (ctx.rank == 0) {
        std::cout << "BC segmented row-pipeline owner " << owner
                  << " rows=[" << rowStart << "," << rowEnd << ")"
                  << " sweeps=[0," << totalSweeps << ")" << std::endl;
      }

      if (ctx.rank == owner && totalSweeps > 0) {
        EVD_CUDA_CHECK(cudaMemcpy(com, hCom.data(), sizeof(int) * hCom.size(), cudaMemcpyHostToDevice));
        int sweepStartInt = 0;
        int sweepEndInt = static_cast<int>(totalSweeps);
        int segmentStartInt = -2;
        int segmentEndInt = static_cast<int>(rowEnd);
        int segmentStopSweepInt = -1;
        void *kernelArgs[] = {
            (void *)&nInt,
            (void *)&bInt,
            (void *)&dBand,
            (void *)&ldBandInt,
            (void *)&dUPacked,
            (void *)&dPackedUOffsets,
            (void *)&packedUSweepCount,
            (void *)&sweepStartInt,
            (void *)&sweepEndInt,
            (void *)&segmentStartInt,
            (void *)&segmentEndInt,
            (void *)&segmentStopSweepInt,
            (void *)&blockNum,
            (void *)&com,
        };
        dim3 dimBlock(32, 32, 1);
        dim3 dimGrid(blockNum, 1, 1);
        long retry = 0;
        bool segmentReached = false;
        long long prevProgressScore = -1;
        do {
          int zeroBatch = 0;
          EVD_CUDA_CHECK(cudaMemcpyToSymbol(bcStopFlag, &zeroBatch, sizeof(int)));
          EVD_CUDA_CHECK(cudaLaunchCooperativeKernel((void *)chaseBulgesSegmentPackedKernel<kBcBandwidth>,
                                            dimGrid,
                                            dimBlock,
                                            kernelArgs));
          EVD_CUDA_CHECK(cudaGetLastError());
          EVD_CUDA_CHECK(cudaDeviceSynchronize());
          EVD_CUDA_CHECK(cudaMemcpy(hCom.data(), com, sizeof(int) * hCom.size(), cudaMemcpyDeviceToHost));
          long reached = 0;
          long long progressScore = 0;
          for (long sweep = 0; sweep < totalSweeps; ++sweep) {
            const long progress = hCom[static_cast<size_t>(sweep)];
            if (progress >= rowEnd || progress >= ctx.n + 3 * ctx.b) {
              reached++;
            }
            progressScore += std::min<long>(progress, rowEnd);
          }
          segmentReached = (reached == totalSweeps);
          if (segmentReached || progressScore == prevProgressScore) {
            break;
          }
          prevProgressScore = progressScore;
          retry++;
        } while (retry < maxSegmentRetries);
        if (!segmentReached) {
          std::cout << "BC segmented row-pipeline owner " << owner
                    << " stopped before row_end=" << rowEnd
                    << " retries=" << (retry + 1) << std::endl;
        }
      }

      comBroadcastInts += static_cast<unsigned long long>(hCom.size());
      MPI_CHECK(MPI_Bcast(hCom.data(),
                          static_cast<int>(hCom.size()),
                          MPI_INT,
                          owner,
                          ctx.comm));
      bandBroadcastDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                              static_cast<unsigned long long>(ctx.n);
      broadcastBandSuffix(ctx, dBand, 0, owner);
    }
