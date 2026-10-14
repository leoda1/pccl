/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "hggc_runtime.h"
#include "pccl.h"
#include "pccl_utils.h"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/*
 * PCCL AllGather example (single process, all GPUs).
 *
 * Usage: ./allgather [bytes_per_rank] [iters]
 *   bytes_per_rank  default 65536. Must be a multiple of 16 for the ext-kernel
 *                   one-shot AllGather (pcclKernel_oneShotAllGatherKernel).
 *   iters           default 20 timed iterations after 5 warm-ups.
 *
 * The ext-kernel path is taken only if the plugin is loaded, the topology is
 * accepted, and total bytes (bytes_per_rank * nranks) <= agBoundHigh
 * (2MB for SIXTEEN_RANKS_810E). Larger sizes fall back to ring/symmetric kernels.
 * Do not set PCCL_ALGO / PCCL_PROTO, they disable the ext kernel.
 */

int main(int argc, char *argv[]) {
  size_t bytesPerRank = 65536;
  int iters = 20;
  if (argc > 1) bytesPerRank = strtoull(argv[1], NULL, 0);
  if (argc > 2) iters = atoi(argv[2]);
  const int warmup = 5;

  int nGpus = 0;
  HGGCCHECK(hggcGetDeviceCount(&nGpus));
  if (nGpus < 2) {
    printf("Need at least 2 HGGC devices, found %d\n", nGpus);
    return EXIT_FAILURE;
  }
  printf("AllGather: %d devices, %zu bytes per rank, %zu bytes total\n", nGpus,
         bytesPerRank, bytesPerRank * nGpus);

  std::vector<pcclComm_t> comms(nGpus);
  std::vector<hggcStream_t> streams(nGpus);
  std::vector<int> devices(nGpus);
  std::vector<char *> sendbuff(nGpus), recvbuff(nGpus);
  for (int i = 0; i < nGpus; i++) devices[i] = i;

  PCCLCHECK(pcclCommInitAll(comms.data(), nGpus, devices.data()));

  // Rank i contributes bytesPerRank bytes, all set to pattern(i).
  auto pattern = [](int r) { return (unsigned char)((r * 7 + 3) & 0xff); };
  for (int i = 0; i < nGpus; i++) {
    HGGCCHECK(hggcSetDevice(i));
    HGGCCHECK(hggcStreamCreate(&streams[i]));
    HGGCCHECK(hggcMalloc((void **)&sendbuff[i], bytesPerRank));
    HGGCCHECK(hggcMalloc((void **)&recvbuff[i], bytesPerRank * nGpus));
    HGGCCHECK(hggcMemset(sendbuff[i], pattern(i), bytesPerRank));
    HGGCCHECK(hggcMemset(recvbuff[i], 0, bytesPerRank * nGpus));
  }

  auto runOnce = [&]() {
    PCCLCHECK(pcclGroupStart());
    for (int i = 0; i < nGpus; i++) {
      PCCLCHECK(pcclAllGather(sendbuff[i], recvbuff[i], bytesPerRank, pcclInt8,
                              comms[i], streams[i]));
    }
    PCCLCHECK(pcclGroupEnd());
  };
  auto syncAll = [&]() {
    for (int i = 0; i < nGpus; i++) {
      HGGCCHECK(hggcSetDevice(i));
      HGGCCHECK(hggcStreamSynchronize(streams[i]));
    }
  };

  // Correctness: run once and verify before timing.
  runOnce();
  syncAll();

  bool success = true;
  std::vector<unsigned char> host(bytesPerRank * nGpus);
  for (int i = 0; i < nGpus; i++) {
    HGGCCHECK(hggcSetDevice(i));
    HGGCCHECK(hggcMemcpy(host.data(), recvbuff[i], host.size(), hggcMemcpyDeviceToHost));
    for (int r = 0; r < nGpus && success; r++) {
      for (size_t b = 0; b < bytesPerRank; b++) {
        if (host[r * bytesPerRank + b] != pattern(r)) {
          printf("  Device %d: wrong byte at rank-slot %d offset %zu: got 0x%02x expected 0x%02x\n",
                 i, r, b, host[r * bytesPerRank + b], pattern(r));
          success = false;
          break;
        }
      }
    }
  }
  printf("Verification: %s\n", success ? "PASS" : "FAIL");

  // Timing (wall clock, includes launch; for kernel time use a profiler).
  if (success && iters > 0) {
    for (int k = 0; k < warmup; k++) runOnce();
    syncAll();
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < iters; k++) runOnce();
    syncAll();
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    double busBytes = (double)bytesPerRank * (nGpus - 1);
    printf("avg %.2f us/iter, busbw %.2f GB/s\n", us, busBytes / (us * 1e3));
  }

  for (int i = 0; i < nGpus; i++) {
    PCCLCHECK(pcclCommFinalize(comms[i]));
    PCCLCHECK(pcclCommDestroy(comms[i]));
  }
  for (int i = 0; i < nGpus; i++) {
    HGGCCHECK(hggcSetDevice(i));
    HGGCCHECK(hggcFree(sendbuff[i]));
    HGGCCHECK(hggcFree(recvbuff[i]));
    HGGCCHECK(hggcStreamDestroy(streams[i]));
  }
  return success ? 0 : EXIT_FAILURE;
}
