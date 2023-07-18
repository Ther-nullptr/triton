#include "deviceQuery.h"
#include <cuda.h>
#include <iostream>
#include <tuple>

constexpr int WarpsStart = 1;
constexpr int WarpsEnd = 32;
constexpr int ThreadsStart = 1;
constexpr int ThreadsEnd = 32;

constexpr int threadsPerWarp = 32;
constexpr int sharedMemSize = (32 * 1024 / 8);
constexpr int iters = 1000;

struct Param_Struct {
  Param_Struct() {
    startClk = (uint32_t *)malloc(WarpsEnd * sizeof(uint32_t));
    stopClk = (uint32_t *)malloc(WarpsEnd * sizeof(uint32_t));
    dsink = (uint64_t *)malloc(WarpsEnd * sizeof(uint64_t));

    gpuErrchk(cudaMalloc(&startClk_g, WarpsEnd * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&stopClk_g, WarpsEnd * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&dsink_g, WarpsEnd * sizeof(uint64_t)));
  }

  ~Param_Struct() {
    free(startClk);
    free(stopClk);
    free(dsink);

    gpuErrchk(cudaFree(startClk_g));
    gpuErrchk(cudaFree(stopClk_g));
    gpuErrchk(cudaFree(dsink_g));
  }

  uint32_t *startClk;
  uint32_t *stopClk;
  uint64_t *dsink;

  uint32_t *startClk_g;
  uint32_t *stopClk_g;
  uint64_t *dsink_g;
};

__global__ void shared_lat(uint32_t *startClk, uint32_t *stopClk,
                           uint64_t *dsink, uint32_t stride,
                           uint32_t active_thread_per_warp) {

  // thread index
  uint32_t tid = threadIdx.x;
  uint32_t wid = tid / threadsPerWarp;

  __shared__ uint64_t s[sharedMemSize]; // static shared memory

  // initialize the pointer-chasing array with 1 thread
  if (tid == 0) {
    for (uint32_t i = 0; i < (sharedMemSize - stride); i += 1)
      s[i] = (i + stride) % sharedMemSize;
    for (uint32_t i = 0; i < stride; i += 1)
      s[sharedMemSize - stride + i] = i;
  }
  __syncthreads();

  // use the first active_thread_per_warp threads of each warp to initialize the
  // pointer-chasing
  if (tid % threadsPerWarp < active_thread_per_warp) {
    // initalize pointer chaser
    uint64_t p_chaser = tid % sharedMemSize; // if stride = 1, shared memory
                                             // accesses are always coalesced

    // start timing
    uint32_t start = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

    // pointer-chasing itersS times
    for (uint32_t i = 0; i < iters; ++i) {
      p_chaser = s[p_chaser];
    }

    // stop timing
    uint32_t stop = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

    // write time and data back to memory
    if (tid % threadsPerWarp == 0) {
      startClk[wid] = start;
      stopClk[wid] = stop;
      dsink[wid] = p_chaser;
    }
  }
}

template <int warps_num, int active_thread_per_warp> struct SharedLatency {
  static void callKernel(const Param_Struct &param) {
    std::cout << "shared_lat<" << warps_num << ", " << active_thread_per_warp
              << ">\n";
    // 在这里添加对应的 CUDA kernel 调用，可以使用 param
    dim3 grid(1);
    dim3 block(threadsPerWarp, warps_num);

    shared_lat<<<grid, block>>>(param.startClk_g, param.stopClk_g,
                                param.dsink_g, 1, active_thread_per_warp);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(param.startClk, param.startClk_g,
                         warps_num * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(param.stopClk, param.stopClk_g,
                         warps_num * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(param.dsink, param.dsink_g,
                         warps_num * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    float duration[WarpsEnd];
    float max_duration = 0;
    for (int i = 0; i < warps_num; ++i) {
      duration[i] = (float)(param.stopClk[i] - param.startClk[i]);
      max_duration = max_duration > duration[i] ? max_duration : duration[i];
    }

    float lat = max_duration / iters;
    printf("#WARP = %d, #ACTIVE_THREAD_PER_WARP = %d\n", warps_num,
           active_thread_per_warp);
    printf("Latency  = %f cycles, ", lat);
    printf("Bandwidth = %f B/cycle\n", warps_num * sizeof(uint64_t) * iters *
                                           active_thread_per_warp /
                                           max_duration);
  }
};

template <typename... Functions, std::size_t... Is>
void callAllFunctionsImpl(const std::tuple<Functions...> &functionList,
                          const Param_Struct &param,
                          std::index_sequence<Is...>) {
  ((std::get<Is>(functionList))(param), ...);
}

template <typename... Functions>
void callAllFunctions(const std::tuple<Functions...> &functionList,
                      const Param_Struct &param) {
  callAllFunctionsImpl(functionList, param,
                       std::index_sequence_for<Functions...>{});
}

template <int WarpsStart, int WarpsEnd, int ThreadsStart, int ThreadsEnd>
struct GenerateSharedLat {
  static auto generate() {
    if constexpr (ThreadsStart <= ThreadsEnd) {
      return std::tuple_cat(
          std::tuple<void (*)(const Param_Struct &)>{
              &SharedLatency<WarpsStart, ThreadsStart>::callKernel},
          GenerateSharedLat<WarpsStart, WarpsEnd, ThreadsStart * 2,
                            ThreadsEnd>::generate());
    } else if constexpr (WarpsStart < WarpsEnd) {
      return std::tuple_cat(
          std::tuple<void (*)(const Param_Struct &)>{
              &SharedLatency<WarpsStart, ThreadsStart>::callKernel},
          GenerateSharedLat<WarpsStart * 2, WarpsEnd, 1,
                            ThreadsEnd>::generate());
    } else {
      return std::tuple<>();
    }
  }
};

int main() {
  Param_Struct param;

  auto functionList = GenerateSharedLat<WarpsStart, WarpsEnd, ThreadsStart,
                                        ThreadsEnd>::generate();
  callAllFunctions(functionList, param);

  return 0;
}
