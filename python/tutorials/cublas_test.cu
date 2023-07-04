#include <iostream>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

int main() {
  int iteration = 10;

  float min_time = 0xffff;
  cublasGemmAlgo_t algo_index;
  for (const auto &algo : algoList) {
    float total_time = 0.0;
    for (int i = 0; i < iteration; i++) {

      cudaEvent_t start, end;
      cudaEventCreate(&start);
      cudaEventCreate(&end);

      cudaEventRecord(start, 0);
      cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a, CUDA_R_16F, k,
          m * k, d_b, CUDA_R_16F, n, k * n, &beta, d_c, CUDA_R_16F, n, m * n,
          batch_count, CUDA_R_16F, static_cast<cublasGemmAlgo_t>(algo));
      cudaEventRecord(end, 0);
      cudaEventSynchronize(end);
      float elapsed_time;
      cudaEventElapsedTime(&elapsed_time, start, end);
      total_time += elapsed_time;
    }
    float current_time = total_time / iteration;
    std::cout << "algo:" << algo << " " << current_time << " ms" << std::endl;
    if (current_time < min_time) {
      min_time = current_time;
      algo_index = algo;
    }
  }
  std::cout << "best:" << algo_index << " " << min_time << " ms" << std::endl;
}