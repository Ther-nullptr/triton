#ifndef MAXFLOPS_half_DEF_H
#define MAXFLOPS_half_DEF_H

#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include "deviceQuery.h"

#define REPEAT_TIMES 4096

template <class T>
__global__ void max_flops(uint32_t *startClk, uint32_t *stopClk, T *data1,
                          T *data2, T *res) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s1 = data1[gid];
  register T s2 = data2[gid];
  register T result = 0;

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  uint32_t start = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

  for (int j = 0; j < REPEAT_TIMES; ++j) {
    // asm volatile("{\t\n"
    //              "fma.rn.f16 %0, %1, %2 , %0;\n\t"
    //              "fma.rn.f16 %0, %1, %2 , %0;\n\t"
    //              "fma.rn.f16 %0, %1, %2 , %0;\n\t"
    //              "fma.rn.f16 %0, %1, %2 , %0;\n\t"
    //              "}"
    //              : "+f"(result), "f"(s1), "f"(s2));
    result = s1 * s2 + result;
    result = s1 * s2 + result;
    result = s1 * s2 + result;
    result = s1 * s2 + result;
  }

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // stop timing
  uint32_t stop = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

  // write time and data back to memory
  startClk[gid] = start;
  stopClk[gid] = stop;
  res[gid] = result;
}

float fp16_max_flops() {
  BLOCKS_NUM = 1;
  int ratio = 1;
  THREADS_PER_BLOCK = THREADS_PER_BLOCK / ratio;
  TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

  uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *stopClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  half *data1 = (half *)malloc(TOTAL_THREADS * sizeof(half));
  half *data2 = (half *)malloc(TOTAL_THREADS * sizeof(half));
  half *res = (half *)malloc(TOTAL_THREADS * sizeof(half));

  uint32_t *startClk_g;
  uint32_t *stopClk_g;
  half *data1_g;
  half *data2_g;
  half *res_g;

  for (uint32_t i = 0; i < TOTAL_THREADS; i++) {
    data1[i] = (half)i;
    data2[i] = (half)i;
  }

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&data1_g, TOTAL_THREADS * sizeof(half)));
  gpuErrchk(cudaMalloc(&data2_g, TOTAL_THREADS * sizeof(half)));
  gpuErrchk(cudaMalloc(&res_g, TOTAL_THREADS * sizeof(half)));

  gpuErrchk(cudaMemcpy(data1_g, data1, TOTAL_THREADS * sizeof(half),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(data2_g, data2, TOTAL_THREADS * sizeof(half),
                       cudaMemcpyHostToDevice));

  max_flops<half><<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(startClk_g, stopClk_g,
                                                       data1_g, data2_g, res_g);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(res, res_g, TOTAL_THREADS * sizeof(half),
                       cudaMemcpyDeviceToHost));

  float flops;
  flops = (float)(REPEAT_TIMES * TOTAL_THREADS * 8) /
          ((float)(stopClk[0] - startClk[0]));
  printf("fp16 FLOP per SM = %f (flop/clk/SM)\n", flops);
  printf("Total Clk number = %u \n", stopClk[0] - startClk[0]);

  return flops;
}

#endif
