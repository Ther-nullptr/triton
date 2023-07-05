#include <iostream>
#include <string>
#include <map>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static std::initializer_list<cublasGemmAlgo_t> algoList = {
    CUBLAS_GEMM_DEFAULT,
    CUBLAS_GEMM_ALGO0,
    CUBLAS_GEMM_ALGO1,
    CUBLAS_GEMM_ALGO2,
    CUBLAS_GEMM_ALGO3,
    CUBLAS_GEMM_ALGO4,
    CUBLAS_GEMM_ALGO5,
    CUBLAS_GEMM_ALGO6,
    CUBLAS_GEMM_ALGO7,
    CUBLAS_GEMM_ALGO8,
    CUBLAS_GEMM_ALGO9,
    CUBLAS_GEMM_ALGO10,
    CUBLAS_GEMM_ALGO11,
    CUBLAS_GEMM_ALGO12,
    CUBLAS_GEMM_ALGO13,
    CUBLAS_GEMM_ALGO14,
    CUBLAS_GEMM_ALGO15,
    CUBLAS_GEMM_ALGO16,
    CUBLAS_GEMM_ALGO17,
    CUBLAS_GEMM_ALGO18,  // sliced 32x32
    CUBLAS_GEMM_ALGO19,  // sliced 64x32
    CUBLAS_GEMM_ALGO20,  // sliced 128x32
    CUBLAS_GEMM_ALGO21,  // sliced 32x32  -splitK
    CUBLAS_GEMM_ALGO22,  // sliced 64x32  -splitK
    CUBLAS_GEMM_ALGO23,  // sliced 128x32 -splitK
    CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    CUBLAS_GEMM_ALGO0_TENSOR_OP,
    CUBLAS_GEMM_ALGO1_TENSOR_OP,
    CUBLAS_GEMM_ALGO2_TENSOR_OP,
    CUBLAS_GEMM_ALGO3_TENSOR_OP,
    CUBLAS_GEMM_ALGO4_TENSOR_OP,
    CUBLAS_GEMM_ALGO18,
    CUBLAS_GEMM_ALGO19,
    CUBLAS_GEMM_ALGO20,
    CUBLAS_GEMM_ALGO21,
    CUBLAS_GEMM_ALGO22,
    CUBLAS_GEMM_ALGO23,
    CUBLAS_GEMM_ALGO5_TENSOR_OP,
    CUBLAS_GEMM_ALGO6_TENSOR_OP,
    CUBLAS_GEMM_ALGO7_TENSOR_OP,
    CUBLAS_GEMM_ALGO8_TENSOR_OP,
    CUBLAS_GEMM_ALGO9_TENSOR_OP,
    CUBLAS_GEMM_ALGO10_TENSOR_OP,
    CUBLAS_GEMM_ALGO11_TENSOR_OP,
    CUBLAS_GEMM_ALGO12_TENSOR_OP,
    CUBLAS_GEMM_ALGO13_TENSOR_OP,
    CUBLAS_GEMM_ALGO14_TENSOR_OP,
    CUBLAS_GEMM_ALGO15_TENSOR_OP
};

struct Shape
{
    int b, m, n, k;
};

int main()
{
    // initialize
    cublasHandle_t handle;
    cublasCreate(&handle);

    // test list of shape
    std::map<std::string, Shape> train_dict = {
        {"XxQKVw", {1, 8192, 4608, 12288}},
        {"QxK^T", {192, 512, 512, 128}},
        {"QK^TxV", {192, 512, 128, 512}},
        {"Proj", {1, 8192, 12288, 1536}},
        {"FC1", {1, 8192, 6144, 12288}},
        {"FC2", {1, 8192, 12288, 6144}},
        {"QxK^TFlat", {1, 512, 512, 24576}},
        {"QK^TxVFlat", {1, 512, 24576, 512}}
    };

    std::map<std::string, Shape> inference_w_o_KV_dict = {
        {"XxQKVw", {1, 8704, 4608, 12288}},
        {"QxK^T", {192, 543, 543, 128}},
        {"QK^TxV", {192, 543, 128, 543}},
        {"Proj", {1, 8688, 12288, 1536}},
        {"FC1", {1, 8688, 6144, 12288}},
        {"FC2", {1, 8688, 12288, 6144}},
        {"QxK^TFlat", {1, 543, 543, 24576}},
        {"QK^TxVFlat", {1, 543, 24576, 543}}
    };

    std::map<std::string, Shape> inference_w_KV_dict = {
        {"XxQKVw", {1, 16, 4608, 12288}},
        {"QxK^T", {192, 1, 543, 128}},
        {"QK^TxV", {192, 1, 128, 543}},
        {"Proj", {1, 16, 12288, 1536}},
        {"FC1", {1, 16, 6144, 12288}},
        {"FC2", {1, 16, 12288, 6144}},
        {"QxK^TFlat", {1, 1, 543, 24576}},
        {"QK^TxVFlat", {1, 1, 24576, 543}}
    };

    // iterate the three dicts
    for (const auto &dict : {train_dict, inference_w_o_KV_dict, inference_w_KV_dict})
    {
        std::cout << "=========================" << std::endl;
        for (const auto &item : dict)
        {
            std::cout << "-------------------------" << std::endl;
            std::cout << item.first << ":[" << item.second.b << "," << item.second.m << ","
                      << item.second.n << "," << item.second.k << "]" << std::endl;
            Shape shape = item.second;
            int b = shape.b;
            int m = shape.m;
            int n = shape.n;
            int k = shape.k;

            // allocate memory
            half *d_a, *d_b, *d_c;
            cudaMallocManaged((void**)&d_a, sizeof(half) * m * k * b);
            cudaMallocManaged((void**)&d_b, sizeof(half) * k * n * b);
            cudaMallocManaged((void**)&d_c, sizeof(half) * m * n * b);

            // randomly initialize data


            // initialize data
            half alpha = __float2half(1.0);
            half beta = __float2half(0.0);

            // test
            int iteration = 1;
            float min_time = 0xffff;
            cublasGemmAlgo_t algo_index;
            for (const auto &algo : algoList)
            {
                float total_time = 0.0;
                for (int i = 0; i < iteration; i++)
                {

                    cudaEvent_t start, end;
                    cudaEventCreate(&start);
                    cudaEventCreate(&end);

                    cudaEventRecord(start, 0);
                    cublasGemmStridedBatchedEx(
                        handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, d_a, CUDA_R_16F, k,
                        m * k, d_b, CUDA_R_16F, n, k * n, &beta, d_c, CUDA_R_16F, n, m * n,
                        b, CUDA_R_16F, static_cast<cublasGemmAlgo_t>(algo));
                    cudaEventRecord(end, 0);
                    cudaEventSynchronize(end);
                    float elapsed_time;
                    cudaEventElapsedTime(&elapsed_time, start, end);
                    total_time += elapsed_time;
                }
                float current_time = total_time / iteration;
                std::cout << "algo:" << algo << " " << current_time << " ms" << std::endl;
                if (current_time < min_time)
                {
                    min_time = current_time;
                    algo_index = algo;
                }
            }
            std::cout << "best:" << algo_index << " " << min_time << " ms" << std::endl;

            // free memory
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
        }
    }
}