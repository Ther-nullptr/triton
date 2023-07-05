#include <stdio.h>
#include <sys/time.h>

#include <string>
#include <map>

#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
 
template <typename T, typename S>
void allocate_memory(int b, int m, int n, int k, T **A, T **B, S **C) {
    cudaMallocManaged(A, b * m * k * sizeof(T));
    cudaMallocManaged(B, b * k * n * sizeof(T));
    cudaMallocManaged(C, b * m * n * sizeof(S));
}
 
template <typename T, typename S>
void free_memory(T *A, T *B, S *C) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
 
template <typename T, typename S>
inline int cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
                   int b, int m, int n, int k, T *A, T *B, S *C, int lda, int ldb, int ldc,
                   S *alpha, S *beta, int algo) {
    cudaDataType_t AType, BType, CType, ComputeType;
    AType = BType = CType = ComputeType = CUDA_R_16F;

    cublasStatus_t status;
    status = cublasGemmStridedBatchedEx(handle,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          alpha,
                          A,
                          AType,
                          lda,
                          m * k,
                          B,
                          BType,
                          ldb,
                          k * n,
                          beta,
                          C,
                          CType,
                          ldc,
                          m * n,
                          b,
                          ComputeType,
                          static_cast<cublasGemmAlgo_t>(algo));
    
    if (status == CUBLAS_STATUS_SUCCESS)
        return 1;
    else
        return -1;
}
 
template <typename T, typename S>
inline float test_gemm(cublasHandle_t handle, int b, int m, int n, int k, T *A, T *B, S *C,
               S *alpha, S *beta, int algo, int iteration) {
    float total_time = 0;
    for (int i = 0; i < iteration; ++i) {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, 0);
        int success = cublas_gemm_ex(handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     b, 
                                     n,
                                     m,
                                     k,
                                     B,
                                     A,
                                     C,
                                     n,
                                     k,
                                     n,
                                     alpha,
                                     beta,
                                     static_cast<cublasGemmAlgo_t>(algo));
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, end);
        if (success > 0 && i > 0)
        {
            total_time += elapsed_time;
        }
    }
    if (total_time > 0)
    {
        printf("algo %d: %.3f ms\n", algo, total_time / (iteration - 1));
    }
    return total_time / (iteration - 1);
}

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
    
    for (const auto &dict : {train_dict, inference_w_o_KV_dict, inference_w_KV_dict})
    {
        printf("=========================\n");
        for (const auto &item : dict)
        {
            printf("-------------------------\n");
            printf("%s: [%d, %d, %d, %d]\n", item.first.c_str(), item.second.b, item.second.m, item.second.n, item.second.k);

            Shape shape = item.second;
            int b = shape.b;
            int m = shape.m;
            int n = shape.n;
            int k = shape.k;

            int start_algo = CUBLAS_GEMM_DEFAULT;
            int end_algo = CUBLAS_GEMM_ALGO23;
            int start_algo_t_op = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            int end_algo_t_op = CUBLAS_GEMM_ALGO15_TENSOR_OP;
            int iteration = 100;
        
            half *hA, *hB, *hC;
            half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
            allocate_memory(b, m, n, k, &hA, &hB, &hC);
            for (int i = 0; i < m * k; ++i) {
                hA[i] = __float2half_rn(float(i % 255 - 127) / 127);
            } 
            for (int i = 0; i < k * n; ++i) {
                hB[i] = __float2half_rn(float(i % 255 - 127) / 127);
            }
            

            // warm up
            printf(">>>>>>>>>>>>>>>>> warm up >>>>>>>>>>>>>>>>>\n");
            for (int algo = start_algo; algo <= end_algo; ++algo)
                test_gemm(handle, b, m, n, k, hA, hB, hC, &h_alpha, &h_beta, algo, 1);
            
            printf(">>>>>>>>>>>>>>>>> test fp16 >>>>>>>>>>>>>>>>>\n");
            float min_time = 0xffff;
            cublasGemmAlgo_t algo_index;
            for (int algo = start_algo; algo <= end_algo; ++algo)
            {
                float current_time = test_gemm(handle, b, m, n, k, hA, hB, hC, &h_alpha, &h_beta, algo, iteration);
                if (current_time < min_time)
                {
                    min_time = current_time;
                    algo_index = static_cast<cublasGemmAlgo_t>(algo);
                }
            }
            for (int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo)
            {
                float current_time = test_gemm(handle, b, m, n, k, hA, hB, hC, &h_alpha, &h_beta, algo, iteration);
                if (current_time < min_time)
                {
                    min_time = current_time;
                    algo_index = static_cast<cublasGemmAlgo_t>(algo);
                }
            }
            printf("[%s] min_time: %.3f ms, best algorithm: %d\n", item.first.c_str(), min_time, static_cast<int>(algo_index));
            free_memory(hA, hB, hC);
        }
    }
    
    return 0;
}