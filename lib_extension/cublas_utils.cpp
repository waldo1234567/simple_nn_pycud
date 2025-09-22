#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h> 
#include <torch/extension.h>


inline void check_cublas(cublasStatus_t status, const char * msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error: ") + msg + " code=" + std::to_string((int)stat));
    }
}

void cublas_matmul_rowmajor(torch::Tensor A , torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32 && C.dtype() == torch::kFloat32,
                "Only float32 supported here");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(), "Tensors must be contiguous");
    // C = A * B
    // A: m x k
    // B: k x n
    // C: m x n
    int64_t m = A.size(0);
    int64_t k = A.size(1);
    int64_t n = B.size(1);

    TORCH_CHECK( B.size(0) == k, "Incompatible matrix dimensions");
    TORCH_CHECK( C.size(0) == m && C.size(1) == n, "Output matrix has incorrect dimensions");

    const float * A_ptr = A.data_ptr<float>();
    const float * B_ptr = B.data_ptr<float>();
    float * C_ptr = C.data_ptr<float>();

    static thread_local cublasHandle_t handle = nullptr;
    if(handle == nullptr) {
        check_cublas(cublasCreate(&handle), "cublasCreate");
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream");

    float alpha = 1.0f;
    float beta = 0.0f;
    // Note: cuBLAS uses column-major order, so we swap A and B and transpose the operation
    check_cublas(cublasSgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_T,
                             (int)n, (int)m, (int)k,
                             &alpha,
                             B_ptr, (int)k,
                             A_ptr, (int)k,
                             &beta,
                             C_ptr, (int)n),
                 "cublasSgemm");
}