#include <torch/extension.h>
#include <tuple>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

void relu_mask_cuda(torch::Tensor dZ, torch::Tensor grad_out, torch::Tensor out);
void cublas_matmul_rowmajor(torch::Tensor A , torch::Tensor B, torch::Tensor C);
void add_bias_relu_cuda(torch::Tensor out, torch::Tensor bias);
void gelu_scale_forward_launcher(const torch::Tensor& inp, const torch::Tensor& scale, torch::Tensor& out);
void gelu_scale_backward_launcher(const torch::Tensor& grad_out, const torch::Tensor& inp, const torch::Tensor& scale, torch::Tensor& grad_in, torch::Tensor& tmp_grad_scale);

inline void check_cublas(cublasStatus_t status, const char * msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error: ") + msg + " code=" + std::to_string((int)stat));
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
cublas_matmul_relu_backward( torch::Tensor grad_output, torch::Tensor mat1, torch::Tensor mat2, torch::Tensor out){
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(mat1.is_cuda() && mat2.is_cuda() && out.is_cuda(), "mat1, mat2, out must be CUDA tensors");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat32 && mat1.dtype() == torch::kFloat32 &&
                mat2.dtype() == torch::kFloat32 && out.dtype() == torch::kFloat32,
                "float32 required");

    int64_t M = mat1.size(0);
    int64_t K = mat1.size(1);
    int64_t N = mat2.size(1);

    TORCH_CHECK(mat2.size(0) == K, "mat2(0) must equal mat1(1)");
    TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out must be M x N");
    TORCH_CHECK(grad_output.size(0) == M && grad_output.size(1) == N, "grad_output must be M x N");
    
    auto grad_mat1 = torch::empty_like(mat1);
    auto grad_mat2 = torch::empty({K, N}, mat2.options());
    auto grad_bias = torch::empty({N}, mat2.options());

    // Step 1: compute dZ = dL/dY * mask, where mask is from ReLU forward
    auto dZ = torch::empty({M,N}, out.options());
    relu_mask_cuda(dZ, grad_output, out);
    
    static thread_local cublasHandle_t handle = nullptr;
    if(handle == nullptr) {
        check_cublas(cublasCreate(&handle), "cublasCreate");
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream");

    float alpha = 1.0f;
    float beta = 0.0f;

    const float * dz_ptr = dZ.data_ptr<float>();
    const float * mat1_ptr = mat1.data_ptr<float>();
    const float * mat2_ptr = mat2.data_ptr<float>();
    float * gradW_ptr = grad_mat2.data_ptr<float>();
    float * gradX_ptr = grad_mat1.data_ptr<float>();

// Treat row-major dZ (M x N) as column-major (N x M).  op(A)=A (no transpose) -> N x M
    // Treat row-major mat1 (M x K) as column-major (K x M). op(B)=B^T -> (K x M)^T = M x K
    // Multiply (N x M) * (M x K) = N x K  (column-major) -> corresponds to row-major K x N
    check_cublas(
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            (int)N,(int)K,(int)M,
            &alpha,
            dz_ptr, (int)N,
            mat1_ptr, (int) K,
            &beta,
            gradW_ptr, (int)N
        ), 
    "cublasSgemm gradW");
    check_cublas(
        cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            (int)K, (int)M, (int)N,
            &alpha,
            mat2_ptr, (int) N,
            dz_ptr, (int) N,
            &beta,
            gradX_ptr, (int)K
        ),
    "cublasSgemm gradX");
    
    grad_bias = dZ.sum(0);

    return std::make_tuple(grad_mat1, grad_mat2, grad_bias);

}

torch::Tensor add_bias_relu_forward( torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32, "float32 required");
    int M  = A.size(0);
    int K  = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "Incompatible matrix dims");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == N, "Bias length must equal N");

    auto out = torch::mm(A, B); // (M,K)
    add_bias_relu_cuda(out, bias); // add bias and apply ReLU
    return out;

}

torch::Tensor gelu_scale_forward(torch::Tensor inp, torch::Tensor scale){
    TORCH_CHECK(inp.is_cuda(), "inp must be CUDA");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA");
    TORCH_CHECK(inp.dtype() == torch::kFloat32 && scale.dtype() == torch::kFloat32, "float32 required");

    auto out = torch::empty_like(inp);
    gelu_scale_forward_launcher(inp, scale, out);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> gelu_scale_backward(torch::Tensor grad_out, torch::Tensor inp, torch::Tensor scale){
    TORCH_CHECK(grad_out.is_cuda() && inp.is_cuda() && scale.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(grad_out.dtype() == torch::kFloat32 && inp.dtype() == torch::kFloat32 && scale.dtype() == torch::kFloat32, "float32 required");

    auto grad_in = torch::empty_like(inp);
    auto tmp_grad_scale = torch::empty_like(inp);
    gelu_scale_backward_launcher(grad_out, inp, scale, grad_in, tmp_grad_scale);
    torch::Tensor grad_scale;
    if (scale.numel() == 1) {
        grad_scale = tmp_grad_scale.sum();
    } else if (scale.sizes() == inp.sizes()) {
        grad_scale = tmp_grad_scale; // elementwise gradient
    } else {
        // For other broadcast shapes, implement broadcast-aware reduction (not shown here)
        grad_scale = tmp_grad_scale.sum(); // fallback
    }
    return std::make_tuple(grad_in, grad_scale);

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cublas_matmul_relu_backward", &cublas_matmul_relu_backward, "Tiled Matmul ReLU backward (CUDA)");
    m.def("add_bias_relu_forward", &add_bias_relu_forward, "Add bias and ReLU (CUDA)");
    m.def("gelu_scale_forward", &gelu_scale_forward,"gelu forward" );
    m.def("gelu_scale_backward", &gelu_scale_backward, "gelu backward");
}
