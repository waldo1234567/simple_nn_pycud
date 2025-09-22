#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h> 
#include <torch/extension.h>
#include <math_constants.h>

__device__ inline float gelu_scalar(float x){
    const float c  = 0.7978845608028654f; 
    const float d = 0.044715f;
    float xx = x + d * x * x * x;
    float t = tanhf(c * xx);
    return 0.5f * x * (1.0f + t);
}


__device__ inline float gelu_grad_scalar(float x){
    const float c  = 0.7978845608028654f; 
    const float d = 0.044715f;
    float xx = x + d * x * x * x;
    float t = tanhf(c * xx);
    float left = 0.5f * (1.0f + t);
    float sech2 = 1.0f - t * t;
    float inner = c * (1.0f + 3.0f * d * x *x);
    float right = 0.5f * x * sech2 * inner;
    return left + right;
}


__global__ void relu_mask_kernel(float* dZ, const float* grad_out, const float* out, int64_t size) {
    int64_t index = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (index < size) {
        dZ[index] = out[index] > 0 ? grad_out[index] : 0.0f;
    }
}

__global__ void add_bias_relu_kernel(float* out, const float* bias, int M, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = M * N;
    if (idx < total) {
        int col = idx % N;
        float val = out[idx] + bias[col];
        out[idx] = val > 0 ? val : 0.0f;
    }
}

__global__ void gelu_scale_forward_kernel(
    const float * __restrict__ inp,
    const float * __restrict__ scale,
    float * __restrict__ out,
    int64_t N,
    bool has_scale_tensor
){
    int64_t idx = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for(int64_t i = idx; i < N; i += stride){
        float x = inp[i];
        float g = gelu_scalar(x);
        float s = has_scale_tensor ? scale[i] : scale[0];
        out[i] = g * s;
    }
}


__global__ void gelu_scale_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ inp,
    const float* __restrict__ scale,
    float* __restrict__ grad_in,
    float* __restrict__ tmp_grad_scale, 
    int64_t N,
    bool has_scale_tensor
){
    int64_t idx = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for(int64_t i = idx; i < N; i += stride){
        float x = inp[i];
        float g = gelu_scalar(x);
        float gprime = gelu_grad_scalar(x);
        float s = has_scale_tensor ? scale[i] : scale[0];
        float gout = grad_out[i];
        grad_in[i] = gout * s * gprime;
        if(tmp_grad_scale){
            tmp_grad_scale[i] = gout * g;
        }
    }
}

void relu_mask_cuda(torch::Tensor dZ, torch::Tensor grad_out, torch::Tensor out) {
    TORCH_CHECK( dZ.is_cuda(), "dZ must be a CUDA tensor");
    TORCH_CHECK( grad_out.is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(dZ.dtype() == torch::kFloat32 && grad_out.dtype() == torch::kFloat32 && out.dtype() == torch::kFloat32,
                "float32 required");
    int64_t size = out.numel();
    int threads = 256;
    int blocks = (int)(size + threads - 1) / threads;
    relu_mask_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        dZ.data_ptr<float>(), 
        grad_out.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        throw std::runtime_error(std::string("relu_mask_kernel launch failed: ") + cudaGetErrorString(err));
    }
}

void add_bias_relu_cuda(torch::Tensor out, torch::Tensor bias) {
    TORCH_CHECK( out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK( bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(out.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32,
                "float32 required");
    int64_t M = out.size(0);
    int64_t N = out.size(1);
    int64_t total = M * N;
    int threads = 256;
    int blocks = (int)(total + threads - 1) / threads;
    add_bias_relu_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        out.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        (int)M, (int)N);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        throw std::runtime_error(std::string("add_bias_relu_kernel launch failed: ") + cudaGetErrorString(err));
    }
}

void gelu_scale_forward_launcher(const torch::Tensor& inp, const torch::Tensor& scale, torch::Tensor& out){
    int64_t N = inp.numel();
    const float * inp_ptr = inp.data_ptr<float>();
    const float * scale_ptr = scale.data_ptr<float>();
    float * out_ptr = out.data_ptr<float>();
    bool has_scale_tensor = (scale.numel() != 1);

    int threads = 256;
    int blocks = (N + threads - 1)/ threads;
    if(blocks > 65535) blocks = 65535;
    gelu_scale_forward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        inp_ptr, scale_ptr, out_ptr, N , has_scale_tensor
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("gelu forward launch failed");
}

void gelu_scale_backward_launcher(const torch::Tensor& grad_out, const torch::Tensor& inp, const torch::Tensor& scale, torch::Tensor& grad_in, torch::Tensor& tmp_grad_scale){
    int64_t N = inp.numel();
    const float * grad_ptr = grad_out.data_ptr<float>();
    const float * inp_ptr = inp.data_ptr<float>();
    const float * scale_ptr = scale.data_ptr<float>();
    float* grad_in_ptr = grad_in.data_ptr<float>();
    float* tmp_ptr = tmp_grad_scale.data_ptr<float>();
    bool has_scale_tensor = (scale.numel() != 1);

    int threads = 256;
    int blocks = (N + threads - 1)/threads;
    if(blocks > 65535) blocks = 65535;
    gelu_scale_backward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        grad_ptr, inp_ptr, scale_ptr, grad_in_ptr, tmp_ptr ,N , has_scale_tensor
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("gelu forward launch failed");
}

