#include <torch/extension.h>


__global__ void scalar_mul_forward_kernel(float* out, const float* in, float scalar, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        out[index] = in[index] * scalar;
    }
}

__global__ void scalar_mul_backward_kernel( float* grad_in, float* grad_out, float scalar, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        grad_in[index] = grad_out[index] * scalar;
    }
}

__global__ void matrix_mul_kernel(float* out, const float* A, const float* B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float value = 0.0f;
        for (int e = 0; e < N; ++e) {
            value += A[row * N + e] * B[e * K + col];
        }
        out[row * K + col] = value;
    }
}

__global__ void relu_kernel(float * data, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        data[index] = fmaxf(0.0f, data[index]);
    }
}

__global__ void relu_backward_kernel(float * grad_in, float * grad_out, float * out, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        grad_in[index] = out[index] > 0 ? grad_out[index] : 0.0f;
    }
}

__global__ void sigmoid_kernel(float * data, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        data[index] = 1.0f / (1.0f + expf(-data[index]));
    }
}

__global__ void sigmoid_backward_kernel(float * grad_in, float * grad_out, float * out, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        grad_in[index] = grad_out[index] * out[index] * (1.0f - out[index]);
    }
}


__global__ void tiled_matmul_kernel_relu_forward(float* out, const float* A, const float* B, const float * bias,int M, int N, int K) {
    __shared__ float tile_A[16][16];
    __shared__ float tile_B[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;

    for (int t = 0; t < (N + 15) / 16; ++t) {
        if (row < M && t * 16 + threadIdx.x < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * 16 + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < K && t * 16 + threadIdx.y < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * K + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int e = 0; e < 16; ++e) {
            value += tile_A[threadIdx.y][e] * tile_B[e][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        out[row * K + col] =fmaxf(value + bias[col], 0.0f); ;
    }
}


__global__ void tiled_matmul_kernel_relu_backward(
    const float* grad_out, // dL/dY
    const float* A,        // input X
    const float* B,        // weights W
    const float* out,      // Y = ReLU(XW+b)
    float* grad_A,         // dL/dX
    float* grad_B,         // dL/dW
    float* grad_bias,      // dL/db
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < K) {
        float grad= grad_out[row * K + col] * (out[row * K + col] > 0 ? 1.0f : 0.0f);

        // Compute gradient w.r.t. bias
        atomicAdd(&grad_bias[col], grad);

        // Compute gradient w.r.t. input
        for (int e = 0; e < N; ++e) {
            atomicAdd(&grad_A[row * N + e], B[e * K + col] * grad);
            atomicAdd(&grad_B[e * K + col], A[row * N + e] * grad);
        }
    }
}

void scalar_mul_forward_cuda(torch::Tensor out, torch::Tensor in, float scalar) {
    int size = in.numel();
    int threads = 256;
    int blocks = ceil((size + threads - 1) / threads);
    scalar_mul_forward_kernel<<<blocks, threads>>>(out.data_ptr<float>(), in.data_ptr<float>(), scalar, size);
}

void scalar_mul_backward_cuda(torch::Tensor grad_in, torch::Tensor grad_out, float scalar) {
    int size = grad_in.numel();
    int threads = 256;
    int blocks = ceil((size + threads - 1) / threads);
    scalar_mul_backward_kernel<<<blocks, threads>>>(grad_in.data_ptr<float>(), grad_out.data_ptr<float>(), scalar, size);
}

void matrix_mul_cuda(torch::Tensor out, torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int N = A.size(1);
    int K = B.size(1);

    dim3 threads(16, 16);
    dim3 blocks((K + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    matrix_mul_kernel<<<blocks, threads>>>(out.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), M, N, K);
}

void relu_cuda(torch::Tensor data) {
    int size = data.numel();
    int threads = 256;
    int blocks = ceil((size + threads - 1) / threads);
    relu_kernel<<<blocks, threads>>>(data.data_ptr<float>(), size);
}

void relu_backward_cuda(torch::Tensor grad_in, torch::Tensor grad_out, torch::Tensor out) {
    int size = grad_in.numel();
    int threads = 256;
    int blocks = ceil((size + threads - 1) / threads);
    relu_backward_kernel<<<blocks, threads>>>(grad_in.data_ptr<float>(), grad_out.data_ptr<float>(), out.data_ptr<float>(), size);
}

void sigmoid_cuda(torch::Tensor data) {
    int size = data.numel();
    int threads = 256;
    int blocks = ceil((size + threads - 1) / threads);
    sigmoid_kernel<<<blocks, threads>>>(data.data_ptr<float>(), size);
}

void sigmoid_backward_cuda(torch::Tensor grad_in, torch::Tensor grad_out, torch::Tensor out) {
    int size = grad_in.numel();
    int threads = 256;
    int blocks = ceil((size + threads - 1) / threads);
    sigmoid_backward_kernel<<<blocks, threads>>>(grad_in.data_ptr<float>(), grad_out.data_ptr<float>(), out.data_ptr<float>(), size);
}

void tiled_matmul_relu_forward_cuda(torch::Tensor out, torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
    int M = A.size(0);
    int N = A.size(1);
    int K = B.size(1);

    dim3 threads(16, 16);
    dim3 blocks((K + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    tiled_matmul_kernel_relu_forward<<<blocks, threads>>>(out.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), bias.data_ptr<float>(), M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("kernel launch failed: ") + cudaGetErrorString(err));
    }
}

void tiled_matmul_relu_backward_cuda(
    torch::Tensor grad_out, // dL/dY
    torch::Tensor A,        // input X
    torch::Tensor B,        // weights W
    torch::Tensor out,      // Y = ReLU(XW+b)
    torch::Tensor grad_A,   // dL/dX
    torch::Tensor grad_B,   // dL/dW
    torch::Tensor grad_bias // dL/db
) {
    TORCH_CHECK(grad_out.is_cuda() && A.is_cuda() && B.is_cuda() && out.is_cuda(), "all must be CUDA");
    TORCH_CHECK(grad_A.is_cuda() && grad_B.is_cuda() && grad_bias.is_cuda(), "grads must be CUDA");
    TORCH_CHECK(grad_A.is_contiguous() && grad_B.is_contiguous() && grad_bias.is_contiguous(), "grads must be contiguous");

    int M = A.size(0);
    int N = A.size(1);
    int K = B.size(1);

    dim3 threads(16, 16);
    dim3 blocks((K + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    tiled_matmul_kernel_relu_backward<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        out.data_ptr<float>(),
        grad_A.data_ptr<float>(),
        grad_B.data_ptr<float>(),
        grad_bias.data_ptr<float>(),
        M, N, K
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("kernel launch failed: ") + cudaGetErrorString(err));
    }
}