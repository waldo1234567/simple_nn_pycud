#include <torch/extension.h>
#include <tuple>

void scalar_mul_forward_cuda( torch::Tensor output, torch::Tensor input, float scalar);
void scalar_mul_backward_cuda( torch::Tensor grad_input, torch::Tensor grad_output, float scalar);
void matrix_mul_cuda( torch::Tensor output, torch::Tensor mat1, torch::Tensor mat2);
void relu_cuda( torch::Tensor output);
void relu_backward_cuda( torch::Tensor grad_input, torch::Tensor grad_output, torch::Tensor input);
void sigmoid_cuda( torch::Tensor output);
void sigmoid_backward_cuda( torch::Tensor grad_input, torch::Tensor grad_output, torch::Tensor output);
void tiled_matmul_relu_forward_cuda( torch::Tensor output, torch::Tensor mat1, torch::Tensor mat2, torch::Tensor bias);
void tiled_matmul_relu_backward_cuda( torch::Tensor grad_out, // dL/dY
    torch::Tensor A,        // input X
    torch::Tensor B,        // weights W
    torch::Tensor out,      // Y = ReLU(XW+b)
    torch::Tensor grad_A,   // dL/dX (output)
    torch::Tensor grad_B,   // dL/dW (output)
    torch::Tensor grad_bias // dL/db (output)
);

torch::Tensor scalar_mul_forward( torch::Tensor input, float scalar) {
    TORCH_CHECK( input.is_cuda(), "input must be a CUDA tensor");
    auto output = torch::empty_like(input);
    scalar_mul_forward_cuda( output, input, scalar);
    return output;
}

torch::Tensor scalar_mul_backward( torch::Tensor grad_output, float scalar) {
    TORCH_CHECK( grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    auto grad_input = torch::empty_like(grad_output);
    scalar_mul_backward_cuda( grad_input, grad_output, scalar);
    return grad_input;
}

torch::Tensor matrix_mul( torch::Tensor mat1, torch::Tensor mat2) {
    TORCH_CHECK( mat1.is_cuda(), "mat1 must be a CUDA tensor");
    TORCH_CHECK( mat2.is_cuda(), "mat2 must be a CUDA tensor");
    TORCH_CHECK( mat1.size(1) == mat2.size(0), "Incompatible matrix dimensions");
    auto output = torch::empty({mat1.size(0), mat2.size(1)}, mat1.options());
    matrix_mul_cuda( output, mat1, mat2);
    return output;
}

torch::Tensor relu( torch::Tensor input) {
    TORCH_CHECK( input.is_cuda(), "input must be a CUDA tensor");
    auto output = torch::empty_like(input);
    relu_cuda( output);
    return output;
}

torch::Tensor relu_backward( torch::Tensor grad_output, torch::Tensor input) {
    TORCH_CHECK( grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK( input.is_cuda(), "input must be a CUDA tensor");
    auto grad_input = torch::empty_like(grad_output);
    relu_backward_cuda( grad_input, grad_output, input);
    return grad_input;
}

torch::Tensor sigmoid( torch::Tensor input) {
    TORCH_CHECK( input.is_cuda(), "input must be a CUDA tensor");
    auto output = torch::empty_like(input);
    sigmoid_cuda( output);
    return output;
}

torch::Tensor sigmoid_backward( torch::Tensor grad_output, torch::Tensor output) {
    TORCH_CHECK( grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK( output.is_cuda(), "output must be a CUDA tensor");
    auto grad_input = torch::empty_like(grad_output);
    sigmoid_backward_cuda( grad_input, grad_output, output);
    return grad_input;
}

torch::Tensor tiled_matmul_relu_forward( torch::Tensor mat1, torch::Tensor mat2, torch::Tensor bias) {
    TORCH_CHECK( mat1.is_cuda(), "mat1 must be a CUDA tensor");
    TORCH_CHECK( mat2.is_cuda(), "mat2 must be a CUDA tensor");
    TORCH_CHECK( bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK( mat1.size(1) == mat2.size(0), "Incompatible matrix dimensions");
    TORCH_CHECK( bias.size(0) == mat2.size(1), "Incompatible bias dimensions");
    auto output = torch::empty({mat1.size(0), mat2.size(1)}, mat1.options());
    tiled_matmul_relu_forward_cuda( output, mat1, mat2, bias);
    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tiled_matmul_relu_backward( torch::Tensor grad_output, torch::Tensor mat1, torch::Tensor mat2, torch::Tensor output) {
    TORCH_CHECK( grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK( mat1.is_cuda(), "mat1 must be a CUDA tensor");
    TORCH_CHECK( mat2.is_cuda(), "mat2 must be a CUDA tensor");
    TORCH_CHECK( output.is_cuda(), "output must be a CUDA tensor");
    auto grad_mat1 = torch::zeros_like(mat1);
    auto grad_mat2 = torch::zeros_like(mat2);
    auto grad_bias = torch::zeros({mat2.size(1)}, mat2.options());

    tiled_matmul_relu_backward_cuda( grad_output,mat1,mat2,output,grad_mat1,grad_mat2, grad_bias);
    return std::make_tuple(grad_mat1, grad_mat2, grad_bias);
}

//pybind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scalar_mul_forward", &scalar_mul_forward, "scalar multiplication forward (CUDA)");
    m.def("scalar_mul_backward", &scalar_mul_backward, "scalar multiplication backward (CUDA)");
    m.def("matrix_mul", &matrix_mul, "Matrix multiplication (CUDA)");
    m.def("relu_forward", &relu, "ReLU forward (CUDA)");
    m.def("relu_backward", &relu_backward, "ReLU backward (CUDA)");
    m.def("sigmoid_forward", &sigmoid, "Sigmoid forward (CUDA)");
    m.def("sigmoid_backward", &sigmoid_backward, "Sigmoid backward (CUDA)");
    m.def("tiled_matmul_relu_forward", &tiled_matmul_relu_forward, "Tiled Matrix multiplication + ReLU forward (CUDA)");
    m.def("tiled_matmul_relu_backward", &tiled_matmul_relu_backward, "Tiled Matrix multiplication + ReLU backward (CUDA)");
}

