import torch
import custom_extension
import lib_extension

class scalarMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scalar):
        ctx.scalar = scalar
        return custom_extension.scalar_mul_forward(input, scalar)

    @staticmethod
    def backward(ctx, grad_output):
        scalar = ctx.scalar
        grad_input = custom_extension.scalar_mul_backward(grad_output, scalar)
        return grad_input, None
    

class ReLUFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return custom_extension.relu_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = custom_extension.relu_backward(grad_output, input)
        return grad_input
    
class SigmoidFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = custom_extension.sigmoid_forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = custom_extension.sigmoid_backward(grad_output, output)
        return grad_input
    
class Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat1, mat2):
        A_c = mat1.contiguous()
        B_c = mat2.contiguous()
        ctx.save_for_backward(A_c, B_c)
        out = torch.matmul(A_c, B_c)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_B = None
        # grad_output: [M, K] (or [batch, K])
        if ctx.needs_input_grad[0]:
            # dL/dA = grad_output @ B^T
            grad_A = torch.matmul(grad_output, B.transpose(-2, -1))
        if ctx.needs_input_grad[1]:
            # dL/dB = A^T @ grad_output
            grad_B = torch.matmul(A.transpose(-2, -1), grad_output)
        return grad_A, grad_B
    
class TiledMatmulReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat1, mat2, bias):
        A_c = mat1.contiguous()
        B_c = mat2.contiguous()
        bias_c = bias.contiguous()
        out = lib_extension.add_bias_relu_forward(A_c, B_c, bias_c)
        ctx.save_for_backward(A_c, B_c, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        A, B, out = ctx.saved_tensors
        grad_A = grad_B = grad_bias = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            go = grad_output.contiguous()
            grad_A, grad_B, grad_bias = lib_extension.cublas_matmul_relu_backward(go, A, B, out)
        return grad_A, grad_B, grad_bias
    
