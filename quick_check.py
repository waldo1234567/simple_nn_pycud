import torch
import lib_extension as lib
import torch.nn.functional as F

# small test in python
A = torch.randn(3,5, device='cuda', dtype=torch.float32, requires_grad=True)
W = torch.randn(5,4, device='cuda', dtype=torch.float32, requires_grad=True)
b = torch.randn(4, device='cuda', dtype=torch.float32, requires_grad=True)

out = F.relu(A @ W + b)
go = torch.randn_like(out)

# PyTorch grads
out.backward(go, retain_graph=True)
gA_ref = A.grad.clone(); gW_ref = W.grad.clone(); gb_ref = b.grad.clone()
A.grad.zero_(); W.grad.zero_(); b.grad.zero_()

# Your custom backward
out_custom = lib.add_bias_relu_forward(A.contiguous(), W.contiguous(), b.contiguous())
gA_c, gW_c, gb_c = lib.cublas_matmul_relu_backward(go.contiguous(), A.contiguous(), W.contiguous(), out_custom)
# compare
print(torch.allclose(gA_ref, gA_c, atol=1e-4))
print(torch.allclose(gW_ref, gW_c, atol=1e-4))
print(torch.allclose(gb_ref, gb_c, atol=1e-5))