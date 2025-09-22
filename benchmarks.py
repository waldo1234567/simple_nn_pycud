import time
import torch
import numpy as np
import argparse
from torch import nn
import custom_extension 
from nn_layer import scalarMul, TiledMatmulReLU

use_wrapper = hasattr(globals().get('scalarMul', None), 'apply')
use_fused_wrapper = hasattr(globals().get('TiledMatmulReLU', None), 'apply')

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def time_it(fn, warmups = 10, repeats = 50):
    for _ in range(warmups):
        fn()
    times=[]
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    arr = np.array(times)
    return arr.mean(), arr.std(), times

def bench_forward(batch_size= 128, feat_dim = 256 , device='cuda', scalar = 1.0, repeats=50):
    device_t = torch.device(device)
    base_x = torch.randn(batch_size, feat_dim, device=device_t, dtype=torch.float32, requires_grad=False)
    def custom_forward():
        x = base_x.clone().requires_grad_(False)  # forward-only: no need for grad
        if use_wrapper:
            out = scalarMul.apply(x, float(scalar))
        else:
            out = custom_extension.scalar_mul_forward(x, float(scalar))
        out = torch.relu(out)
    def native_forward():
        x = base_x.clone().requires_grad_(False)
        _ = (x * float(scalar)).relu()
    return time_it(custom_forward, warmups=10 ,repeats=repeats), time_it(native_forward,warmups=10 ,repeats=repeats)   
 
def bench_forward_backward(batch_size= 128, feat_dim = 256 , device='cuda', scalar = 1.0, repeats=50):
    device_t = torch.device(device)
    base_x = torch.randn(batch_size, feat_dim, device=device_t, dtype=torch.float32)
    def custom_forward_backward():
        x = base_x.clone().requires_grad_(True)       # important: fresh leaf requiring grad
        if use_wrapper:
            out = scalarMul.apply(x, float(scalar))
        else:
            out = custom_extension.scalar_mul_forward(x, float(scalar))
        out = torch.relu(out)
        loss = out.sum()
        loss.backward()
    def native_forward_backward():
        x = base_x.clone().requires_grad_(True)
        out = (x * float(scalar)).relu()
        loss = out.sum()
        loss.backward()
    return time_it(custom_forward_backward, warmups=10 ,repeats=repeats), time_it(native_forward_backward,warmups=10 ,repeats=repeats)


def bench_matmul_forward(batch_size= 128, in_features=1024, hidden=256, device='cuda', repeats=50):
    device_t = torch.device(device)
    base_x = torch.randn(batch_size, in_features, device=device_t, dtype=torch.float32, requires_grad=False)
    W = torch.randn(in_features, hidden, device=device_t, dtype=torch.float32)
    b = torch.randn(hidden, device=device_t, dtype=torch.float32)
    
    def custom_forward():
        x = base_x.clone().requires_grad_(False)
        if use_fused_wrapper:
            out = TiledMatmulReLU.apply(x, W, b)   # python autograd wrapper calls C++ fused kernel
        else:
            out = custom_extension.tiled_matmul_relu_forward(x.contiguous(), W.contiguous(), b.contiguous())
    def native_forward():
        x = base_x.clone().requires_grad_(False)
        _ = (x @ W + b).relu()

    return time_it(custom_forward, warmups=10, repeats=repeats), time_it(native_forward, warmups=10, repeats=repeats)

def bench_matmul_forward_backward(batch_size= 128, in_features=1024, hidden=256, device='cuda', repeats=50):
    device_t = torch.device(device)
    base_x = torch.randn(batch_size, in_features, device=device_t, dtype=torch.float32)
    W = torch.randn(in_features, hidden, device=device_t, dtype=torch.float32)
    b = torch.randn(hidden, device=device_t, dtype=torch.float32)
    
    def custom_forward_backward():
        x = base_x.clone().requires_grad_(True)   # important: fresh leaf requiring grad
        if use_fused_wrapper:
            out = TiledMatmulReLU.apply(x, W, b)   # python autograd wrapper calls C++ fused kernel
        else:
            out = custom_extension.tiled_matmul_relu_forward(x.contiguous(), W.contiguous(), b.contiguous())
        loss = out.sum()
        loss.backward()
        
    def native_forward_backward():
        x = base_x.clone().requires_grad_(True)
        out = (x @ W + b).relu()
        loss = out.sum()
        loss.backward()
        
    return time_it(custom_forward_backward, warmups=10 ,repeats=repeats), time_it(native_forward_backward,warmups=10 ,repeats=repeats)

def bench_cpu(batch_size= 128, feat_dim = 256 , scalar = 1.0, repeats=50):
    device_t = torch.device('cpu')
    base_x = torch.randn(batch_size, feat_dim, device='cpu', dtype=torch.float32)
    def cpu_forward_backward():
        x = base_x.clone().requires_grad_(True)
        out = (x * float(scalar)).relu()
        loss = out.sum()
        loss.backward()
    return time_it(cpu_forward_backward, warmups=10 ,repeats=repeats)

def run_all(batch_size=128, in_features=1024, hidden=512, scalar=1.0, repeats=50):
    print(f"Benchmarking with batch size {batch_size}, in_features {in_features}, hidden {hidden}, scalar {scalar}, repeats {repeats}")
    if torch.cuda.is_available():
        print("\n-- Scalar Forward (GPU): custom vs native --")
        (c_mean, c_std, _), (n_mean, n_std, _) = bench_forward(batch_size, feat_dim=hidden, device='cuda', scalar=scalar, repeats=repeats)
        print(f"custom scalar forward: {c_mean*1000:.3f} ms ± {c_std*1000:.3f} ms")
        print(f"native scalar forward: {n_mean*1000:.3f} ms ± {n_std*1000:.3f} ms")

        print("\n-- Scalar Forward+Backward (GPU): custom vs native --")
        (c_mean, c_std, _), (n_mean, n_std, _) = bench_forward_backward(batch_size, feat_dim=hidden, device='cuda', scalar=scalar, repeats=repeats)
        print(f"custom scalar fwd+bwd: {c_mean*1000:.3f} ms ± {c_std*1000:.3f} ms")
        print(f"native scalar fwd+bwd: {n_mean*1000:.3f} ms ± {n_std*1000:.3f} ms")

        print("\n== Matmul+ReLU Forward (GPU): fused vs native ==")
        (cm_mean, cm_std, _), (nm_mean, nm_std, _) = bench_matmul_forward(batch_size, in_features, hidden, device='cuda', repeats=repeats)
        print(f"custom fused matmul forward: {cm_mean*1000:.3f} ms ± {cm_std*1000:.3f} ms")
        print(f"native matmul forward: {nm_mean*1000:.3f} ms ± {nm_std*1000:.3f} ms")

        print("\n== Matmul+ReLU Forward+Backward (GPU): fused vs native ==")
        (cm_mean, cm_std, _), (nm_mean, nm_std, _) = bench_matmul_forward_backward(batch_size, in_features, hidden, device='cuda', repeats=repeats)
        print(f"custom fused matmul fwd+bwd: {cm_mean*1000:.3f} ms ± {cm_std*1000:.3f} ms")
        print(f"native matmul fwd+bwd: {nm_mean*1000:.3f} ms ± {nm_std*1000:.3f} ms")

    else:
        print("CUDA not available on this machine; skipping GPU tests.")

    print("\n-- Forward+Backward (CPU native) --")
    cpu_mean, cpu_std, _ = bench_cpu(batch_size, feat_dim=hidden, scalar=scalar, repeats=repeats)
    print(f"cpu fwd+bwd: {cpu_mean*1000:.3f} ms ± {cpu_std*1000:.3f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--in_features", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--scalar", type=float, default=1.0)
    args = parser.parse_args()
    run_all(batch_size=args.batch, in_features=args.in_features, hidden=args.hidden, repeats=args.repeats, scalar=args.scalar)