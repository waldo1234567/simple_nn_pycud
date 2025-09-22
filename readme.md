# simple_nn_pycud

Small PyTorch extension demonstrating fused MatMul + Bias + ReLU with cuBLAS-accelerated backward.

## Build
```bash
python setup.py build_ext --inplace

Run

Quick check: python quick_check.py

Train example: python my_nn.py

Benchmark: python benchmarks.py --batch 1024 --in_features 2048 --hidden 1024

Troubleshooting
Ensure tensors are float32, CUDA and contiguous.
