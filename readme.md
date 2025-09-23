# simple_nn_pycud

A compact PyTorch C++/CUDA extension that implements a fused MatMul + Bias + ReLU layer (forward) and a cuBLAS-accelerated backward (with a CUDA ReLU-mask kernel).
This repo is a correctness-first, performance-aware example showing how to write, debug, and benchmark small custom CUDA ops that interoperate with PyTorch.

## Build
```bash
cd lib_extension

python setup.py build_ext --inplace

python -m pip install -e .
cd ..
```

## Run

```bash
Quick gradient correctness check: python quick_check.py

Train example: python my_nn.py

Benchmark: python benchmarks.py --batch 1024 --in_features 2048 --hidden 1024
```

## Troubleshooting
Ensure tensors are float32, CUDA and contiguous.
