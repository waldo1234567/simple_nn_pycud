import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension,CUDAExtension

CUDA_PATH = os.environ.get('CUDA_PATH', r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9")

lib_dir = os.path.join(CUDA_PATH, 'lib', 'x64')
cublas_lib = os.path.join(lib_dir, 'cublas.lib')
cudart_lib = os.path.join(lib_dir, 'cudart.lib')    

if not os.path.exists(cublas_lib):
    raise RuntimeError(f"Could not find cublas.lib at {cublas_lib}. Check your CUDA_PATH or install CUDA.")

setup(
    name='lib_extension',
    ext_modules=[
        CUDAExtension(
            name = 'lib_extension',
            sources = ['lib_extension.cpp', 'lib_extension_kernel.cu', 'cublas_utils.cpp'],
            extra_objects = [cublas_lib, cudart_lib],
            extra_compile_args = {'cxx': [''], 'nvcc': ['']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

