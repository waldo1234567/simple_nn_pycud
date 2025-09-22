from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension,CUDAExtension

setup(
    name='custom_extension',
    ext_modules=[
        CUDAExtension(
            'custom_extension',
            ['custom_extension.cpp', 'custom_extension_kernel.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)