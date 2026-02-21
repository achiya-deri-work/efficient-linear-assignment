
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob

# Find sources
sources = glob.glob('src/csrc/*.cpp') + glob.glob('src/csrc/*.cu')

setup(
    name='efficient_linear_assignment',
    version='0.1.0',
    package_dir={'': 'src'},
    packages=['efficient_linear_assignment', 'efficient_linear_assignment.auction', 'efficient_linear_assignment.sinkhorn', 'efficient_linear_assignment.dual_ascent', 'efficient_linear_assignment.routing'],
    ext_modules=[
        CUDAExtension(
            name='efficient_linear_assignment.efficient_linear_assignment_cpp',
            sources=sources,
            include_dirs=[os.path.abspath('src/csrc'), os.path.abspath('third_party/cutlass/include')],
            extra_compile_args={'cxx': ['-O3'],
                                'nvcc': ['-O3', '--use_fast_math',
                                         '-gencode=arch=compute_80,code=sm_80', 
                                         '-gencode=arch=compute_89,code=sm_89', 
                                         '-gencode=arch=compute_90,code=sm_90',
                                         '-gencode=arch=compute_89,code=compute_89', # PTX for Ada (Safe JIT fallback for Blackwell)
                                         ]}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['torch', 'numpy', 'scipy', 'triton'],
)
