from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension

setup(
    name='minillm',
    version='0.1.0',
    packages=find_packages(include=['minillm', 'minillm.*']),
    entry_points={
        'console_scripts': ['minillm=minillm.run:main']
    }
)

setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'quant_cuda', 
        [
        	'minillm/engine/cuda/quant_cuda.cpp', 
        	'minillm/engine/cuda/quant_cuda_kernel.cu'
        ]
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
