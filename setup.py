from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

thisdir = Path(__file__).parent

ext_modules = [
    CUDAExtension(
        'pointnext._C',
        [
            'csrc/pointnet2_api.cpp',
            'csrc/ball_query.cpp',
            'csrc/ball_query_gpu.cu',
            'csrc/group_points.cpp',
            'csrc/group_points_gpu.cu',
            'csrc/interpolate.cpp',
            'csrc/interpolate_gpu.cu',
            'csrc/sampling.cpp',
            'csrc/sampling_gpu.cu',
        ],
        extra_compile_args={'nvcc': ['-O3']},
        include_dirs=[thisdir / 'csrc'],
    )
]

setup(
    name='pointnext',
    packages=find_packages(exclude=("csrc")),
    version='0.0.5',
    license='MIT',
    description='PointNext - Pytorch',
    author='Kaidi Shen',
    url='https://github.com/kentechx/pointnext',
    long_description_content_type='text/markdown',
    keywords=[
        '3D segmentation',
        '3D classification',
        'point cloud understanding',
    ],
    install_requires=[
        'torch>=1.10',
        'einops>=0.6.1',
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
