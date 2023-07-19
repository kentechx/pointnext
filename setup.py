from setuptools import setup, find_packages

setup(
    name='pointnext',
    packages=find_packages(),
    version='0.0.1',
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
        'pykeops>=2.1.2',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
