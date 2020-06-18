import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig

# cpp_args = ['-std=c++17', '--verbose']
#cpp_args = ['-std=c++17', '-DNDEBUG', '-O3']
#cpp_args = ['-fopenmp', '-std=c++17', '-DNDEBUG', '-O3', '-liomp5', '-lipthread', '-lirc']
#cpp_args = ['-fopenmp', '-std=c++17', '-DNDEBUG', '-O3', '-fno-elide-constructors']
cpp_args = ['-fopenmp', '-std=c++17', '-DNDEBUG', '-O3']
linker_args = ['-fopenmp']
# cpp_args = ['-std=c++17', '-std=c++11', '-std=c++14']

ext_modules = [
    Extension(
    'Nigh',
        ['src/python/SE3_bindings.cpp'],
        include_dirs=['/usr/include/pybind11/include', '/usr/include/eigen3/', 'src/', '/usr/include/python3.6m'],
    language='c++',
    extra_compile_args = cpp_args,
    extra_link_args = linker_args,
    ),
]

setup(
    name='NighWrapper',
    version='0.0.1',
    author='Ming Xu',
    author_email='mingda.xu@hdr.qut.edu.au',
    description='SE(3) NN search wrapper for Nigh',
    ext_modules=ext_modules,
)
