import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ["-fopenmp", "-std=c++17", "-DNDEBUG", "-O3"]
linker_args = ["-fopenmp"]


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __str__(self):
        import pybind11

        return pybind11.get_include()


ext_modules = [
    Extension(
        "Nigh",
        ["src/python/SE3_bindings.cpp"],
        include_dirs=[get_pybind_include(), "src/",],
        language="c++",
        extra_compile_args=cpp_args,
        extra_link_args=linker_args,
    ),
]

setup(
    name="NighWrapper",
    version="0.0.1",
    author="Ming Xu",
    author_email="mingda.xu@hdr.qut.edu.au",
    description="SE(3) NN search wrapper for Nigh",
    ext_modules=ext_modules,
)
