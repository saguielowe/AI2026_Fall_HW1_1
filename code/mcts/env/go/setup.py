from Cython.Build import cythonize
from setuptools import setup, Extension
import sys


# MSVC 和 GCC/Clang 的 C++17 选项写法不同。
cxx17_flag = "/std:c++17" if sys.platform == "win32" else "-std=c++17"

extensions = [
    Extension("goboard",
        sources=["goboard.pyx"],
        language="c++",
        extra_compile_args=[cxx17_flag],
        extra_link_args=[])]

setup(
    name = "goboard",
    ext_modules = cythonize(extensions)
)

# python setup.py build_ext --inplace
