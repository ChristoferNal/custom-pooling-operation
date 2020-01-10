from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(name='pooling_cpp',
     ext_modules=[CppExtension('pooling_cpp', ['pooling.cpp'])],
     cmdclass={'build_ext':BuildExtension})
