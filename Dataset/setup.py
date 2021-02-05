from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='test_module',
    ext_modules=cythonize('test.pyx'),
)