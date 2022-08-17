# all .pyx files in a folder
from distutils.core import setup
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

import numpy


setup(
    name = 'mot',
    ext_modules = cythonize(["utils/mot/*.pyx"], annotate=True),
    include_dirs=[numpy.get_include()],
    # extra_compile_args = ["-ffast-math"]
)