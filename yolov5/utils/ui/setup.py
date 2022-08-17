# all .pyx files in a folder
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

import numpy


setup(
    name = 'cui_util',
    ext_modules = cythonize(
        Extension(
            "cui_util", sources=["cui_util.pyx"],
            extra_compile_args=[
                "-O3", 
                "-ffast-math", "-fno-finite-math-only"  # need isnan function in code
            ],
        ),
        annotate=True),
    include_dirs=[numpy.get_include()],
    # extra_compile_args = ["-ffast-math"]
)