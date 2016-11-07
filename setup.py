from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import line_profiler
print numpy.get_include()

#from Cython.Compiler.Options import directive_defaults

#directive_defaults['linetrace'] = True
#directive_defaults['binding'] = True

extensions = [
    Extension("friction", ["friction.pyx"],
            include_dirs = [numpy.get_include()],
            compiler_directives={'linetrace': True},
            define_macros=[('CYTHON_TRACE', '1')],
            )
]
setup(
    name = "friction",
    ext_modules = cythonize(extensions)
)
