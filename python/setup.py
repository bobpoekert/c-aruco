from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os

here = os.path.abspath(__file__)
here = here.split('/')


cwd = os.path.join(*here[:-2])

setup(ext_modules=cythonize([
    Extension("au_cv",
        sources=["cv.pyx"],
        include_dirs=['../include'],
        extra_objects=['/%s' % os.path.join(cwd, 'lib', 'cv.o')],
        extra_compile_args=['-g'],
        extra_link_args=['-lm'])],
    language_level=3,
    gdb_debug=True))
