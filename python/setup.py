from setuptools.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(ext_modules=[
    Extension("au_cv",
        sources=["cv.pyx", "../src/cv.c"],
        extra_link_args=['-lm'])])
