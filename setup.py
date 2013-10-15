from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

cython_modules = ['fast_pga.pyx', 'inpaint.pyx']

setup(name='sfs',
      version='0.1',
      description='Shape from shading',
      author='Patrick Snape',
      author_email='p.snape@imperial.ac.uk',
      include_dirs=[np.get_include()],
      ext_modules=cythonize(cython_modules, nthreads=2,
                            quiet=True, language='c++'),
      packages=find_packages()
)
