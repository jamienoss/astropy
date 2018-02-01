# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from numpy import get_include as get_numpy_include
from distutils.extension import Extension
from astropy_helpers.openmp_helpers import add_openmp_flags_if_available

C_CONVOLVE_PKGDIR = os.path.relpath(os.path.dirname(__file__))

SRC_FILES = [os.path.join(C_CONVOLVE_PKGDIR, filename)
              for filename in ['boundary_none_direct.c',
                               'boundary_padded_direct.c',
                               'openmp_enabled.c']]

def get_extensions():
    print(get_numpy_include())
    c_convolve_ext = Extension(name='c_convolve', sources=SRC_FILES,
                 extra_compile_args=['-O3', '-fPIC'],# '-Rpass-missed=.*'],
                 include_dirs=[get_numpy_include()],
                 language='c')

    add_openmp_flags_if_available(c_convolve_ext)
    
    return [c_convolve_ext]
