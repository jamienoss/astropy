# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from distutils.extension import Extension
from astropy_helpers.openmp_helpers import add_openmp_flags_if_available

CONVOLVE_PKGDIR = os.path.relpath(os.path.dirname(__file__))

SRC_FILES = [os.path.join(CONVOLVE_PKGDIR, filename)
              for filename in ['boundary_none_direct.c']]

def get_extensions():
    c_convolve_ext = Extension(name='c_convolve', sources=SRC_FILES,
                 extra_compile_args=['-O3', '-fPIC'],
                 language='c')

    add_openmp_flags_if_available(c_convolve_ext)
    
    return [c_convolve_ext]
