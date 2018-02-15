# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from distutils.extension import Extension
from astropy_helpers.openmp_helpers import add_openmp_flags_if_available

CONV_PKGDIR = os.path.relpath(os.path.dirname(__file__))

#SRC_FILES = glob.glob(os.path.join(CONV_SRC, '*.c'))
SRC_FILES = [os.path.join(CONV_PKGDIR, filename)
              for filename in ['boundary_none_direct.c']]

def get_extensions():
    conv_c_ext = Extension(name='conv_c', sources=SRC_FILES,
                 extra_compile_args=['-O3'],
                 language='c')

    add_openmp_flags_if_available(conv_c_ext)
    
    return [conv_c_ext]
