# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from distutils.extension import Extension
from astropy_helpers.openmp_helpers import add_openmp_flags_if_available

CONVOLVE_PKGDIR = os.path.relpath(os.path.dirname(__file__))

SRC_DIR = os.path.join(CONVOLVE_PKGDIR, 'src')
SRC_FILES = [os.path.join(SRC_DIR, filename)
              for filename in ['boundary_none.c',
                               'boundary_padded.c']]

def get_extensions():
    # Add '-Rpass-missed=.*' to ``extra_compile_args`` when compiling with clang
    # to report missed optimizations
    lib_convolve_ext = Extension(name='astropy.convolution.lib_convolve',
                                 sources=SRC_FILES,
                                 extra_compile_args=['-UNDEBUG', '-fPIC'],
                                 include_dirs=["numpy", SRC_DIR],
                                 language='c')

    add_openmp_flags_if_available(lib_convolve_ext)

    cython_wrapper_ext = Extension(name='astropy.convolution.convolve_wrappers',
                                   sources=[os.path.join(CONVOLVE_PKGDIR, 'convolve_wrappers.pyx')],
                                   include_dirs=["numpy", SRC_DIR],
                                   libraries=['astropy.convolution.lib_convolve'],
                                   library_dirs=[])

    return [lib_convolve_ext, cython_wrapper_ext]

#extra_compile_args=['-I'+SRC_DIR],
#extra_link_args=['-L'+CONVOLVE_PKGDIR],