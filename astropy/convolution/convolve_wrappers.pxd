# Licensed under a 3-clause BSD style license - see LICENSE.rst

cdef extern from "lib_convolve/boundary_none.h":

    cimport "numpy/ndarrayobject.h"
    ctypedef npy_float64 DTYPE

    void convolveNd_boundary_none_c(DTYPE * const result,
        const DTYPE * const f,
        const unsigned n_dim,
        const size_t * const image_shape,
        const DTYPE * const g,
        const size_t * const kernel_shape,
        const bool nan_interpolate,
        const unsigned n_threads);
