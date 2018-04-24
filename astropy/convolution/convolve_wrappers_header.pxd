# Licensed under a 3-clause BSD style license - see LICENSE.rst

from numpy cimport npy_float64
ctypedef npy_float64 DTYPE

cdef extern from "src/boundary_none.h":

    void convolveNd_boundary_none_c(DTYPE * const result,
        const DTYPE * const f,
        const unsigned n_dim,
        const size_t * const image_shape,
        const DTYPE * const g,
        const size_t * const kernel_shape,
        const bint nan_interpolate,
        const unsigned n_threads);

cdef extern from "src/boundary_padded.h":

    void convolveNd_padded_boundary_c(DTYPE * const result,
        const DTYPE * const f,
        const unsigned n_dim,
        const size_t * const image_shape,
        const DTYPE * const g,
        const size_t * const kernel_shape,
        const bint nan_interpolate,
        const unsigned n_threads);
