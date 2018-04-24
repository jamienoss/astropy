# Licensed under a 3-clause BSD style license - see LICENSE.rst

cimport libc.stdlib

import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DTYPE

# import convolve_wrappers_header.pxd
cimport convolve_wrappers_header

def _convolveNd_boundary_none(np.ndarray[DTYPE, mode='c'] result,
                              np.ndarray[DTYPE, mode='c'] f,
                              unsigned n_dim,
                              np.ndarray[size_t, mode='c'] image_shape,
                              np.ndarray[DTYPE, mode='c'] g,
                              np.ndarray[size_t, mode='c'] kernel_shape,
                              bint nan_interpolate,
                              int n_threads):

    return convolve_wrappers_header.convolveNd_boundary_none_c(<DTYPE*> result.data,
                                                        <DTYPE*> f.data,
                                                        n_dim,
                                                        <size_t*> image_shape.data,
                                                        <DTYPE*> g.data,
                                                        <size_t*> kernel_shape.data,
                                                        nan_interpolate,
                                                        n_threads)

def _convolveNd_padded_boundary(np.ndarray[DTYPE, mode='c'] result,
                                np.ndarray[DTYPE, mode='c'] f,
                                unsigned n_dim,
                                np.ndarray[size_t, mode='c'] image_shape,
                                np.ndarray[DTYPE, mode='c'] g,
                                np.ndarray[size_t, mode='c'] kernel_shape,
                                bint nan_interpolate,
                                int n_threads):

    return convolve_wrappers_header.convolveNd_padded_boundary_c(<DTYPE*> result.data,
                                                          <DTYPE*> f.data,
                                                          n_dim,
                                                          <size_t*> image_shape.data,
                                                          <DTYPE*> g.data,
                                                          <size_t*> kernel_shape.data,
                                                          nan_interpolate,
                                                          n_threads)
