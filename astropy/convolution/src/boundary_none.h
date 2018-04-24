// Licensed under a 3-clause BSD style license - see LICENSE.rst
#ifndef BOUNDARY_NONE_INCL
#define BOUNDARY_NONE_INCL

#include <stdbool.h>
#include <stdlib.h>

#include "numpy/ndarrayobject.h"
#define DTYPE npy_float64

void convolveNd_boundary_none_c(DTYPE * const result,
        const DTYPE * const f,
        const unsigned n_dim,
        const size_t * const image_shape,
        const DTYPE * const g,
        const size_t * const kernel_shape,
        const bool nan_interpolate,
        const unsigned n_threads);

// 1D
void convolve1d_boundary_none_c(DTYPE * const result,
        const DTYPE * const f, const size_t nx,
        const DTYPE * const g, const size_t nkx,
        const bool nan_interpolate,
        const unsigned n_threads);
inline __attribute__((always_inline)) void convolve1d_boundary_none(DTYPE * const result,
        const DTYPE * const f, const size_t nx,
        const DTYPE * const g, const size_t nkx,
        const bool nan_interpolate,
        const unsigned n_threads);

// 2D
void convolve2d_boundary_none_c(DTYPE * const result,
        const DTYPE * const f, const size_t nx, const size_t ny,
        const DTYPE * const g, const size_t nkx, const size_t nky,
        const bool nan_interpolate,
        const unsigned n_threads);
inline __attribute__((always_inline)) void convolve2d_boundary_none(DTYPE * const result,
        const DTYPE * const f, const size_t nx, const size_t ny,
        const DTYPE * const g, const size_t nkx, const size_t nky,
        const bool nan_interpolate,
        const unsigned n_threads);

// 3D
void convolve3d_boundary_none_c(DTYPE * const result,
        const DTYPE * const f, const size_t nx, const size_t ny, const size_t nz,
        const DTYPE * const g, const size_t nkx, const size_t nky, const size_t nkz,
        const bool nan_interpolate,
        const unsigned n_threads);
inline __attribute__((always_inline)) void convolve3d_boundary_none(DTYPE * const result,
        const DTYPE * const f, const size_t nx, const size_t ny, const size_t nz,
        const DTYPE * const g, const size_t nkx, const size_t nky, const size_t nkz,
        const bool nan_interpolate,
        const unsigned n_threads);

#endif
