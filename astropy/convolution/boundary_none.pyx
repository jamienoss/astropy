# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
cimport numpy as np


DTYPE = float
ctypedef np.float_t DTYPE_t

cdef extern from "numpy/npy_math.h" nogil:
    bint npy_isnan(double x)

cimport cython


@cython.boundscheck(False)  # turn off bounds-checking for entire function
def convolve1d_boundary_none(np.ndarray[DTYPE_t, ndim=1] f,
                             np.ndarray[DTYPE_t, ndim=1] g,
                             bint normalize_by_kernel):

    if g.shape[0] % 2 != 1:
        raise ValueError("Convolution kernel must have odd dimensions")

    assert f.dtype == DTYPE and g.dtype == DTYPE

    cdef int nx = f.shape[0]
    cdef int nkx = g.shape[0]
    cdef int wkx = nkx // 2

    # The following need to be set to zeros rather than empty because the
    # boundary does not get reset.
    cdef np.ndarray[DTYPE_t, ndim=1] conv = np.zeros([nx], dtype=DTYPE)

    cdef unsigned int i, ii

    cdef int iimin, iimax

    cdef DTYPE_t top, bot, ker, val

    # release the GIL
    with nogil:

        # Now run the proper convolution
        for i in range(wkx, nx - wkx):
            top = 0.
            bot = 0.
            for ii in range(i - wkx, i + wkx + 1):
                val = f[ii]
                ker = g[<unsigned int>(nkx - 1 - (wkx + ii - i))]
                if not npy_isnan(val):
                    top += val * ker
                    bot += ker
            if normalize_by_kernel:
                if bot == 0:
                    conv[i] = f[i]
                else:
                    conv[i] = top / bot
            else:
                conv[i] = top
    # GIL acquired again here
    return conv

@cython.boundscheck(False)  # turn off bounds-checking for entire function
def convolve2d_boundary_none(np.ndarray[DTYPE_t, ndim=2] f,
                             np.ndarray[DTYPE_t, ndim=2] g,
                             bint normalize_by_kernel):

    if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
        raise ValueError("Convolution kernel must have odd dimensions")

    assert f.dtype == DTYPE and g.dtype == DTYPE

    cdef int nx = f.shape[0]
    cdef int ny = f.shape[1]
    cdef int nkx = g.shape[0]
    cdef int nky = g.shape[1]
    cdef int wkx = nkx // 2
    cdef int wky = nky // 2

    # The following need to be set to zeros rather than empty because the
    # boundary does not get reset.
    cdef np.ndarray[DTYPE_t, ndim=2] conv = np.zeros([nx, ny], dtype=DTYPE)

    cdef unsigned int i, j, ii, jj

    cdef int iimin, iimax, jjmin, jjmax

    cdef DTYPE_t top, bot, ker, val

    # release the GIL
    with nogil:

        # Now run the proper convolution
        for i in range(wkx, nx - wkx):
            for j in range(wky, ny - wky):
                top = 0.
                bot = 0.
                for ii in range(i - wkx, i + wkx + 1):
                    for jj in range(j - wky, j + wky + 1):
                        val = f[ii, jj]
                        ker = g[<unsigned int>(nkx - 1 - (wkx + ii - i)),
                                <unsigned int>(nky - 1 - (wky + jj - j))]
                        if not npy_isnan(val):
                            top += val * ker
                            bot += ker
                if normalize_by_kernel:
                    if bot == 0:
                        conv[i, j] = f[i, j]
                    else:
                        conv[i, j] = top / bot
                else:
                    conv[i, j] = top
    # GIL acquired again here
    return conv

@cython.boundscheck(False)  # turn off bounds-checking for entire function
def convolve2d_boundary_none_dev(np.ndarray[DTYPE_t, ndim=2] f,
                             np.ndarray[DTYPE_t, ndim=2] g,
                             bint normalize_by_kernel):

    if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
        raise ValueError("Convolution kernel must have odd dimensions")

    assert f.dtype == DTYPE and g.dtype == DTYPE

    cdef int nx = f.shape[0]
    cdef int ny = f.shape[1]
    cdef int nkx = g.shape[0]
    cdef int nky = g.shape[1]
    cdef int wkx = nkx // 2
    cdef int wky = nky // 2

    # The following need to be set to zeros rather than empty because the
    # boundary does not get reset.
    cdef np.ndarray[DTYPE_t, ndim=2] conv = np.zeros([nx, ny], dtype=DTYPE)

    cdef unsigned int i, j, ii, jj
    cdef unsigned int nkx_minus_1 = nkx-1, nky_minus_1 = nky-1
    cdef unsigned int wkx_minus_i, wky_minus_j
    cdef unsigned int ker_i, ker_j
    cdef unsigned int ny_minus_wky = ny - wky
    cdef unsigned int i_minus_wkx, wkx_plus_1 = wkx + 1
    cdef unsigned int j_minus_wky, wky_plus_1 = wky + 1
    cdef unsigned int i_plus_wkx_plus_1, j_plus_wky_plus_1
    cdef unsigned int nkx_minus_1_minus_wkx_plus_i, nky_minus_1_minus_wky_plus_j
    cdef int iimin, iimax, jjmin, jjmax

    cdef DTYPE_t top, bot, ker, val

    # release the GIL
    with nogil:

        # Now run the proper convolution
        for i in range(wkx, nx_minus_wkx):
            wkx_minus_i = wkx - i # wkx - 1
            i_minus_wkx = i - wkx # i - wkx
            i_plus_wkx_plus_1 = i + wkx_plus_1 # i + wkx + 1
            nkx_minus_1_minus_wkx_plus_i = nkx_minus_1 - wkx_minus_i # nkx - 1 - (wkx - i)
    
            for j in range(wky, ny_minus_wky):
                wky_minus_j = wkx - j # wky - j
                j_minus_wky = j - wky # j - wky
                j_plus_wky_plus_1 = j + wky_plus_1 # j + wky + 1
                nky_minus_1_minus_wky_plus_j = nky_minus_1 - wky_minus_j # nky - 1 - (wky - i)
                top = 0.
                #bot = 0.
                for ii in range(i_minus_wkx, i_plus_wkx_plus_1):
                    ker_i = nkx_minus_1_minus_wkx_plus_i - ii # nkx - 1 - (wkx + ii - i)
                    for jj in range(j_wky_wky, j_plus_wky_plus_1):
                        ker_j = nky_minus_1_minus_wky_plus_j - jj # nky - 1 - (wky + jj - j)
                        val = f[ii, jj]
                        ker = g[ker_i, ker_j]
                        #if not npy_isnan(val):#replace NaNs with 0 to remove this IF
                        top += val * ker
                            #bot += ker
                #if normalize_by_kernel:
                    #if bot == 0:
                    #    conv[i, j] = f[i, j]
                    #else:
                    #conv[i, j] = top / bot
                #else:
                conv[i, j] = top
    # GIL acquired again here
    return conv


@cython.boundscheck(False)  # turn off bounds-checking for entire function
def convolve3d_boundary_none(np.ndarray[DTYPE_t, ndim=3] f,
                             np.ndarray[DTYPE_t, ndim=3] g,
                             bint normalize_by_kernel):

    if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1 or g.shape[2] % 2 != 1:
        raise ValueError("Convolution kernel must have odd dimensions")

    assert f.dtype == DTYPE and g.dtype == DTYPE

    cdef int nx = f.shape[0]
    cdef int ny = f.shape[1]
    cdef int nz = f.shape[2]
    cdef int nkx = g.shape[0]
    cdef int nky = g.shape[1]
    cdef int nkz = g.shape[2]
    cdef int wkx = nkx // 2
    cdef int wky = nky // 2
    cdef int wkz = nkz // 2

    # The following need to be set to zeros rather than empty because the
    # boundary does not get reset.
    cdef np.ndarray[DTYPE_t, ndim=3] conv = np.zeros([nx, ny, nz], dtype=DTYPE)

    cdef unsigned int i, j, k, ii, jj, kk

    cdef int iimin, iimax, jjmin, jjmax, kkmin, kkmax

    cdef DTYPE_t top, bot, ker, val

    # release the GIL
    with nogil:

        # Now run the proper convolution
        for i in range(wkx, nx - wkx):
            for j in range(wky, ny - wky):
                for k in range(wkz, nz - wkz):
                    top = 0.
                    bot = 0.
                    for ii in range(i - wkx, i + wkx + 1):
                        for jj in range(j - wky, j + wky + 1):
                            for kk in range(k - wkz, k + wkz + 1):
                                val = f[ii, jj, kk]
                                ker = g[<unsigned int>(nkx - 1 - (wkx + ii - i)),
                                        <unsigned int>(nky - 1 - (wky + jj - j)),
                                        <unsigned int>(nkz - 1 - (wkz + kk - k))]
                                if not npy_isnan(val):
                                    top += val * ker
                                    bot += ker
                    if normalize_by_kernel:
                        if bot == 0:
                            conv[i, j, k] = f[i, j, k]
                        else:
                            conv[i, j, k] = top / bot
                    else:
                        conv[i, j, k] = top
    # GIL acquired again here
    return conv
