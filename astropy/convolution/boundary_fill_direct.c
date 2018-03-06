// Licensed under a 3-clause BSD style license - see LICENSE.rst

/*------------------------------WARNING!------------------------------
 * The C functions below are NOT designed to be called externally to
 * the Python function astropy/astropy/convolution/convolve.py.
 * They do NOT include any of the required correct usage checking.
 *------------------------------WARNING!------------------------------
 */

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// 1D
void convolve1d_boundary_fill_c(double * const result,
        const double * const f, const size_t nx,
        const double * const g, const size_t nkx,
        const double fill_value, const bool skip_fill,
        const bool nan_interpolate,
        const unsigned n_threads);
inline __attribute__((always_inline)) void convolve1d_boundary_fill(double * const result,
        const double * const f, const size_t nx,
        const double * const g, const size_t nkx,
        const double fill_value, const bool skip_fill,
        const bool nan_interpolate,
        const unsigned n_threads);
// 2D
void convolve2d_boundary_fill_c(double * const result,
        const double * const f, const size_t nx, const size_t ny,
        const double * const g, const size_t nkx, const size_t nky,
        const double fill_value, const bool skip_fill,
        const bool nan_interpolate,
        const unsigned n_threads);
inline __attribute__((always_inline))  void convolve2d_boundary_fill(double * const result,
        const double * const f, const size_t nx, const size_t ny,
        const double * const g, const size_t nkx, const size_t nky,
        const double fill_value, const bool skip_fill,
        const bool nan_interpolate,
        const unsigned n_threads);
// 3D
void convolve3d_boundary_fill_c(double * const result,
        const double * const f, const size_t nx, const size_t ny, const size_t nz,
        const double * const g, const size_t nkx, const size_t nky, const size_t nkz,
        const double fill_value, const bool skip_fill,
        const bool nan_interpolate,
        const unsigned n_threads);
inline __attribute__((always_inline)) void convolve3d_boundary_fill(double * const result,
        const double * const f, const size_t nx, const size_t ny, const size_t nz,
        const double * const g, const size_t nkx, const size_t nky, const size_t nkz,
        const double fill_value, const bool skip_fill,
        const bool nan_interpolate,
        const unsigned n_threads);

/*-------------------------PERFORMANCE NOTES--------------------------------
 * The function wrappers below are designed to take advantage of the following:
 * The preprocessor will inline compute_convolution(), effectively
 * expanding the two logical branches, replacing nan_interpolate
 * for their literal equivalents. The corresponding conditionals
 * within compute_convolution() will then be optimized away, this
 * being the goal - removing the unnecessary conditionals from
 * the loops without duplicating code.
 */

void convolve1d_boundary_fill_c(double * const result,
        const double * const f, const size_t nx,
        const double * const g, const size_t nkx,
        const double fill_value, const bool skip_fill,
        const bool nan_interpolate,
        const unsigned n_threads)
{
    if (!result || !f || !g)
        return;

    if (nan_interpolate)
        convolve1d_boundary_fill(result, f, nx, g, nkx, fill_value, false, true, n_threads);
    else
    {
        if (skip_fill)
            convolve1d_boundary_fill(result, f, nx, g, nkx, fill_value, true, false, n_threads);
        else
            convolve1d_boundary_fill(result, f, nx, g, nkx, fill_value, false, false, n_threads);
    }
}

void convolve2d_boundary_fill_c(double * const result,
        const double * const f, const size_t nx, const size_t ny,
        const double * const g, const size_t nkx, const size_t nky,
        const double fill_value, const bool skip_fill,
        const bool nan_interpolate,
        const unsigned n_threads)

{
    if (!result || !f || !g)
        return;

    if (nan_interpolate)
        convolve2d_boundary_fill(result, f, nx, ny, g, nkx, nky, fill_value, false, true, n_threads);
    else
    {
        if (skip_fill)
            convolve2d_boundary_fill(result, f, nx, ny, g, nkx, nky, fill_value, true, false, n_threads);
        else
            convolve2d_boundary_fill(result, f, nx, ny, g, nkx, nky, fill_value, false, false, n_threads);
    }
}

void convolve3d_boundary_fill_c(double * const result,
        const double * const f, const size_t nx, const size_t ny, const size_t nz,
        const double * const g, const size_t nkx, const size_t nky, const size_t nkz,
        const double fill_value, const bool skip_fill,
        const bool nan_interpolate,
        const unsigned n_threads)
{
    if (!result || !f || !g)
        return;

    if (nan_interpolate)
        convolve3d_boundary_fill(result, f, nx, ny, nz, g, nkx, nky, nkz, fill_value, false, true, n_threads);
    else
    {
        if (skip_fill)
            convolve3d_boundary_fill(result, f, nx, ny, nz, g, nkx, nky, nkz, fill_value, true, false, n_threads);
        else
            convolve3d_boundary_fill(result, f, nx, ny, nz, g, nkx, nky, nkz, fill_value, false, false, n_threads);
    }
}

/*-------------------------PERFORMANCE NOTES--------------------------------
 * Only 2 scenarios exist for boundary='fill':
 * (image dims: nx, ny, nz; kernel dims: nkx, nky, nkz).
 * (ii, jj, kk, refer to image locations but have iteration length of kernel dims).
 * 1) kernel over-steps image.
 *   I.e. (ii < 0 || ii >= nx) || (jj < 0 || jj >= ny) || (kk < 0 || kk >= nz) == true
 * 2) kernel completely within image.
 *  I.e. (ii < 0 || ii >= nx) || (jj < 0 || jj >= ny) || (kk < 0 || kk >= nz) == false
 * Given that the kernel is normally much smaller than the image, (2)
 * occupies the majority of the iteration space.
 * For (2) the check is exhaustive (no short-circuit), all conditions MUST be checked.
 * It is not possible to completely optimize for both (1) & (2). Better to make
 * majority case more efficient.
 * Original Cython had the above conditional within the inner most loop.
 * For (2) that's, at most, 6*nkx*nky*nkz checks. 3*nkx*nky (2D case).
 * 1st tier optimization is to split these out, to un-nest them, i.e. hoist
 * ii condition to ii, kk to kk etc. Worse case -> c*(nkx + nky + nkz), c = 2.
 * For (2) `val` (c.f. code) can only be assigned a value within the inner
 * most loop as all indices are needed. Therefore, with the condition hoisted
 * the result must now be propagated to the next nest level. Use a bool flag
 * for this.
 * I.e. (2D case)
 * ```
 * ...
 *  for (int ii = i_minus_wkx; ii < (int)i_plus_wkx_plus_1; ++ii)
    {
        ...
        bool fill_value_used = false;
        if (ii < 0 || ii >= (int)nx)
        {
            val = fill_value;
            fill_value_used = true;
        }
        ...
        for (int jj = j_minus_wky; jj < (int)j_plus_wky_plus_1; ++jj)
        {
            ...
            if (!fill_value_used)
            {
                if (jj < 0 || jj >= (int)ny)
                    val = fill_value;
                else
                    val = f[(unsigned)ii*ny + jj];
            }
            ...
        }
    ...
    }
    ...
````
 * This yields a perf boost for (1) when a true condition in an outer loop
 * short-circuiting the next loop checks, bar the introduced gateway condition
 * on `fill_value_used`. For (2) it makes the worse case 2*nkx + 3*(nky + nkz)
 * which is still significantly better than 6*nkx*nky*nkz.
 *
 * Further opt:
 * 1) The default fill_value = 0. => sub-nested loops can be
 * skipped entirely. val = fill_value = 0 => val*ker = 0, top += val*ker = top.
 * This is NOT possible for nan interpolation.
 * 2) Scrap all of the above and just pad the image array by, e.g. wkx = nkx/2,
 * with fill_value and remove all conditions. (mem issues for large kernels).
 */

// 1D
inline __attribute__((always_inline)) void convolve1d_boundary_fill(double * const result,
        const double * const f, const size_t _nx,
        const double * const g, const size_t _nkx,
        const double _fill_value, const bool _skip_fill,
        const bool _nan_interpolate,
        const unsigned n_threads)
{
    if (!result || !f || !g)
        return;

#ifdef _OPENMP
    omp_set_num_threads(n_threads); // Set number of threads to use
#pragma omp parallel
    { // Code within this block is threaded
#endif

    // Copy these to thread locals to allow compiler to optimize (hoist/loads licm)
    // when threaded. Without these, compile time constant conditionals may
    // not be optimized away.
    const size_t nx = _nx;
    const size_t nkx = _nkx;
    const double fill_value = _fill_value;
    const bool skip_fill = _skip_fill;
    const bool nan_interpolate = _nan_interpolate;

    // Thread locals
    const unsigned wkx = nkx / 2;
    const unsigned nkx_minus_1 = nkx-1;
    int wkx_minus_i;
    unsigned ker_i;
    int i_minus_wkx;
    const unsigned wkx_plus_1 = wkx + 1;
    unsigned i_plus_wkx_plus_1;
    int nkx_minus_1_minus_wkx_plus_i;
    double top, bot=0., ker, val;

    const unsigned half_all = wkx + nx/2;
    const unsigned half_nx = nx/2;

    const double fill[1] = {fill_value};
    const double * lookup[2] = {fill, f};


#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (unsigned i = 0; i < nx; ++i)
    {
        wkx_minus_i = wkx - i; // wkx - 1
        i_minus_wkx = i - wkx; //i - wkx
        i_plus_wkx_plus_1 = i + wkx_plus_1; // i + wkx + 1
        nkx_minus_1_minus_wkx_plus_i = nkx_minus_1 - wkx_minus_i; // nkx - 1 - (wkx - i)

        top = 0.;
        if (nan_interpolate) // compile time constant
            bot = 0.;

        for (int ii = i_minus_wkx; ii < (int)i_plus_wkx_plus_1; ++ii)
        {
            unsigned iii = ii + wkx;
            unsigned sw = !(abs( (int)(iii+!(iii/half_all)-half_all))/half_nx);

            const double * array = lookup[sw];
            val = array[ii*sw];


            /*if (sw)//ii < 0 || ii >= (int)nx)
            {
                if (skip_fill) // compile time constant
                    continue;
                else
                    val = fill_value;
            }
            else
                val = f[(unsigned)ii];
*/
            ker_i = nkx_minus_1_minus_wkx_plus_i - ii; // nkx - 1 - (wkx + ii - i)
            ker = g[ker_i];
            if (nan_interpolate) // compile time constant
            {
                if (!isnan(val))
                {
                    top += val * ker;
                    bot += ker;
                }
            }
            else
                top += val * ker;
        }

        if (nan_interpolate) // compile time constant
        {
            if (bot == 0) // This should prob be np.isclose(kernel_sum, 0, atol=normalization_zero_tol)
                result[i]  = f[i];
            else
                result[i]  = top / bot;
        }
        else
            result[i] = top;
    }
#ifdef _OPENMP
    }//end parallel scope
#endif
}


// 2D
inline __attribute__((always_inline)) void convolve2d_boundary_fill(double * const result,
        const double * const f, const size_t _nx, const size_t _ny,
        const double * const g, const size_t _nkx, const size_t _nky,
        const double _fill_value, const bool _skip_fill,
        const bool _nan_interpolate,
        const unsigned n_threads)
{
    if (!result || !f || !g)
        return;

#ifdef _OPENMP
    omp_set_num_threads(n_threads); // Set number of threads to use
#pragma omp parallel
    { // Code within this block is threaded
#endif

    // Copy these to thread locals to allow compiler to optimize (hoist/loads licm)
    // when threaded. Without these, compile time constant conditionals may
    // not be optimized away.
    const size_t nx = _nx, ny = _ny;
    const size_t nkx = _nkx, nky = _nky;
    const double fill_value = _fill_value;
    const bool skip_fill = _skip_fill;
    const bool nan_interpolate = _nan_interpolate;

    // Thread locals
    const unsigned wkx = nkx / 2;
    const unsigned wky = nky / 2;
    const unsigned nkx_minus_1 = nkx-1, nky_minus_1 = nky-1;
    int wkx_minus_i, wky_minus_j;
    unsigned ker_i, ker_j;
    int i_minus_wkx, j_minus_wky;
    const unsigned wkx_plus_1 = wkx + 1;
    const unsigned wky_plus_1 = wky + 1;
    unsigned i_plus_wkx_plus_1, j_plus_wky_plus_1;
    int nkx_minus_1_minus_wkx_plus_i, nky_minus_1_minus_wky_plus_j;

    double top, bot=0., ker, val;

    const unsigned half_all = wkx + nx/2;
        const unsigned half_nx = nx/2;

        const double fill[1] = {fill_value};
        const double * lookup[2] = {fill, f};
unsigned iii, jjj;
bool sw_i, sw;
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (unsigned i = 0; i < nx; ++i)
    {
        wkx_minus_i = wkx - i; // wkx - 1
        i_minus_wkx = i - wkx; //i - wkx
        i_plus_wkx_plus_1 = i + wkx_plus_1; // i + wkx + 1
        nkx_minus_1_minus_wkx_plus_i = nkx_minus_1 - wkx_minus_i; // nkx - 1 - (wkx - i)
    
        for (unsigned j = 0; j < ny; ++j)
        {
            wky_minus_j = wkx - j; // wky - j
            j_minus_wky = j - wky; // j - wky
            j_plus_wky_plus_1 = j + wky_plus_1; // j + wky + 1
            nky_minus_1_minus_wky_plus_j = nky_minus_1 - wky_minus_j; // nky - 1 - (wky - i)

            top = 0.;
            if (nan_interpolate) // compile time constant
                bot = 0.;
            for (int ii = i_minus_wkx; ii < (int)i_plus_wkx_plus_1; ++ii)
            {
                iii = ii + wkx;
                sw_i = (abs( (int)(iii+!(iii/half_all)-half_all))/half_nx);

                /*bool fill_value_used = false;
                if (ii < 0 || ii >= (int)nx)
                {
                    if (skip_fill) // compile time constant
                        continue;
                    else
                    {
                        val = fill_value;
                        fill_value_used = true;
                    }
                }*/
                ker_i = nkx_minus_1_minus_wkx_plus_i - ii; // nkx - 1 - (wkx + ii - i)
                for (int jj = j_minus_wky; jj < (int)j_plus_wky_plus_1; ++jj)
                {

                  /*  if (!fill_value_used)
                    {
                        if (jj < 0 || jj >= (int)ny)
                        {
                            if (skip_fill) // compile time constant
                                continue;
                            else
                                val = fill_value;
                        }
                        else
                            val = f[(unsigned)ii*ny + jj];
                    }*/
                   jjj = jj + wky;
                   sw = !(sw_i||((abs( (int)(jjj+!(jjj/half_all)-half_all))/half_nx)));
                   val = (lookup[sw])[(ii*ny + jj)*sw];

                    ker_j = nky_minus_1_minus_wky_plus_j - jj; // nky - 1 - (wky + jj - j)
                    ker = g[ker_i*nky + ker_j]; // [ker_i, ker_j];
                    if (nan_interpolate) // compile time constant
                    {
                        if (!isnan(val))
                        {
                            top += val * ker;
                            bot += ker;
                        }
                    }
                    else
                        top += val * ker;
                }
            }
            if (nan_interpolate) // compile time constant
            {
                if (bot == 0) // This should prob be np.isclose(kernel_sum, 0, atol=normalization_zero_tol)
                    result[i * ny + j] = f[i * ny + j];
                else
                    result[i * ny + j] = top / bot;
            }
            else
            result[i * ny + j] = top;
        }
    }
#ifdef _OPENMP
    }//end parallel scope
#endif
}

// 3D
inline __attribute__((always_inline)) void convolve3d_boundary_fill(double * const result,
        const double * const f, const size_t _nx, const size_t _ny, const size_t _nz,
        const double * const g, const size_t _nkx, const size_t _nky, const size_t _nkz,
        const double _fill_value, const bool _skip_fill,
        const bool _nan_interpolate,
        const unsigned n_threads)
{
    if (!result || !f || !g)
        return;

#ifdef _OPENMP
    omp_set_num_threads(n_threads); // Set number of threads to use
#pragma omp parallel
    { // Code within this block is threaded
#endif

    // Copy these to thread locals to allow compiler to optimize (hoist/loads licm)
    // when threaded. Without these, compile time constant conditionals may
    // not be optimized away.
    const size_t nx = _nx, ny = _ny, nz = _nz;
    const size_t nkx = _nkx, nky = _nky, nkz = _nkz;
    const double fill_value = _fill_value;
    const bool skip_fill = _skip_fill;
    const bool nan_interpolate = _nan_interpolate;

    // Thread locals
    const unsigned wkx = nkx / 2;
    const unsigned wky = nky / 2;
    const unsigned wkz = nkz / 2;
    const unsigned nkx_minus_1 = nkx-1, nky_minus_1 = nky-1, nkz_minus_1 = nkz-1;
    int wkx_minus_i, wky_minus_j, wkz_minus_k;
    unsigned ker_i, ker_j, ker_k;
    int i_minus_wkx, j_minus_wky, k_minus_wkz;
    const unsigned wkx_plus_1 = wkx + 1;
    const unsigned wky_plus_1 = wky + 1;
    const unsigned wkz_plus_1 = wkz + 1;
    unsigned i_plus_wkx_plus_1, j_plus_wky_plus_1, k_plus_wkz_plus_1;
    int nkx_minus_1_minus_wkx_plus_i, nky_minus_1_minus_wky_plus_j, nkz_minus_1_minus_wkz_plus_k;

    double top, bot=0., ker;
    double val = 0.; // Assign value to prevent gcc "-Wmaybe-uninitialized" false positive.

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (unsigned i = 0; i < nx; ++i)
    {
        wkx_minus_i = wkx - i; // wkx - 1
        i_minus_wkx = i - wkx; //i - wkx
        i_plus_wkx_plus_1 = i + wkx_plus_1; // i + wkx + 1
        nkx_minus_1_minus_wkx_plus_i = nkx_minus_1 - wkx_minus_i; // nkx - 1 - (wkx - i)

        for (unsigned j = 0; j < ny; ++j)
        {
            wky_minus_j = wkx - j; // wky - j
            j_minus_wky = j - wky; // j - wky
            j_plus_wky_plus_1 = j + wky_plus_1; // j + wky + 1
            nky_minus_1_minus_wky_plus_j = nky_minus_1 - wky_minus_j; // nky - 1 - (wky - i)

            for (unsigned k = 0; k < nz; ++k)
            {
                wkz_minus_k = wkz - k; // wkz - k
                k_minus_wkz = k - wkz; // k - wkz
                k_plus_wkz_plus_1 = k + wkz_plus_1; // k + wkz + 1
                nkz_minus_1_minus_wkz_plus_k = nkz_minus_1 - wkz_minus_k; // nkz - 1 - (wkz - i)

                top = 0.;
                if (nan_interpolate) // compile time constant
                    bot = 0.;
                for (int ii = i_minus_wkx; ii < (int)i_plus_wkx_plus_1; ++ii)
                {
                    bool fill_value_used_ii = false;
                    if (ii < 0 || ii >= (int)nx)
                    {
                        if (skip_fill) // compile time constant
                            continue;
                        else
                        {
                            val = fill_value;
                            fill_value_used_ii = true;
                        }
                    }
                    ker_i = nkx_minus_1_minus_wkx_plus_i - ii; // nkx - 1 - (wkx + ii - i)
                    for (int jj = j_minus_wky; jj < (int)j_plus_wky_plus_1; ++jj)
                    {
                        bool fill_value_used_jj = false;
                        if (!fill_value_used_ii)
                        {
                            if (jj < 0 || jj >= (int)ny)
                            {
                                if (skip_fill) // compile time constant
                                    continue;
                                else
                                {
                                    val = fill_value;
                                    fill_value_used_jj = true;
                                }
                            }
                        }
                        ker_j = nky_minus_1_minus_wky_plus_j - jj; // nky - 1 - (wky + jj - j)
                        for (int kk = k_minus_wkz; kk < (int)k_plus_wkz_plus_1; ++kk)
                        {
                            ker_k = nkz_minus_1_minus_wkz_plus_k - kk; // nkz - 1 - (wkz + kk - k)
                            if (!fill_value_used_ii && !fill_value_used_jj)
                            {
                                if (kk < 0 || kk >= (int)nz)
                                {
                                    if (skip_fill) // compile time constant
                                        continue;
                                    else
                                        val = fill_value;
                                }
                                else
                                    val = f[((unsigned)ii*ny + (unsigned)jj)*nz + (unsigned)kk]; //[ii, jj, kk];
                            }
                            ker = g[(ker_i*nky + ker_j)*nkz + ker_k]; // [ker_i, ker_j, ker_k];
                            if (nan_interpolate) // compile time constant
                            {
                                if (!isnan(val))
                                {
                                    top += val * ker;
                                    bot += ker;
                                }
                            }
                            else
                                top += val * ker;
                        }
                    }
                }
                if (nan_interpolate) // compile time constant
                {
                    if (bot == 0) // This should prob be np.isclose(kernel_sum, 0, atol=normalization_zero_tol)
                        result[(i*ny + j)*nz + k]  = f[(i*ny + j)*nz + k] ;
                    else
                        result[(i*ny + j)*nz + k]  = top / bot;
                }
                else
                    result[(i*ny + j)*nz + k] = top;
            }
        }
    }
#ifdef _OPENMP
    }//end parallel scope
#endif
}
