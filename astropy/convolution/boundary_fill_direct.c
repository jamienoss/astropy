// Licensed under a 3-clause BSD style license - see LICENSE.rst

/*------------------------------WARNING!------------------------------
 * The C functions below are NOT designed to be called externally to
 * the Python function astropy/astropy/convolution/convolve.py.
 * They do NOT include any of the required correct usage checking.
 *------------------------------WARNING!------------------------------
 */

#include <math.h>
#include <stdbool.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// 1D
void convolve1d_boundary_fill_c(double * const result,
		const double * const f, const size_t nx,
		const double * const g, const size_t nkx,
		const double fill_value,
		const bool nan_interpolate,
		const unsigned n_threads);
void convolve1d_boundary_fill(double * const result,
		const double * const f, const size_t nx,
		const double * const g, const size_t nkx,
		const double fill_value,
		const bool nan_interpolate,
		const unsigned n_threads);
// 2D
void convolve2d_boundary_fill_c(double * const result,
		const double * const f, const size_t nx, const size_t ny,
		const double * const g, const size_t nkx, const size_t nky,
		const double fill_value,
		const bool nan_interpolate,
		const unsigned n_threads);
void convolve2d_boundary_fill(double * const result,
		const double * const f, const size_t nx, const size_t ny,
		const double * const g, const size_t nkx, const size_t nky,
		const double fill_value,
		const bool nan_interpolate,
		const unsigned n_threads);
// 3D
void convolve3d_boundary_fill_c(double * const result,
		const double * const f, const size_t nx, const size_t ny, const size_t nz,
		const double * const g, const size_t nkx, const size_t nky, const size_t nkz,
		const double fill_value,
		const bool nan_interpolate,
		const unsigned n_threads);
void convolve3d_boundary_fill(double * const result,
		const double * const f, const size_t nx, const size_t ny, const size_t nz,
		const double * const g, const size_t nkx, const size_t nky, const size_t nkz,
		const double fill_value,
		const bool nan_interpolate,
		const unsigned n_threads);

// The function wrappers below are designed to take advantage of the following:
// The preprocessor will inline compute_convolution(), effectively
// expanding the two logical branches, replacing nan_interpolate
// for their literal equivalents. The corresponding conditionals
// within compute_convolution() will then be optimized away, this
// being the goal - removing the unnecessary conditionals from
// the loops without duplicating code.

void convolve1d_boundary_fill_c(double * const result,
		const double * const f, const size_t nx,
		const double * const g, const size_t nkx,
		const double fill_value,
		const bool nan_interpolate,
		const unsigned n_threads)
{
	if (!result || !f || !g)
		return;

    if (nan_interpolate)
    	convolve1d_boundary_fill(result, f, nx, g, nkx, fill_value, true, n_threads);
    else
    	convolve1d_boundary_fill(result, f, nx, g, nkx, fill_value, false, n_threads);
}

void convolve2d_boundary_fill_c(double * const result,
		const double * const f, const size_t nx, const size_t ny,
		const double * const g, const size_t nkx, const size_t nky,
		const double fill_value,
		const bool nan_interpolate,
		const unsigned n_threads)
{
	if (!result || !f || !g)
		return;

    if (nan_interpolate)
    	convolve2d_boundary_fill(result, f, nx, ny, g, nkx, nky, fill_value, true, n_threads);
    else
    	convolve2d_boundary_fill(result, f, nx, ny, g, nkx, nky, fill_value, false, n_threads);
}

void convolve3d_boundary_fill_c(double * const result,
		const double * const f, const size_t nx, const size_t ny, const size_t nz,
		const double * const g, const size_t nkx, const size_t nky, const size_t nkz,
		const double fill_value,
		const bool nan_interpolate,
		const unsigned n_threads)
{
	if (!result || !f || !g)
		return;

    if (nan_interpolate)
    	convolve3d_boundary_fill(result, f, nx, ny, nz, g, nkx, nky, nkz, fill_value, true, n_threads);
    else
    	convolve3d_boundary_fill(result, f, nx, ny, nz, g, nkx, nky, nkz, fill_value, false, n_threads);
}

// 1D
inline __attribute__((always_inline)) void convolve1d_boundary_fill(double * const result,
		const double * const f, const size_t nx,
		const double * const g, const size_t nkx,
		const double fill_value,
		const bool nan_interpolate,
		const unsigned n_threads)
{
	if (!result || !f || !g)
		return;

    // Thread globals
    const unsigned wkx = nkx / 2;

#ifdef _OPENMP
    omp_set_num_threads(n_threads); // Set number of threads to use
#pragma omp parallel shared(result, f, g) // All other consts are declared shared by default
    { // Code within this block is threaded
#endif

    // Thread locals
    const unsigned nkx_minus_1 = nkx-1;
    int wkx_minus_i;
    unsigned ker_i;
    int i_minus_wkx;
    const unsigned wkx_plus_1 = wkx + 1;
    unsigned i_plus_wkx_plus_1;
    int nkx_minus_1_minus_wkx_plus_i;
    double top, bot=0., ker, val;

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
		if (nan_interpolate)
			bot = 0.;

		for (int ii = i_minus_wkx; ii < (int)i_plus_wkx_plus_1; ++ii)
		{
			if (ii < 0 || ii >= (int)nx)
				val = fill_value;
			else
				val = f[(unsigned)ii];

			ker_i = nkx_minus_1_minus_wkx_plus_i - ii; // nkx - 1 - (wkx + ii - i)
			ker = g[ker_i];
			if (nan_interpolate)
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

		if (nan_interpolate)
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
		const double * const f, const size_t nx, const size_t ny,
		const double * const g, const size_t nkx, const size_t nky,
		const double fill_value,
		const bool nan_interpolate,
		const unsigned n_threads)
{
	if (!result || !f || !g)
		return;

    // Thread globals
    const unsigned wkx = nkx / 2;
    const unsigned wky = nky / 2;

#ifdef _OPENMP
    omp_set_num_threads(n_threads); // Set number of threads to use
#pragma omp parallel shared(result, f, g) // All other consts are declared shared by default
    { // Code within this block is threaded
#endif
    
    // Thread locals
    const unsigned nkx_minus_1 = nkx-1, nky_minus_1 = nky-1;
    int wkx_minus_i, wky_minus_j;
    unsigned ker_i, ker_j;
    int i_minus_wkx, j_minus_wky;
    const unsigned wkx_plus_1 = wkx + 1;
    const unsigned wky_plus_1 = wky + 1;
    unsigned i_plus_wkx_plus_1, j_plus_wky_plus_1;
    int nkx_minus_1_minus_wkx_plus_i, nky_minus_1_minus_wky_plus_j;
    
    double top, bot=0., ker, val;
    
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
            if (nan_interpolate)
                bot = 0.;
            for (int ii = i_minus_wkx; ii < (int)i_plus_wkx_plus_1; ++ii)
            {
            	bool fill_value_used = false;
            	if (ii < 0 || ii >= (int)nx)
            	{
            		val = fill_value;
            		fill_value_used = true;
            	}
                ker_i = nkx_minus_1_minus_wkx_plus_i - ii; // nkx - 1 - (wkx + ii - i)
                for (int jj = j_minus_wky; jj < (int)j_plus_wky_plus_1; ++jj)
                {
                	if (!fill_value_used && (jj < 0 || jj >= (int)ny))
						val = fill_value;
					else
						val = f[(unsigned)ii*ny + jj];
                    ker_j = nky_minus_1_minus_wky_plus_j - jj; // nky - 1 - (wky + jj - j)
                    ker = g[ker_i*nky + ker_j]; // [ker_i, ker_j];
                    if (nan_interpolate)
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
            if (nan_interpolate)
            {
                if (bot == 0) // This should prob be np.isclose(kernel_sum, 0, atol=normalization_zero_tol)
                	result[i*ny + j]  = f[i*ny + j] ;
                else
                	result[i*ny + j]  = top / bot;
            }
            else
            	result[i*ny + j] = top;
        }
    }
#ifdef _OPENMP
    }//end parallel scope
#endif
}

// 3D
inline __attribute__((always_inline)) void convolve3d_boundary_fill(double * const result,
		const double * const f, const size_t nx, const size_t ny, const size_t nz,
		const double * const g, const size_t nkx, const size_t nky, const size_t nkz,
		const double fill_value,
		const bool nan_interpolate,
		const unsigned n_threads)
{
	if (!result || !f || !g)
		return;

    // Thread globals
    const unsigned wkx = nkx / 2;
    const unsigned wky = nky / 2;
    const unsigned wkz = nkz / 2;

#ifdef _OPENMP
    omp_set_num_threads(n_threads); // Set number of threads to use
#pragma omp parallel shared(result, f, g) // All other consts are declared shared by default
    { // Code within this block is threaded
#endif

    // Thread locals
    const unsigned nkx_minus_1 = nkx-1, nky_minus_1 = nky-1, nkz_minus_1 = nkz-1;
    int wkx_minus_i, wky_minus_j, wkz_minus_k;
    unsigned ker_i, ker_j, ker_k;
    int i_minus_wkx, j_minus_wky, k_minus_wkz;
    const unsigned wkx_plus_1 = wkx + 1;
    const unsigned wky_plus_1 = wky + 1;
    const unsigned wkz_plus_1 = wkz + 1;
    unsigned i_plus_wkx_plus_1, j_plus_wky_plus_1, k_plus_wkz_plus_1;
    int nkx_minus_1_minus_wkx_plus_i, nky_minus_1_minus_wky_plus_j, nkz_minus_1_minus_wkz_plus_k;

    double top, bot=0., ker, val;

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
				if (nan_interpolate)
					bot = 0.;
				for (int ii = i_minus_wkx; ii < (int)i_plus_wkx_plus_1; ++ii)
				{
					bool fill_value_used = false;
					if (ii < 0 || ii >= (int)nx)
					{
						val = fill_value;
						fill_value_used = true;
					}
					ker_i = nkx_minus_1_minus_wkx_plus_i - ii; // nkx - 1 - (wkx + ii - i)
					for (int jj = j_minus_wky; jj < (int)j_plus_wky_plus_1; ++jj)
					{

						ker_j = nky_minus_1_minus_wky_plus_j - jj; // nky - 1 - (wky + jj - j)
						for (int kk = k_minus_wkz; kk < (int)k_plus_wkz_plus_1; ++kk)
						{
							ker_k = nkz_minus_1_minus_wkz_plus_k - kk; // nkz - 1 - (wkz + kk - k)
							if (!fill_value_used && (kk < 0 || kk >= nz))
								val = fill_value;
							else
								val = f[((unsigned)ii*ny + (unsigned)jj)*nz + (unsigned)kk]; //[ii, jj, kk];
							ker = g[(ker_i*nky + ker_j)*nkz + ker_k]; // [ker_i, ker_j, ker_k];
							if (nan_interpolate)
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
				if (nan_interpolate)
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
