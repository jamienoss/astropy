#include <math.h>
#include <stdbool.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void convolve2d_boundary_none_c(double * const result,
		const double * const f, const unsigned nx, const unsigned ny,
		const double * const g, const unsigned nkx, const unsigned nky,
		const bool nan_interpolate);
void compute_convolution(double * const result,
		const double * const f, const unsigned nx, const unsigned ny,
		const double * const g, const unsigned nkx, const unsigned nky,
		const bool nan_interpolate);

void convolve2d_boundary_none_c(double * const result,
		const double * const f, const unsigned nx, const unsigned ny,
		const double * const g, const unsigned nkx, const unsigned nky,
		const bool nan_interpolate)
{
	if (!result || !f || !g)
		return;

    if (nan_interpolate)
    	compute_convolution(result, f, nx, ny, g, nkx, nky, true);
    else
    	compute_convolution(result, f, nx, ny, g, nkx, nky, false);
}

inline __attribute__((always_inline)) void compute_convolution(double * const result,
		const double * const f, const unsigned nx, const unsigned ny,
		const double * const g, const unsigned nkx, const unsigned nky,
		const bool nan_interpolate)
{
    //thread globals
    const unsigned wkx = nkx / 2;
    const unsigned wky = nky / 2;

#ifdef _OPENMP
#pragma omp parallel shared(result, f, g) // All other consts are declared shared by default
    {
#endif
    
    //thread locals
    const unsigned int nkx_minus_1 = nkx-1, nky_minus_1 = nky-1;
    unsigned int wkx_minus_i, wky_minus_j;
    unsigned int ker_i, ker_j;
    const unsigned int nx_minus_wkx = nx - wkx;
    const unsigned int ny_minus_wky = ny - wky;
    unsigned int i_minus_wkx;
    const unsigned wkx_plus_1 = wkx + 1;
    unsigned int j_minus_wky;
    const unsigned wky_plus_1 = wky + 1;
    unsigned int i_plus_wkx_plus_1, j_plus_wky_plus_1;
    unsigned int nkx_minus_1_minus_wkx_plus_i, nky_minus_1_minus_wky_plus_j;
    
    double top, bot=0., ker, val;
    
    
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (unsigned i = wkx; i < nx_minus_wkx; ++i)
    {
        wkx_minus_i = wkx - i; // wkx - 1
        i_minus_wkx = i - wkx; //i - wkx
        i_plus_wkx_plus_1 = i + wkx_plus_1; // i + wkx + 1
        nkx_minus_1_minus_wkx_plus_i = nkx_minus_1 - wkx_minus_i; // nkx - 1 - (wkx - i)
    
        for (unsigned j = wky; j < ny_minus_wky; ++j)
        {
            wky_minus_j = wkx - j; // wky - j
            j_minus_wky = j - wky; // j - wky
            j_plus_wky_plus_1 = j + wky_plus_1; // j + wky + 1
            nky_minus_1_minus_wky_plus_j = nky_minus_1 - wky_minus_j; // nky - 1 - (wky - i)
            top = 0.;
            if (nan_interpolate)
                bot = 0.;
            for (unsigned ii = i_minus_wkx; ii < i_plus_wkx_plus_1; ++ii)
            {
                ker_i = nkx_minus_1_minus_wkx_plus_i - ii; // nkx - 1 - (wkx + ii - i)
//#pragma clang loop vectorize(enable) // takes twice as long with this set, values are correct though.
                for (unsigned jj = j_minus_wky; jj < j_plus_wky_plus_1; ++jj)
                {
                    ker_j = nky_minus_1_minus_wky_plus_j - jj; // nky - 1 - (wky + jj - j)
                    val = f[ii*ny + jj]; //[ii, jj];
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
