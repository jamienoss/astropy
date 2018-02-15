#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    unsigned nx;
    double * data;
} Array;

typedef struct {
    const unsigned nx;
    const double * data;
} ConstArray;

int comp_conv(Array * conv, const ConstArray * f, const ConstArray * g, const bool nan_interpolate);
void comp_conv_inline(Array * conv, const ConstArray * f, const ConstArray * g, const bool nan_interpolate);
int py_comp_conv(double * _conv, const double * _f, const unsigned nx, const unsigned ny,
        const double * _g, const unsigned nkx, const unsigned nky, const bool nan_interpolate);

void writeBin(const Array * array, const char * fname);
double timeDiff(clock_t start);

int py_comp_conv(double * _conv, const double * _f, const unsigned nx, const unsigned ny,
        const double * _g, const unsigned nkx, const unsigned nky, const bool nan_interpolate)
{
    Array conv = {.nx = nx, .data = _conv};
    
    ConstArray f = {.nx = nx, .data = _f};
    ConstArray g = {.nx = nkx, .data = _g};
//return 88;
    
   return  comp_conv(&conv, &f, &g, nan_interpolate);
    
}

int comp_conv(Array * conv, const ConstArray * f, const ConstArray * g, const bool nan_interpolate)
{
    if (nan_interpolate)
        comp_conv_inline(conv, f, g, true);
    else
        comp_conv_inline(conv, f, g, false);
    return 1;
}

void writeBin(const Array * array, const char * fname)
{
    printf("Writing data to file '%s'\n", fname);
    fflush(stdout);
    FILE * fp = fopen(fname, "wb");
    assert(fp);

    unsigned nx = array->nx;
    size_t count = (size_t)(nx*nx);
    assert(fwrite(array->data, sizeof(*array->data),count, fp) == count);
    assert(!fclose(fp));
}

inline double timeDiff(clock_t start)
{
    return (double)(clock() - start)/CLOCKS_PER_SEC;
}

int main(int argc, char *argv[])
{
    if (argc < 6)
    {
        printf("not enough params");
        return 1;
    }

    const unsigned nx = atoi(argv[1]);
    const unsigned ny = atoi(argv[1]);
    
    const unsigned nkx = atoi(argv[2]);
    const unsigned nky = atoi(argv[2]);

    const bool nan_interpolate = (bool)atoi(argv[3]);
    const bool normalize = (bool)atoi(argv[4]);

    const unsigned nThreads = atoi(argv[5]);
#ifdef _OPENMP
    printf("Using '%d' thread(s).\n", nThreads);
    omp_set_num_threads(nThreads);
#else
    printf("Using '%d' thread(s).\n", 1);
#endif
    
    clock_t start = clock();

    printf("nx = %d\n", nx);
    printf("nkx = %d\n", nkx);

    
    Array f,g,c;

    f.nx = nx;
    g.nx = nkx;
    c.nx = nx;

    f.data = malloc(nx*ny*sizeof(*f.data));
    assert(f.data);
    g.data = malloc(nkx*nky*sizeof(*g.data));
    assert(g.data);
    c.data = calloc(nx*ny, sizeof(*c.data));
    assert(c.data);

    for (unsigned i = 0; i < nx; ++i)
    {
        for (unsigned j = 0; j < nx; ++j)
        {
            f.data[i*ny + j] = rand() % 100;
        }
    }

    double k_sum = 0.;
    for (unsigned i = 0; i < nkx; ++i)
    {
        for (unsigned j = 0; j < nkx; ++j)
        {
            double val = 1. + rand() % 10;
            g.data[i*nky + j] = val;
            k_sum += val;
        }
    }
    printf("k sum = %lf\n", k_sum);
    printf("data setup time = %lf (s)\n", timeDiff(start));
    fflush(stdout);

    
    ConstArray constf = {.nx = f.nx, .data = f.data};
    ConstArray constg = {.nx = g.nx, .data = g.data};
    
    comp_conv(&c,&constf,&constg,nan_interpolate);
    
    if (normalize)
    {
        if (!nan_interpolate)
        {
            for (unsigned i = 0; i < nx; ++i)
            {
                for (unsigned j = 0; j < nx; ++j)
                {
                    c.data[i*ny + j] /= k_sum;
                }
            }
        }
    }
    else
    {
        if (nan_interpolate)
        {
            for (unsigned i = 0; i < nx; ++i)
            {
                for (unsigned j = 0; j < nx; ++j)
                {
                    c.data[i*ny + j] *= k_sum;
                }
            }
        }
    }

#if true
    start = clock();
    writeBin(&f, "image.dat");
    writeBin(&g, "kernel.dat");
    writeBin(&c, "conv.dat");  
    printf("data written out time = %lf (s)\n", timeDiff(start));
    fflush(stdout);
#endif

    free(c.data);
    free(f.data);
    free(g.data);
    return 0;
}

inline __attribute__((always_inline)) void comp_conv_inline(Array * conv, const ConstArray * f, const ConstArray * g, const bool nan_interpolate)
{
    //thread globals
    const unsigned nx = conv->nx;
    const unsigned ny = conv->nx;
    const unsigned nkx = g->nx;
    const unsigned nky = g->nx;
    const unsigned wkx = nkx / 2;
    const unsigned wky = nky / 2;
    const clock_t start = clock();

#ifdef _OPENMP
#pragma omp parallel shared(conv, f, g) // All other consts are declared shared by default
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
    //int iimin, iimax, jjmin, jjmax;
    
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
                    val = f->data[ii*ny + jj]; //[ii, jj];
                    ker = g->data[ker_i*nky + ker_j]; // [ker_i, ker_j];
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
                    conv->data[i*ny + j]  = f->data[i*ny + j] ;
                else
                    conv->data[i*ny + j]  = top / bot;
            }
            else
                conv->data[i*ny + j] = top;
        }
    }
#ifdef _OPENMP
    }//end parallel scope
#endif
    printf("core comp time = %lf (s)\n", timeDiff(start));
    fflush(stdout);
}
