#ifndef CONVOLVE_INCLUDE
#define CONVOLVE_INCLUDE

#if defined(_MSC_VER)

#define FORCE_INLINE  __forceinline
#define NEVER_INLINE  __declspec(noinline)

// Other compilers (including GCC & Clang)
#else

#define FORCE_INLINE inline __attribute__((always_inline))
#define NEVER_INLINE __attribute__((noinline))

#endif

// MSVC mandates singed integers for its OpenMP loops
#if defined(_MSC_VER)
typedef signed omp_unsigned;
#else
typedef unsigned omp_unsigned;
#endif

#endif
