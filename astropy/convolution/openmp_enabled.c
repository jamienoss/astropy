#include <stdbool.h>

bool is_openmp_enabled(void)
{
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}
