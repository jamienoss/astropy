#include <stdbool.h>

bool openmp_enabled(void)
{
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}
