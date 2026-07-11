#include "libhmsbeagle/beagle.h"

#include <cstdio>

int main()
{
    BeagleInstanceDetails details;
    const int instance = beagleCreateInstance(
        3, 2, 3, 4, 4, 1, 4, 1, 0, NULL, 0,
        BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PRECISION_DOUBLE |
            BEAGLE_FLAG_THREADING_NONE,
        BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PRECISION_DOUBLE, &details);

    if (instance < 0) {
        std::fprintf(stderr, "CPU instance creation failed with BEAGLE status %d\n", instance);
        return 1;
    }
    if ((details.flags & BEAGLE_FLAG_PROCESSOR_CPU) == 0) {
        std::fprintf(stderr, "BEAGLE selected a non-CPU resource\n");
        beagleFinalizeInstance(instance);
        return 1;
    }
    if (beagleFinalizeInstance(instance) != BEAGLE_SUCCESS) {
        std::fprintf(stderr, "BEAGLE failed to finalize the CPU instance\n");
        return 1;
    }
    return 0;
}
