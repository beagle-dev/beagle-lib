/*
 * Copyright 2026 Phylogenetic Likelihood Working Group
 * This file is part of BEAGLE.
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * The flags at bit 31 and above are 64-bit, from C, on every platform (long is 32 bits on Windows):
 *   - at compile time, each of them is 1LL << n and positive, no two flags share a bit, and every flag field of the
 *     API structs is 8 bytes;
 *   - at run time, the CPU resource's flags from bit 31 up are exactly PREORDER_TRANSPOSE_AUTO (bit 31), _MANUAL
 *     (bit 34) and SPECTRAL_REPRESENTATION (bit 40) (its lower bits depend on the build), with no undefined bit set
 *     (no sign extension); a CPU instance that requires PREORDER_TRANSPOSE_MANUAL and SPECTRAL_REPRESENTATION
 *     (bit 40) is a spectral implementation (so both bits reached the selection; a CPU instance does not report
 *     them in its flags), with nothing above bit 46 and no undefined bit in its flags; and one that requires
 *     PREORDER_TRANSPOSE_AUTO is created (a sign-extended bit 31 would require bits 31-63 and match no resource);
 *   - beagleGetBenchmarkedResourceList on the CPU resource, with the spectral requirement and AUTO preferred,
 *     benchmarks a spectral implementation, returns the resource's supportFlags and requiredFlags unchanged, and
 *     benchedFlags with nothing above bit 46 and no undefined bit.
 * Being C, it also checks that beagle.h compiles as C. Exits non-zero on a failed check.
 */

#include <stdio.h>
#include <string.h>

#include "libhmsbeagle/beagle.h"

/* _Static_assert where the compiler has it (C11, GCC and clang in any mode), else a negative array size */
#if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L) || defined(__GNUC__) || defined(__clang__)
#define FLAGBITS_ASSERT(c, m) _Static_assert(c, m)
#else
#define FLAGBITS_CAT2(a, b) a##b
#define FLAGBITS_CAT(a, b) FLAGBITS_CAT2(a, b)
#define FLAGBITS_ASSERT(c, m) typedef char FLAGBITS_CAT(flagbits_assert_, __LINE__)[(c) ? 1 : -1]
#endif

#define HIGH_FLAG(name, n) \
    FLAGBITS_ASSERT(BEAGLE_FLAG_##name == (1LL << n) && BEAGLE_FLAG_##name > 0 && \
                    sizeof(BEAGLE_FLAG_##name) == 8, "BEAGLE_FLAG_" #name " is not the 64-bit 1LL << " #n);
HIGH_FLAG(PREORDER_TRANSPOSE_AUTO, 31)
HIGH_FLAG(VECTOR_TENSOR, 33)
HIGH_FLAG(PREORDER_TRANSPOSE_MANUAL, 34)
HIGH_FLAG(SPECTRAL_REPRESENTATION, 40)
HIGH_FLAG(FRAMEWORK_TINYGPU, 45)
HIGH_FLAG(PARALLELOPS_FUSED, 46)

/* every flag in beagle.h */
#define ALL_FLAGS(X) \
    X(PRECISION_SINGLE) X(PRECISION_DOUBLE) X(COMPUTATION_SYNCH) X(COMPUTATION_ASYNCH) X(EIGEN_REAL) \
    X(EIGEN_COMPLEX) X(SCALING_MANUAL) X(SCALING_AUTO) X(SCALING_ALWAYS) X(SCALING_DYNAMIC) X(SCALERS_RAW) \
    X(SCALERS_LOG) X(INVEVEC_STANDARD) X(INVEVEC_TRANSPOSED) X(VECTOR_SSE) X(VECTOR_AVX) X(VECTOR_TENSOR) \
    X(VECTOR_NONE) X(THREADING_CPP) X(THREADING_OPENMP) X(THREADING_NONE) X(PROCESSOR_CPU) X(PROCESSOR_GPU) \
    X(PROCESSOR_FPGA) X(PROCESSOR_CELL) X(PROCESSOR_PHI) X(PROCESSOR_OTHER) X(FRAMEWORK_CUDA) \
    X(FRAMEWORK_OPENCL) X(FRAMEWORK_CPU) X(FRAMEWORK_TINYGPU) X(PARALLELOPS_STREAMS) X(PARALLELOPS_GRID) \
    X(PARALLELOPS_FUSED) X(PREORDER_TRANSPOSE_MANUAL) X(PREORDER_TRANSPOSE_AUTO) X(SPECTRAL_REPRESENTATION)
#define FLAG_OR(name) | (long long) BEAGLE_FLAG_##name
#define FLAG_SUM(name) + (long long) BEAGLE_FLAG_##name
#define FLAG_ONE_BIT(name) && BEAGLE_FLAG_##name > 0 && (BEAGLE_FLAG_##name & (BEAGLE_FLAG_##name - 1)) == 0
#define DEFINED_FLAGS (0 ALL_FLAGS(FLAG_OR))

/* each flag is one bit; bits that are all distinct add up to their union */
FLAGBITS_ASSERT(1 ALL_FLAGS(FLAG_ONE_BIT), "a flag is not a single bit");
FLAGBITS_ASSERT((0 ALL_FLAGS(FLAG_SUM)) == DEFINED_FLAGS, "two flags share a bit");

#define FIELD_SIZE(type, field) sizeof(((type*) 0)->field)
FLAGBITS_ASSERT(FIELD_SIZE(BeagleInstanceDetails, flags) == 8, "BeagleInstanceDetails::flags is not 8 bytes");
FLAGBITS_ASSERT(FIELD_SIZE(BeagleResource, supportFlags) == 8, "BeagleResource::supportFlags is not 8 bytes");
FLAGBITS_ASSERT(FIELD_SIZE(BeagleResource, requiredFlags) == 8, "BeagleResource::requiredFlags is not 8 bytes");
FLAGBITS_ASSERT(FIELD_SIZE(BeagleBenchmarkedResource, supportFlags) == 8,
                "BeagleBenchmarkedResource::supportFlags is not 8 bytes");
FLAGBITS_ASSERT(FIELD_SIZE(BeagleBenchmarkedResource, requiredFlags) == 8,
                "BeagleBenchmarkedResource::requiredFlags is not 8 bytes");
FLAGBITS_ASSERT(FIELD_SIZE(BeagleBenchmarkedResource, benchedFlags) == 8,
                "BeagleBenchmarkedResource::benchedFlags is not 8 bytes");

static int failures = 0;

static void check(int ok, const char* what, long long flags) {
    printf("%s: %s (0x%llx)\n", ok ? "ok  " : "FAIL", what, (unsigned long long) flags);
    if (!ok) failures++;
}

int main(void) {
    const long long cpuHighFlags = BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO | BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL |
                                   BEAGLE_FLAG_SPECTRAL_REPRESENTATION;
    const long long fromBit31 = ~((1LL << 31) - 1);
    const long long required = BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_SPECTRAL_REPRESENTATION |
                               BEAGLE_FLAG_FRAMEWORK_CPU;
    const long long aboveBit46 = ~((1LL << 47) - 1);
    BeagleResourceList* resources = beagleGetResourceList();
    BeagleInstanceDetails details;
    BeagleBenchmarkedResourceList* benchmarked;
    int cpu = -1, i, instance;

    for (i = 0; resources != NULL && i < resources->length; i++) {
        if (resources->list[i].requiredFlags & BEAGLE_FLAG_FRAMEWORK_CPU) {
            cpu = i;
            break;
        }
    }
    if (cpu < 0) {
        printf("FAIL: no CPU resource\n");
        return 1;
    }
    printf("CPU resource %d: %s\n", cpu, resources->list[cpu].name);
    check((resources->list[cpu].supportFlags & fromBit31) == cpuHighFlags,
          "CPU resource supportFlags from bit 31 up are PREORDER_TRANSPOSE_AUTO, _MANUAL and SPECTRAL_REPRESENTATION",
          resources->list[cpu].supportFlags);
    check((resources->list[cpu].supportFlags & ~DEFINED_FLAGS) == 0,
          "CPU resource supportFlags have no undefined bit", resources->list[cpu].supportFlags);
    check((resources->list[cpu].requiredFlags & ~DEFINED_FLAGS) == 0,
          "CPU resource requiredFlags have no undefined bit", resources->list[cpu].requiredFlags);

    instance = beagleCreateInstance(3, 5, 0, 4, 2, 1, 4, 1, 0, NULL, 0, 0, required, &details);
    check(instance >= 0, "CPU instance requiring PREORDER_TRANSPOSE_MANUAL | SPECTRAL_REPRESENTATION | FRAMEWORK_CPU",
          required);
    if (instance >= 0) {
        printf("instance %d: %s on resource %d\n", instance, details.implName, details.resourceNumber);
        check(strstr(details.implName, "Spectral") != NULL, "it is a spectral implementation", details.flags);
        check((details.flags & aboveBit46) == 0, "its flags have nothing above bit 46", details.flags);
        check((details.flags & ~DEFINED_FLAGS) == 0, "its flags have no undefined bit", details.flags);
        beagleFinalizeInstance(instance);
    }

    instance = beagleCreateInstance(3, 5, 0, 4, 2, 1, 4, 1, 0, NULL, 0, 0,
                                    BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO | BEAGLE_FLAG_FRAMEWORK_CPU, &details);
    check(instance >= 0, "CPU instance requiring PREORDER_TRANSPOSE_AUTO | FRAMEWORK_CPU",
          BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO | BEAGLE_FLAG_FRAMEWORK_CPU);
    if (instance >= 0) {
        printf("instance %d: %s on resource %d\n", instance, details.implName, details.resourceNumber);
        beagleFinalizeInstance(instance);
    }

    /* the flag parameters of beagleGetBenchmarkedResourceList and the flag fields of BeagleBenchmarkedResource */
    benchmarked = beagleGetBenchmarkedResourceList(3, 3, 4, 10, 1, &cpu, 1, BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO,
                                                   required, 1, 1, 0, 0);
    check(benchmarked != NULL && benchmarked->length == 1,
          "benchmarked CPU resource list, requiring PREORDER_TRANSPOSE_MANUAL | SPECTRAL_REPRESENTATION", required);
    if (benchmarked != NULL && benchmarked->length == 1) {
        const BeagleBenchmarkedResource* b = &benchmarked->list[0];
        printf("benchmarked resource %d: return code %d, %s\n", b->number, b->returnCode,
               b->returnCode == BEAGLE_SUCCESS ? b->implName : "-");
        check(b->returnCode == BEAGLE_SUCCESS && strstr(b->implName, "Spectral") != NULL,
              "it benchmarked a spectral implementation", b->benchedFlags);
        check(b->supportFlags == resources->list[cpu].supportFlags, "its supportFlags are the resource's",
              b->supportFlags);
        check(b->requiredFlags == resources->list[cpu].requiredFlags, "its requiredFlags are the resource's",
              b->requiredFlags);
        check((b->benchedFlags & aboveBit46) == 0, "its benchedFlags have nothing above bit 46", b->benchedFlags);
        check((b->benchedFlags & ~DEFINED_FLAGS) == 0, "its benchedFlags have no undefined bit", b->benchedFlags);
    }

    printf("%s\n", failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
