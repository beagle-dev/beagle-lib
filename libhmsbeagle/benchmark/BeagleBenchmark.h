/*
 *  beagleBenchmark.h
 *  Resource/implementation benchmarking
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @author Daniel Ayres
 */

#ifndef __beagle_benchmark__
#define __beagle_benchmark__

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <stack>
#include <queue>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/benchmark/linalg.h"

#define BENCHMARK_REPLICATES 15
#define GT_RAND_MAX 0x7fffffff

#ifdef _WIN32
    //From January 1, 1601 (UTC). to January 1,1970
    #define FACTOR 0x19db1ded53e8000
	#include <winsock.h>
	#include <string>
#else
    #include <sys/time.h>
#endif

namespace beagle {
namespace benchmark {

int benchmarkResource(int resource,
                         int stateCount,
                         int ntaxa,
                         int nsites,
                         bool manualScaling,
                         int rateCategoryCount,
                         int nreps,
                         int compactTipCount,
                         int rescaleFrequency,
                         bool unrooted,
                         bool calcderivs,
                         int eigenCount,
                         int partitionCount,
                         long preferenceFlags,
                         long requirementFlags,
                         int* resourceNumber,
                         char** implName,
                         long* benchedFlags,
                         double* benchmarkResult,
                         bool instOnly);

#endif // __beagle_benchmark__


}   // namespace benchmark
}   // namespace beagle
