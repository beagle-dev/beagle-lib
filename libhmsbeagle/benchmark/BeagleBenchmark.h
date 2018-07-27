/*
 *  beagleBenchmark.h
 *  Resource/implementation benchmarking
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * BEAGLE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * BEAGLE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAGLE.  If not, see
 * <http://www.gnu.org/licenses/>.
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

#define BENCHMARK_REPLICATES 10
#define GT_RAND_MAX 0x7fffffff

#ifdef _WIN32
    //From January 1, 1601 (UTC). to January 1,1970
    #define FACTOR 0x19db1ded53e8000 
#else
    #include <sys/time.h>
#endif

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
