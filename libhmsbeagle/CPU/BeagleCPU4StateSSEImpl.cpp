/*
 *  BeagleCPU4StateSSEImpl.cpp
 *  BEAGLE
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
 * @author Marc Suchard
 */

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPU4StateSSEImpl.h"

#ifdef __GNUC__
    #define cpuid(func,ax,bx,cx,dx)\
            __asm__ __volatile__ ("cpuid":\
            "=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "a" (func)); 
#endif

#ifdef _WIN32

#endif

using namespace beagle;
using namespace beagle::cpu;

#if defined (BEAGLE_IMPL_DEBUGGING_OUTPUT) && BEAGLE_IMPL_DEBUGGING_OUTPUT
const bool DEBUGGING_OUTPUT = true;
#else
const bool DEBUGGING_OUTPUT = false;
#endif

BeagleCPU4StateSSEImpl::~BeagleCPU4StateSSEImpl() {
}

int BeagleCPU4StateSSEImpl::CPUSupportsSSE() {
    //int a,b,c,d;
    //cpuid(0,a,b,c,d);
    //fprintf(stderr,"a = %d\nb = %d\nc = %d\nd = %d\n",a,b,c,d);
    return 1;
}

void BeagleCPU4StateSSEImpl::calcPartialsPartials(double* destP,
                                                  const double* partials1,
                                                  const double* matrices1,
                                                  const double* partials2,
                                                  const double* matrices2) {
    fprintf(stderr,"EXPERIMENTAL calcPartialsPartials -- SSE!!\n");
    BeagleCPU4StateImpl::calcPartialsPartials(destP,
                                              partials1,matrices1,
                                              partials2,matrices2);
}


///////////////////////////////////////////////////////////////////////////////
// BeagleImplFactory public methods

BeagleImpl* BeagleCPU4StateSSEImplFactory::createImpl(int tipCount,
                                             int partialsBufferCount,
                                             int compactBufferCount,
                                             int stateCount,
                                             int patternCount,
                                             int eigenBufferCount,
                                             int matrixBufferCount,
                                             int categoryCount,
                                             int scaleBufferCount,
                                             long preferenceFlags,
                                             long requirementFlags) {

    if (stateCount != 4) {
        return NULL;
    }
    
    fprintf(stderr,"EXPERIMENTAL FACTORY -- SSE!!\n");
    
    BeagleCPU4StateSSEImpl* impl = new BeagleCPU4StateSSEImpl();
    
    if (impl->CPUSupportsSSE()) {        
        fprintf(stderr,"CPU supports SSE!\n");
    } else {
        delete impl;
        return NULL;            
    }

    try {
        if (impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount,scaleBufferCount, preferenceFlags, requirementFlags) == 0)
            return impl;
    }
    catch(...) {
        if (DEBUGGING_OUTPUT)
            std::cerr << "exception in initialize\n";
        delete impl;
        throw;
    }

    delete impl;

    return NULL;
}

const char* BeagleCPU4StateSSEImplFactory::getName() {
    return "CPU-4StateSSE";
}

const long BeagleCPU4StateSSEImplFactory::getFlags() {
    return BEAGLE_FLAG_ASYNCH | BEAGLE_FLAG_CPU | BEAGLE_FLAG_DOUBLE | BEAGLE_FLAG_SSE;
}

