/*
 *  BeagleCPU4StateGCDImpl.h
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
 * @author Andrew Rambaut
 */

#ifndef __BeagleCPU4StateGCDImpl__
#define __BeagleCPU4StateGCDImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/CPU/BeagleCPU4StateImpl.h"

#include <vector>

namespace beagle {
namespace cpu {

class BeagleCPU4StateGCDImpl : public BeagleCPU4StateImpl {

};

class BeagleCPU4StateGCDImplFactory : public BeagleImplFactory {
public:
    virtual BeagleImpl* createImpl(int tipCount,
                                   int partialsBufferCount,
                                   int compactBufferCount,
                                   int stateCount,
                                   int patternCount,
                                   int eigenBufferCount,
                                   int matrixBufferCount,
                                   int categoryCount,
                                   int scaleBufferCount,
                                   long preferenceFlags,
                                   long requirementFlags);

    virtual const char* getName();
    virtual const long getFlags();
};

}	// namespace cpu
}	// namespace beagle

#endif // __BeagleCPU4StateGCDImpl__
