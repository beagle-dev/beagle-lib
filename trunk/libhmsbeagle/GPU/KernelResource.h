/*
 *  KernelResource.h
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

#ifndef KERNELRESOURCE_H_
#define KERNELRESOURCE_H_

class KernelResource {
public:
    KernelResource();
    
    KernelResource(
        int inPaddedStateCount,
        char* inKernelString,
        int inPatternBlockSize,
        int inMatrixBlockSize,
        int inBlockPeelingSize,
        int inSlowReweighing,
        int inMultiplyBlockSize,
        int inCategoryCount,
        int inPatternCount,
        long inFlags
        );
    
    KernelResource(const KernelResource& krIn,
                   char* inKernelCode);
    
    virtual ~KernelResource();
    
    int paddedStateCount;
    int categoryCount;
    int patternCount;
    char* kernelCode;
    int patternBlockSize;
    int matrixBlockSize;
    int blockPeelingSize;
    int isPowerOfTwo;
    int smallestPowerOfTwo;
    int slowReweighing;
    int multiplyBlockSize;
    long flags;
    
    KernelResource* copy();
};

#endif /* KERNELRESOURCE_H_ */
