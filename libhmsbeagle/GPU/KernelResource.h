/*
 *  KernelResource.h
 *  BEAGLE
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
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
        int inSumIntervalBlockSize,
        int inSumAcrossBlockSize,
        int inBlockPeelingSizeSCA,
        int inCategoryCount,
        int inPatternCount,
        int inUnpaddedPatternCount,
        long inFlags
        );

    KernelResource(const KernelResource& krIn,
                   char* inKernelCode);

    virtual ~KernelResource();

    int paddedStateCount;
    int categoryCount;
    int patternCount;
    int unpaddedPatternCount;
    char* kernelCode;
    int patternBlockSize;
    int matrixBlockSize;
    int sumIntervalBlockSize;
    int sumAcrossBlockSize;
    int blockPeelingSizeSCA;
    int blockPeelingSize;
    int isPowerOfTwo;
    int smallestPowerOfTwo;
    int slowReweighing;
    int multiplyBlockSize;
    long flags;

    KernelResource* copy();
};

#endif /* KERNELRESOURCE_H_ */
