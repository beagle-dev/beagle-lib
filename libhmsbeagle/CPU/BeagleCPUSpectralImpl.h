/*
 *  BeagleCPUSpectralImpl.h
 *  BEAGLE
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @author Marc Suchard
 */

#ifndef __BeagleCPUSpectralImp__
#define __BeagleCPUSpectralImp__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/CPU/BeagleCPUImpl.h"

#include <vector>

namespace beagle {
    namespace cpu {

        BEAGLE_CPU_TEMPLATE
        class BeagleCPUSpectralImpl : public BeagleCPUImpl<BEAGLE_CPU_GENERIC> {

        protected:
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gEigenDecomposition;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPartialsSize;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kFlags;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kTipCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kMatrixCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPartials;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::integrationTmp;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gTransitionMatrices;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPatternCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kMatrixSize;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPaddedPatternCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kTransPaddedStateCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPartialsPaddedStateCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kExtraPatterns;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kStateCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gTipStates;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kCategoryCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gScaleBuffers;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gStateFrequencies;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gCategoryWeights;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gCategoryRates;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::grandNumeratorDerivTmp;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::grandDenominatorDerivTmp;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPatternWeights;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::outLogLikelihoodsTmp;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::realtypeMin;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::scalingExponentThreshold;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPatternPartitionsStartPatterns;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateDerivatives;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::calcEdgeLogDerivatives;

            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateScaleFactors;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::rescalePartials;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::rescalePartialsByPartition;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::autoRescalePartials;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::removeScaleFactors;

            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gActiveScalingFactors;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gAutoScaleBuffers;

        public:
            // BeagleCPUSpectralImpl();
            virtual ~BeagleCPUSpectralImpl();
            virtual const char *getName();

            int createInstance(int tipCount,
                              int partialsBufferCount,
                              int compactBufferCount,
                              int stateCount,
                              int patternCount,
                              int eigenDecompositionCount,
                              int matrixCount,
                              int categoryCount,
                              int scaleBufferCount,
                              int resourceNumber,
                              int pluginResourceNumber,
                              long preferenceFlags,
                              long requirementFlags);

        private:
            struct BranchEigenInfo {
                REALTYPE branchLength;
                int eigenIndex;
                int categoryRatesIndex;
            };
            std::vector<BranchEigenInfo> gBranchEigenInfo;

            std::vector<REALTYPE> gPartialTmp1;
            std::vector<REALTYPE> gPartialTmp2;

        protected:
            int updateTransitionMatrices(int eigenIndex,
                                         const int* probabilityIndices,
                                         const int* firstDerivativeIndices,
                                         const int* secondDerivativeIndices,
                                         const double* edgeLengths,
                                         int count);

            virtual int upPartials(bool byPartition,
                                   const int* operations,
                                   int operationCount,
                                   int cumulativeScalingIndex,
                                   BeaglePartialsType partialsType);

            void calcPartialsPartials(REALTYPE *destPartials,
                                      const REALTYPE *partials1,
                                      const int branchEigenIndex1,
                                      const REALTYPE *partials2,
                                      const int branchEigenIndex2,
                                      int startPattern,
                                      int endPattern,
                                      bool isComplex);

            virtual void calcStatesStates(REALTYPE *destP,
                                          const int *states1,
                                          const int branchEigenIndex1,
                                          const int *states2,
                                          const int branchEigenIndex2,
                                          int startPattern,
                                          int endPattern);

            virtual void calcStatesPartials(REALTYPE *destP,
                                            const int *states1,
                                            const int branchEigenIndex1,
                                            const REALTYPE *partials2,
                                            const int branchEigenIndex2,
                                            int startPattern,
                                            int endPattern);

            virtual void calcPrePartialsPartials(REALTYPE *destP,
                                                 const REALTYPE *partialsParent,
                                                 const REALTYPE *matricesSelf,
                                                 const REALTYPE *partialsSibling,
                                                 const REALTYPE *matricesSibling,
                                                 int startPattern,
                                                 int endPattern);

        //     virtual void calcPrePartialsStates(REALTYPE *destP,
        //                                        const REALTYPE *partials1,
        //                                        const REALTYPE *matrices1,
        //                                        const int *states2,
        //                                        const REALTYPE *matrices2,
        //                                        int startPattern,
        //                                        int endPattern);

        };

        BEAGLE_CPU_FACTORY_TEMPLATE
        class BeagleCPUSpectralImplFactory : public BeagleImplFactory
        {
        public:
            virtual BeagleImpl *createImpl(int tipCount,
                                           int partialsBufferCount,
                                           int compactBufferCount,
                                           int stateCount,
                                           int patternCount,
                                           int eigenBufferCount,
                                           int matrixBufferCount,
                                           int categoryCount,
                                           int scaleBufferCount,
                                           int resourceNumber,
                                           int pluginResourceNumber,
                                           long preferenceFlags,
                                           long requirementFlags,
                                           int *errorCode);

            virtual const char *getName();
            virtual const long getFlags();
        };

    } // namespace cpu
} // namespace beagle

// now include the file containing template function implementations
#include "libhmsbeagle/CPU/BeagleCPUSpectralImpl.hpp"

#endif // __BeagleCPUSpectralImp__
