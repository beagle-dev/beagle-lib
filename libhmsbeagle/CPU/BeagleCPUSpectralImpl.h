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
#include <cmath>

#define TEST_EB

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
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kEigenDecompCount;
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

            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kThreadingEnabled;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kAutoPartitioningEnabled;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPartitionCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPatternScaleTmp;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gOuterProductTmp;

        public:
            // BeagleCPUSpectralImpl();
            ~BeagleCPUSpectralImpl() override;
            const char *getName() override;

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
                              long requirementFlags) override;

            int setCPUThreadCount(int threadCount) override;

        protected:
            EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>* createEigenDecomposition(
                    int decompositionCount,
                    int stateCount,
                    int categoryCount,
                    long flags) override;

        private:
            struct BranchEigenInfo {
                REALTYPE branchLength;

                int eigenIndex;
                int categoryRatesIndex;

                bool allReal;

#ifdef TEST_EB
                REALTYPE* time;

                REALTYPE* expat;
                REALTYPE* cosbt;
                REALTYPE* sinbt;
                REALTYPE* expatcosbt;
                REALTYPE* expatsinbt;

                const REALTYPE* eval;
#endif
            };
            std::vector<BranchEigenInfo> gBranchEigenInfo;

#ifdef TEST_EB
            std::vector<REALTYPE> gTime;
            std::vector<REALTYPE> gExpAt;
            std::vector<REALTYPE> gCosBt;
            std::vector<REALTYPE> gSinBt;
            std::vector<REALTYPE> gExpAtCosBt;
            std::vector<REALTYPE> gExpAtSinBt;
#endif

            std::vector<REALTYPE> gPartialTmp1;
            std::vector<REALTYPE> gPartialTmp2;


            int kStateCountModFour;

        protected:
            int updateTransitionMatrices(int eigenIndex,
                                         const int* probabilityIndices,
                                         const int* firstDerivativeIndices,
                                         const int* secondDerivativeIndices,
                                         const double* edgeLengths,
                                         int count) override;

            void calcAdjointCrossProductsRange(const int *postBufferIndices,
                                               const int *preBufferIndices,
                                               const int *branchEigenIndices,
                                               const double *categoryRates,
                                               const REALTYPE *categoryWeights,
                                               const REALTYPE *perSiteLikelihoods,
                                               REALTYPE *buffer,
                                               int count,
                                               int startPattern,
                                               int endPattern,
                                               int currentPartition) override;

            int upPartials(bool byPartition,
                           const int* operations,
                           int operationCount,
                           int cumulativeScalingIndex,
                           BeaglePartialsType partialsType) override;

            int upPrePartials(bool byPartition,
                              const int* operations,
                              int count,
                              int cumulativeScaleIndex,
                              BeaglePartialsType preorderType) override;

            template <typename T>
            int upPrePartialsImpl(bool byPartition,
                                  const int* operations,
                                  int count,
                                  int cumulativeScaleIndex);

            template <typename T>
            void calcPartialsPartials(REALTYPE *destPartials,
                                      const REALTYPE *partials1,
                                      const int branchEigenIndex1,
                                      const REALTYPE *partials2,
                                      const int branchEigenIndex2,
                                      const REALTYPE* scaleFactors,
                                      int startPattern,
                                      int endPattern,
                                      int currentPartition);

            template <typename T>
            void calcStatesStates(REALTYPE *destP,
                                  const int *states1,
                                  const int branchEigenIndex1,
                                  const int *states2,
                                  const int branchEigenIndex2,
                                  const REALTYPE* scaleFactors,
                                  int startPattern,
                                  int endPattern,
                                  int currentPartition);

            template <typename T>
            void calcStatesPartials(REALTYPE *destP,
                                    const int *states1,
                                    const int branchEigenIndex1,
                                    const REALTYPE *partials2,
                                    const int branchEigenIndex2,
                                    const REALTYPE* scaleFactors,
                                    int startPattern,
                                    int endPattern,
                                    int currentPartition);

            template <typename T, typename V>
            void calcPrePartialsPartials(REALTYPE *destP,
                                         const REALTYPE *partialsParent,
                                         const int branchEigenIndex1,
                                         const REALTYPE *partials2,
                                         const int branchEigenIndex2,
                                         int startPattern,
                                         int endPattern,
                                         int currentPartition);

            template <typename T, typename V>
            void calcPrePartialsStates(REALTYPE *destP,
                                       const REALTYPE *partials1,
                                       const int branchEigenIndex1,
                                       const int *states2,
                                       const int branchEigenIndex2,
                                       int startPattern,
                                       int endPattern,
                                       int currentPartition);

            template <typename T, typename V, typename COLLECTOR>
            void calcAdjointCrossProducts(const REALTYPE* partialsPost,
                                          const int* tipsPost,
                                          const REALTYPE* partialsPre,
                                          const int branchIndex,
                                          const REALTYPE* perSiteLikelihoods,
                                          const int category,
                                          const REALTYPE categoryWeight,
                                          COLLECTOR& collector,
                                          int startPattern,
                                          int endPattern,
                                          int currentPartition);


            template <typename First, typename Second, typename Body>
            inline void matVecDual(
                    const REALTYPE* mat1, const REALTYPE* vec1, const int state1,
                    const REALTYPE* mat2, const REALTYPE* vec2, const int state2,
                    const int matrixIncr,
                    Body body);

            // template <typename T>
            // void calcPrePartialsPartialsRoot(REALTYPE *destP,
            //                                  const REALTYPE *partialsParent,
            //                                  const int branchEigenIndexSelf,
            //                                  const REALTYPE *partialsSibling,
            //                                  const int branchEigenIndexSibling,
            //                                  int startPattern,
            //                                  int endPattern);

            // template <typename T>
            // void calcPrePartialsStatesRoot(REALTYPE *destP,
            //                                const REALTYPE *partials1,
            //                                const int branchEigenIndex1,
            //                                const int *states2,
            //                                const int branchEigenIndex2,
            //                                int startPattern,
            //                                int endPattern);

            template <typename First, typename Second>
            void expScaledMatrixVectorMultiple(REALTYPE* outPartials1, REALTYPE* outPartials2,
                                               const REALTYPE* inPartials1, const int state1,
                                               const REALTYPE* eigenValuesReal1,
                                               const REALTYPE* eigenValuesImag1,
                                               const REALTYPE* inverseEigenVectors1,
                                               const REALTYPE scaledBranchLength1,
                                               const REALTYPE* inPartials2, const int state2,
                                               const REALTYPE* eigenValuesReal2,
                                               const REALTYPE* eigenValuesImag2,
                                               const REALTYPE* inverseEigenVectors2,
                                               const REALTYPE scaledBranchLength2,
                                               const int matrixIncr);

#ifdef TEST_EB
            template <typename First, typename Second, typename Direction>
            void expScaledMatrixVectorMultiple2(REALTYPE* outPartials1, REALTYPE* outPartials2,
                                               const REALTYPE* inPartials1, const int state1,
                                               const BranchEigenInfo& info1,
                                               const REALTYPE* inverseEigenVectors1,
                                               const REALTYPE* inPartials2, const int state2,
                                               const BranchEigenInfo& info2,
                                               const REALTYPE* inverseEigenVectors2,
                                               const int matrixIncr);
            template <typename First, typename Second, typename Direction>
            void expScaledMatrixVectorMultiple3(REALTYPE* outPartials1, REALTYPE* outPartials2,
                                               const REALTYPE* inPartials1, const int state1,
                                               const BranchEigenInfo& info1,
                                               const REALTYPE* inverseEigenVectors1,
                                               const REALTYPE* inPartials2, const int state2,
                                               const BranchEigenInfo& info2,
                                               const REALTYPE* inverseEigenVectors2,
                                               const int matrixIncr);
#endif
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

        template <typename T>
        struct always_false : std::false_type {};

        struct States {};
        struct Partials {};
        struct None {};

        struct NoRotation {};
        struct WithRotation {};

        struct Top {};
        struct Bottom {};

        struct Forward {};
        struct Backward {};

        struct NoScaling {
            static constexpr bool useScaleFactors = false;
        };

        struct WithScaling {
            static constexpr bool useScaleFactors = true;
        };

        struct Root { };
        struct NotRoot { };
        struct NotUsed { };

    } // namespace cpu
} // namespace beagle

// now include the file containing template function implementations
#include "libhmsbeagle/CPU/BeagleCPUSpectralImpl.hpp"

#endif // __BeagleCPUSpectralImp__
