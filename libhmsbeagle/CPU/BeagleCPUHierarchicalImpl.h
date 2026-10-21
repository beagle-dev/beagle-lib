/*
 *  BeagleCPUHierarchicalImpl.h
 *  BEAGLE
 *
 * Copyright 2026 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#ifndef __BeagleCPUHierarchicalImpl__
#define __BeagleCPUHierarchicalImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/CPU/BeagleCPUImpl.h"
#include "libhmsbeagle/CPU/HierarchicalMethods.h"

#include <vector>

namespace beagle {
    namespace cpu {

        BEAGLE_CPU_TEMPLATE
        class BeagleCPUHierarchicalImpl : public BeagleCPUImpl<BEAGLE_CPU_GENERIC> {

        protected:
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kFlags;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kStateCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kMatrixCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kTipCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kBufferCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPatternCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPaddedPatternCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPartialsPaddedStateCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kCategoryCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kEigenDecompCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPartialsSize;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPartitionCount;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::kPatternsReordered;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPartials;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gTipStates;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gScaleBuffers;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gStateFrequencies;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPatternWeights;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPatternPartitions;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPatternPartitionsStartPatterns;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gPatternsNewOrder;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::mallocAligned;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gActiveScalingFactors;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::accumulateScaleFactors;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::removeScaleFactors;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gCategoryWeights;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::gCategoryRates;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::integrationTmp;
            using BeagleCPUImpl<BEAGLE_CPU_GENERIC>::outLogLikelihoodsTmp;

            std::vector<REALTYPE> gW;
            std::vector<REALTYPE> gV;
            std::vector<REALTYPE> gInvScaleTmp;
            int kTileSize;

            std::vector<HierarchicalSlot<REALTYPE>> gSlots;
            std::vector<HierarchicalModel> gModels;
            std::vector<std::vector<double>> gFrequenciesDouble;
            std::vector<double> gPiNodeTmp;
            std::vector<double> gPrZ0Tmp;
            std::vector<double> gSumHTmp;
            int kKernel;
            bool kSmallBand;
            int kShapeMaxStates;
            bool kRescaleRows;
            bool kFusedBand;
            bool kColumnGather;
            size_t kTableBudget;
            size_t gTableBytes;
            int kGroupLanes;
            int kGroupOrder;
            unsigned long gFrequencyStamp;
            std::vector<REALTYPE> gSiteScaleTmp;
            std::vector<double> gSiteAutoScaleTmp;

            inline int pIdx(int l, int k, int i) const {
                return (l * kPartialsPaddedStateCount + i) * kPaddedPatternCount + k;
            }

            void transposeToSiteMajor(REALTYPE* dest,
                                      const double* src,
                                      int srcCategoryStride,
                                      int srcPatternStride);

            void transposeFromSiteMajor(double* dest,
                                        const REALTYPE* src);

            int ensurePartials(int bufferIndex);

            bool slotsReady(const int* operations, int count, int numOps);

            int smallPlan(bool states1,
                          const HierarchicalSlot<REALTYPE>& slot1,
                          bool singleChild,
                          bool states2,
                          const HierarchicalSlot<REALTYPE>* slot2);

            bool tableReady(const HierarchicalSlot<REALTYPE>& slot);

            void prepareTables(const int* operations, int count, int numOps);

            void buildTable(HierarchicalSlot<REALTYPE>& slot);

            bool fusedColumnPartials(REALTYPE* destP,
                                     const REALTYPE* partials1,
                                     const int* states1,
                                     const HierarchicalSlot<REALTYPE>& slot1,
                                     bool singleChild,
                                     const REALTYPE* partials2,
                                     const int* states2,
                                     const HierarchicalSlot<REALTYPE>* slot2,
                                     const REALTYPE* fixedScaleFactors,
                                     REALTYPE* scaleFactors,
                                     REALTYPE* cumulativeScaleFactors,
                                     int startPattern,
                                     int endPattern);

            bool smallPartials(REALTYPE* destP,
                               const REALTYPE* partials1,
                               const int* states1,
                               const HierarchicalSlot<REALTYPE>& slot1,
                               bool singleChild,
                               const REALTYPE* partials2,
                               const int* states2,
                               const HierarchicalSlot<REALTYPE>* slot2,
                               const REALTYPE* scaleFactors,
                               int startPattern,
                               int endPattern);

            template <int S>
            bool smallPartialsS(int plan,
                                REALTYPE* destP,
                                const REALTYPE* partials1,
                                const int* states1,
                                const HierarchicalSlot<REALTYPE>& slot1,
                                const REALTYPE* partials2,
                                const int* states2,
                                const HierarchicalSlot<REALTYPE>* slot2,
                                const REALTYPE* scaleFactors,
                                int startPattern,
                                int endPattern);

            template <int S, bool Scaled>
            void smallCategory(int plan,
                               int category,
                               REALTYPE* destP,
                               const REALTYPE* partials1,
                               const int* states1,
                               const HierarchicalSlot<REALTYPE>& slot1,
                               const REALTYPE* partials2,
                               const int* states2,
                               const HierarchicalSlot<REALTYPE>* slot2,
                               const REALTYPE* invScale,
                               int startPattern,
                               int endPattern);

            int buildBundles(int modelIndex,
                             int categoryRatesIndex,
                             const int* probabilityIndices,
                             const double* edgeLengths,
                             int count);

            template <int Mode, typename Source>
            void sweep(const HierarchicalSlot<REALTYPE>& slot,
                       const Source& source,
                       REALTYPE* destP,
                       const REALTYPE* invScale,
                       int category,
                       int startPattern,
                       int endPattern);

            template <typename Source1, typename Source2>
            void hierarchicalPartials(REALTYPE* destP,
                                      const Source1& source1,
                                      const HierarchicalSlot<REALTYPE>& slot1,
                                      const Source2* source2,
                                      const HierarchicalSlot<REALTYPE>* slot2,
                                      const REALTYPE* scaleFactors,
                                      int startPattern,
                                      int endPattern);

            template <typename Source1, typename Source2>
            void hierarchicalPartialsFused(REALTYPE* destP,
                                           const Source1& source1,
                                           const HierarchicalSlot<REALTYPE>& slot1,
                                           const Source2* source2,
                                           const HierarchicalSlot<REALTYPE>* slot2,
                                           const REALTYPE* fixedScaleFactors,
                                           REALTYPE* scaleFactors,
                                           REALTYPE* cumulativeScaleFactors,
                                           int startPattern,
                                           int endPattern);

            template <typename Source1, typename Source2>
            void groupSweep(REALTYPE* destP,
                            const Source1& source1,
                            const HierarchicalSlot<REALTYPE>& slot1,
                            const Source2* source2,
                            const HierarchicalSlot<REALTYPE>* slot2,
                            const REALTYPE* fixedScaleFactors,
                            int category,
                            int k0,
                            int k1);

            void rescaleRange(REALTYPE* destP,
                              REALTYPE* scaleFactors,
                              REALTYPE* cumulativeScaleFactors,
                              int startPattern,
                              int endPattern);

        public:
            ~BeagleCPUHierarchicalImpl() override;

            const char* getName() override;

            const long getFlags() override;

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

            int setTipPartials(int tipIndex,
                               const double* inPartials) override;

            int setPartials(int bufferIndex,
                            const double* inPartials) override;

            int setRootPrePartials(const int* bufferIndices,
                                   const int* stateFrequenciesIndices,
                                   int count) override;

            int getPartials(int bufferIndex,
                            int scaleBuffer,
                            double* outPartials) override;

            int setEigenDecomposition(int eigenIndex,
                                      const double* inEigenVectors,
                                      const double* inInverseEigenVectors,
                                      const double* inEigenValues) override;

            int setStateFrequencies(int stateFrequenciesIndex,
                                    const double* inStateFrequencies) override;

            int setTransitionMatrix(int matrixIndex,
                                    const double* inMatrix,
                                    double paddedValue) override;

            int setTransitionMatrices(const int* matrixIndices,
                                      const double* inMatrices,
                                      const double* paddedValues,
                                      int count) override;

            int getTransitionMatrix(int matrixIndex,
                                    double* outMatrix) override;

            int updateTransitionMatrices(int eigenIndex,
                                         const int* probabilityIndices,
                                         const int* firstDerivativeIndices,
                                         const int* secondDerivativeIndices,
                                         const double* edgeLengths,
                                         int count) override;

            int updateTransitionMatricesWithModelCategories(int* eigenIndices,
                                                            const int* probabilityIndices,
                                                            const int* firstDerivativeIndices,
                                                            const int* secondDerivativeIndices,
                                                            const double* edgeLengths,
                                                            int count) override;

            int updateTransitionMatricesWithMultipleModels(const int* eigenIndices,
                                                           const int* categoryRateIndices,
                                                           const int* probabilityIndices,
                                                           const int* firstDerivativeIndices,
                                                           const int* secondDerivativeIndices,
                                                           const double* edgeLengths,
                                                           int count) override;

            int updatePartials(const int* operations,
                               int operationCount,
                               int cumulativeScalingIndex,
                               BeaglePartialsType partialsType) override;

            int updatePrePartials(const int* operations,
                                  int operationCount,
                                  int cumulativeScalingIndex,
                                  BeaglePartialsType partialsType) override;

            int updatePartialsByPartition(const int* operations,
                                          int operationCount) override;

            int updatePrePartialsByPartition(const int* operations,
                                             int operationCount,
                                             BeaglePartialsType partialsType) override;

            int calculateEdgeLogLikelihoods(const int* parentBufferIndices,
                                            const int* childBufferIndices,
                                            const int* probabilityIndices,
                                            const int* firstDerivativeIndices,
                                            const int* secondDerivativeIndices,
                                            const int* categoryWeightsIndices,
                                            const int* stateFrequenciesIndices,
                                            const int* cumulativeScaleIndices,
                                            int count,
                                            double* outSumLogLikelihood,
                                            double* outSumFirstDerivative,
                                            double* outSumSecondDerivative) override;

            int calculateEdgeLogLikelihoodsByPartition(const int* parentBufferIndices,
                                                       const int* childBufferIndices,
                                                       const int* probabilityIndices,
                                                       const int* firstDerivativeIndices,
                                                       const int* secondDerivativeIndices,
                                                       const int* categoryWeightsIndices,
                                                       const int* stateFrequenciesIndices,
                                                       const int* cumulativeScaleIndices,
                                                       const int* partitionIndices,
                                                       int partitionCount,
                                                       int count,
                                                       double* outSumLogLikelihoodByPartition,
                                                       double* outSumLogLikelihood,
                                                       double* outSumFirstDerivativeByPartition,
                                                       double* outSumFirstDerivative,
                                                       double* outSumSecondDerivativeByPartition,
                                                       double* outSumSecondDerivative) override;

            int calculateEdgeDerivatives(const int* postBufferIndices,
                                         const int* preBufferIndices,
                                         const int* derivativeMatrixIndices,
                                         const int* categoryWeightsIndices,
                                         const int* categoryRatesIndices,
                                         const int* cumulativeScaleIndices,
                                         int count,
                                         double* outDerivatives,
                                         double* outSumDerivatives,
                                         double* outSumSquaredDerivatives) override;

            int calculateCrossProducts(const int* postBufferIndices,
                                       const int* preBufferIndices,
                                       const int* categoryRatesIndices,
                                       const int* categoryWeightsIndices,
                                       const double* edgeLengths,
                                       int count,
                                       double* outSumDerivatives,
                                       double* outSumSquaredDerivatives) override;

            int calculateAdjointCrossProducts(const int* postBufferIndices,
                                              const int* preBufferIndices,
                                              const int* eigenIndices,
                                              const int* categoryRatesIndices,
                                              const int* categoryWeightsIndices,
                                              const int rootPostOrderIndex,
                                              const int stateFrequenciesIndex,
                                              int count,
                                              double* outSumDerivatives,
                                              double* outSumSquaredDerivatives,
                                              const int* postScaleIndices = nullptr,
                                              const int* preScaleIndices = nullptr,
                                              const int cumulativeScaleIndex = BEAGLE_OP_NONE) override;

            int calculateAdjointDerivative(const BeagleBranchOperation* operations,
                                           const int categoryRatesIndex,
                                           const int categoryWeightsIndex,
                                           const int rootPostOrderIndex,
                                           const int stateFrequenciesIndex,
                                           const int operationCount,
                                           double* outSumDerivatives,
                                           double* outSumSquaredDerivatives) override;

        protected:
            bool denseMatricesOnDemand() const override { return true; }

            int getPaddedPatternsModulus() override;

            void rescalePartials(REALTYPE* destP,
                                 REALTYPE* scaleFactors,
                                 REALTYPE* cumulativeScaleFactors,
                                 const int fillWithOnes) override;

            void rescalePartialsByPartition(REALTYPE* destP,
                                            REALTYPE* scaleFactors,
                                            REALTYPE* cumulativeScaleFactors,
                                            const int fillWithOnes,
                                            const int partitionIndex) override;

            void autoRescalePartials(REALTYPE* destP,
                                     signed short* scaleFactors) override;

            EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>* createEigenDecomposition(
                    int decompositionCount,
                    int stateCount,
                    int categoryCount,
                    long flags) override;

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

            int reorderPatternsByPartition() override;

            int calcRootLogLikelihoods(const int bufferIndex,
                                       const int categoryWeightsIndex,
                                       const int stateFrequenciesIndex,
                                       const int scaleBufferIndex,
                                       double* outSumLogLikelihood) override;

            int calcRootLogLikelihoodsPerCategory(const int bufferIndex,
                                                  const int stateFrequenciesIndex,
                                                  const int scaleBufferIndex,
                                                  double* outSumLogLikelihood) override;

            void calcRootLogLikelihoodsByPartition(const int* bufferIndices,
                                                   const int* categoryWeightsIndices,
                                                   const int* stateFrequenciesIndices,
                                                   const int* cumulativeScaleIndices,
                                                   const int* partitionIndices,
                                                   int partitionCount,
                                                   double* outSumLogLikelihoodByPartition) override;

            int calcRootLogLikelihoodsMulti(const int* bufferIndices,
                                            const int* categoryWeightsIndices,
                                            const int* stateFrequenciesIndices,
                                            const int* scaleBufferIndices,
                                            int count,
                                            double* outSumLogLikelihood) override;

            double rootLogLikelihoodRange(const int bufferIndex,
                                          const int categoryWeightsIndex,
                                          const int stateFrequenciesIndex,
                                          const int scaleBufferIndex,
                                          int startPattern,
                                          int endPattern);
        };

        BEAGLE_CPU_FACTORY_TEMPLATE
        class BeagleCPUHierarchicalImplFactory : public BeagleImplFactory {
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
                                           int resourceNumber,
                                           int pluginResourceNumber,
                                           long preferenceFlags,
                                           long requirementFlags,
                                           int* errorCode);

            virtual const char* getName();
            virtual const long getFlags();
        };

    }
}

#include "libhmsbeagle/CPU/BeagleCPUHierarchicalImpl.hpp"

#endif
