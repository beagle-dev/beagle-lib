/*
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 *
 * @brief GPU kernel launcher
 *
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#ifndef __KernelLauncher__
#define __KernelLauncher__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/GPU/GPUImplDefs.h"
#include "libhmsbeagle/GPU/GPUInterface.h"

#ifdef CUDA
	namespace cuda_device {
#else
	namespace opencl_device {
#endif

class KernelLauncher {

private:
    GPUInterface* gpu;

    GPUFunction fMatrixConvolution;
    GPUFunction fMatrixTranspose;
    GPUFunction fMatrixMulADBMulti;
    GPUFunction fMatrixMulADB;
    GPUFunction fMatrixMulADBFirstDeriv;
    GPUFunction fMatrixMulADBSecondDeriv;

    GPUFunction fPartialsPartialsByPatternBlockCoherentMulti;
    GPUFunction fPartialsPartialsByPatternBlockCoherentPartition;
    GPUFunction fPartialsPartialsByPatternBlockCoherent;
    GPUFunction fPartialsPartialsByPatternBlockFixedScalingMulti;
    GPUFunction fPartialsPartialsByPatternBlockFixedScalingPartition;
    GPUFunction fPartialsPartialsByPatternBlockFixedScaling;
    GPUFunction fPartialsPartialsByPatternBlockAutoScaling;
    GPUFunction fPartialsPartialsByPatternBlockCheckScaling;
    GPUFunction fPartialsPartialsByPatternBlockFixedCheckScaling;
	GPUFunction fPartialsPartialsEdgeFirstDerivatives;
	GPUFunction fPartialsStatesEdgeFirstDerivatives;
	GPUFunction fPartialsPartialsCrossProducts;
	GPUFunction fPartialsStatesCrossProducts;
	GPUFunction fMultipleNodeSiteReduction;
	GPUFunction fMultipleNodeSiteSquaredReduction;
	GPUFunction fPartialsPartialsGrowing;
	GPUFunction fPartialsStatesGrowing;
    GPUFunction fStatesPartialsByPatternBlockCoherentMulti;
    GPUFunction fStatesPartialsByPatternBlockCoherentPartition;
    GPUFunction fStatesPartialsByPatternBlockCoherent;
    GPUFunction fStatesPartialsByPatternBlockFixedScalingMulti;
    GPUFunction fStatesPartialsByPatternBlockFixedScalingPartition;
    GPUFunction fStatesPartialsByPatternBlockFixedScaling;
    GPUFunction fStatesStatesByPatternBlockCoherentMulti;
    GPUFunction fStatesStatesByPatternBlockCoherentPartition;
    GPUFunction fStatesStatesByPatternBlockCoherent;
    GPUFunction fStatesStatesByPatternBlockFixedScalingMulti;
    GPUFunction fStatesStatesByPatternBlockFixedScalingPartition;
    GPUFunction fStatesStatesByPatternBlockFixedScaling;
    GPUFunction fPartialsPartialsEdgeLikelihoods;
    GPUFunction fPartialsPartialsEdgeLikelihoodsByPartition;
    GPUFunction fPartialsPartialsEdgeLikelihoodsSecondDeriv;
    GPUFunction fStatesPartialsEdgeLikelihoods;
    GPUFunction fStatesPartialsEdgeLikelihoodsByPartition;
    GPUFunction fStatesPartialsEdgeLikelihoodsSecondDeriv;

    GPUFunction fIntegrateLikelihoodsDynamicScaling;
    GPUFunction fIntegrateLikelihoodsDynamicScalingPartition;
    GPUFunction fIntegrateLikelihoodsDynamicScalingSecondDeriv;
    GPUFunction fAccumulateFactorsDynamicScaling;
    GPUFunction fAccumulateFactorsDynamicScalingByPartition;
    GPUFunction fAccumulateFactorsAutoScaling;
    GPUFunction fRemoveFactorsDynamicScaling;
    GPUFunction fRemoveFactorsDynamicScalingByPartition;
    GPUFunction fResetFactorsDynamicScalingByPartition;
    GPUFunction fPartialsDynamicScaling;
    GPUFunction fPartialsDynamicScalingByPartition;
    GPUFunction fPartialsDynamicScalingAccumulate;
    GPUFunction fPartialsDynamicScalingAccumulateByPartition;
    GPUFunction fPartialsDynamicScalingAccumulateDifference;
    GPUFunction fPartialsDynamicScalingAccumulateReciprocal;
    GPUFunction fPartialsDynamicScalingSlow;
    GPUFunction fIntegrateLikelihoods;
    GPUFunction fIntegrateLikelihoodsPartition;
    GPUFunction fIntegrateLikelihoodsSecondDeriv;
	  GPUFunction fIntegrateLikelihoodsMulti;
	  GPUFunction fIntegrateLikelihoodsFixedScaleMulti;
    GPUFunction fIntegrateLikelihoodsAutoScaling;

    GPUFunction fSumSites1;
    GPUFunction fSumSites1Partition;
    GPUFunction fSumSites2;
    GPUFunction fSumSites3;

	GPUFunction fInnerBastaPartialsCoalescent;
	GPUFunction fReduceWithinIntervalMerged;
	GPUFunction fReduceWithinIntervalMergedSlab;
	GPUFunction fReduceAcrossInterval;
    GPUFunction fReorderPatterns;

	GPUFunction fComputeHazardAdjoints;
	GPUFunction fAdjointBastaPartials;
	GPUFunction fTiledBatchGemmABt;
	GPUFunction fTiledBatchGemmAtB;
	GPUFunction fReduceMatrices;
	GPUFunction fApplyComplexLoewnerInPlace;
	GPUFunction fLoewnerExpPrecompute;
	GPUFunction fApplyComplexLoewnerInPlacePrecomp;
	GPUFunction fTiledSingleGemmAtB;
	GPUFunction fTiledSingleGemmABt;
	GPUFunction fAccumulateMatrixAdjoints;
	GPUFunction fAccumulateMatrixAdjointsSlabEigen;


	GPUFunction fProjectPartialsToEigenbasis;
	GPUFunction fProjectHazardsToEigen;
	GPUFunction fCoalescentSlab;

	GPUFunction fAdjointBranchSlabLocal;
	GPUFunction fAdjointBranchSlabSpinePerBranch;
	GPUFunction fAdjointBranchSlabApply;

	GPUFunction fForwardBranchSlab;
	GPUFunction fForwardCoalescentSlab;

    Dim3Int bgTransitionProbabilitiesBlock;
    Dim3Int bgTransitionProbabilitiesGrid;
    Dim3Int bgPeelingBlock;
    Dim3Int bgPeelingGrid;
	Dim3Int bgDerivativeBlock;
	Dim3Int bgDerivativeGrid;
    Dim3Int bgLikelihoodBlock;
    Dim3Int bgLikelihoodGrid;
    Dim3Int bgAccumulateBlock;
    Dim3Int bgAccumulateGrid;
    Dim3Int bgScaleBlock;
    Dim3Int bgScaleGrid;
    Dim3Int bgSumSitesBlock;
    Dim3Int bgSumSitesGrid;
    Dim3Int bgReorderPatternsBlock;
    Dim3Int bgReorderPatternsGrid;
	Dim3Int bgMultiNodeSumBlock;
	Dim3Int bgMultiNodeSumGrid;
	Dim3Int bgCrossProductBlock;
	Dim3Int bgCrossProductGrid;
	Dim3Int bgBastaPeelingBlock;
	Dim3Int bgBastaPeelingGrid;
	Dim3Int bgBastaReductionBlock;
	Dim3Int bgBastaReductionGrid;
	Dim3Int bgBastaPreBlock;
	Dim3Int bgBastaPreGrid;
	Dim3Int bgBastaSumBlock;
	Dim3Int bgBastaSumGrid;
	Dim3Int bgAdjointBastaPeelingBlock;
	Dim3Int bgAdjointBastaPeelingGrid;
	Dim3Int bgHazardAdjointBlock;
	Dim3Int bgEigenTileBlock;
	Dim3Int bgReduceBlock;
	Dim3Int bgCoalescentSlabBlock;
	Dim3Int bgPartialProjectBlock;
	Dim3Int bgHazardEigenBlock;
	Dim3Int bgSlabSegLocalBlock;
	Dim3Int bgSlabSegApplyBlock;


    unsigned int kPaddedStateCount;
    unsigned int kCategoryCount;
    unsigned int kPatternCount;
    unsigned int kUnpaddedPatternCount;
    unsigned int kPatternBlockSize;
    unsigned int kMatrixBlockSize;
    unsigned int kSlowReweighing;
    unsigned int kMultiplyBlockSize;
	unsigned int kSumIntervalBlockSize;
	unsigned int kSlabOpsPerBlock;
	unsigned int kSumAcrossBlockSize;
    unsigned int kSumSitesBlockSize;
    long kFlags;
    bool kCPUImplementation;
    bool kAppleCPUImplementation;


public:
    KernelLauncher(GPUInterface* inGpu);

    ~KernelLauncher();

    // void SetupPartitioningKernelGrid(unsigned int partitionBlockCount);

// Kernel links

    void ReorderPatterns(GPUPtr dPartials,
                         GPUPtr dStates,
                         GPUPtr dStatesSort,
                         GPUPtr dTipOffsets,
                         GPUPtr dTipTypes,
                         GPUPtr dPatternsNewOrder,
                         GPUPtr dPatternWeights,
                         GPUPtr dPatternWeightsSort,
                         int    patternCount,
                         int    paddedPatternCount,
                         int    tipCount);

    void ConvolveTransitionMatrices(GPUPtr dMatrices,
                          GPUPtr dPtrQueue,
                          unsigned int totalMatrixCount);

    void TransposeTransitionMatrices(GPUPtr dMatrices,
    								 GPUPtr dPtrQueue,
    								 unsigned int totalMatrixCount);

    void GetTransitionProbabilitiesSquareMulti(GPUPtr dMatrices,
                                               GPUPtr dPtrQueue,
                                               GPUPtr dEvec,
                                               GPUPtr dIevc,
                                               GPUPtr dEigenValues,
                                               GPUPtr distanceQueue,
                                               unsigned int totalMatrix);

    void GetTransitionProbabilitiesSquare(GPUPtr dMatrices,
                                          GPUPtr dPtrQueue,
                                          GPUPtr dEvec,
                                          GPUPtr dIevc,
                                          GPUPtr dEigenValues,
                                          GPUPtr distanceQueue,
                                          unsigned int totalMatrix);

    void GetTransitionProbabilitiesSquareFirstDeriv(GPUPtr dMatrices,
                                                    GPUPtr dPtrQueue,
                                                     GPUPtr dEvec,
                                                     GPUPtr dIevc,
                                                     GPUPtr dEigenValues,
                                                     GPUPtr distanceQueue,
                                                     unsigned int totalMatrix);

    void GetTransitionProbabilitiesSquareSecondDeriv(GPUPtr dMatrices,
                                                     GPUPtr dPtrQueue,
                                          GPUPtr dEvec,
                                          GPUPtr dIevc,
                                          GPUPtr dEigenValues,
                                          GPUPtr distanceQueue,
                                          unsigned int totalMatrix);

    void PartialsPartialsPruningDynamicCheckScaling(GPUPtr partials1,
                                                    GPUPtr partials2,
                                                    GPUPtr partials3,
                                                    GPUPtr matrices1,
                                                    GPUPtr matrices2,
                                                    int writeScalingIndex,
                                                    int readScalingIndex,
                                                    int cumulativeScalingIndex,
                                                    GPUPtr* dScalingFactors,
                                                    GPUPtr* dScalingFactorsMaster,
                                                    unsigned int patternCount,
                                                    unsigned int categoryCount,
                                                    int doRescaling,
                                                    int* hRescalingTrigger,
                                                    GPUPtr dRescalingTrigger,
                                                    int sizeReal);

    void PartialsPartialsPruningMulti(GPUPtr partials,
                                      GPUPtr matrices,
                                      GPUPtr scalingFactors,
                                      GPUPtr ptrOffsets,
                                      unsigned int patternCount,
                                      int gridStartOp,
                                      int gridSize,
                                      int doRescaling);

    void PartialsPartialsPruningDynamicScaling(GPUPtr partials1,
                                               GPUPtr partials2,
                                               GPUPtr partials3,
                                               GPUPtr matrices1,
                                               GPUPtr matrices2,
                                               GPUPtr scalingFactors,
                                               GPUPtr cumulativeScaling,
                                               unsigned int startPattern,
                                               unsigned int endPattern,
                                               unsigned int patternCount,
                                               unsigned int categoryCount,
                                               int doRescaling,
                                               int streamIndex,
                                               int waitIndex);

    void StatesPartialsPruningMulti(GPUPtr states,
                                    GPUPtr partials,
                                    GPUPtr matrices,
                                    GPUPtr scalingFactors,
                                    GPUPtr ptrOffsets,
                                    unsigned int patternCount,
                                    int gridStartOp,
                                    int gridSize,
                                    int doRescaling);

    void StatesPartialsPruningDynamicScaling(GPUPtr states1,
                                             GPUPtr partials2,
                                             GPUPtr partials3,
                                             GPUPtr matrices1,
                                             GPUPtr matrices2,
                                             GPUPtr scalingFactors,
                                             GPUPtr cumulativeScaling,
                                             unsigned int startPattern,
                                             unsigned int endPattern,
                                             unsigned int patternCount,
                                             unsigned int categoryCount,
                                             int doRescaling,
                                             int streamIndex,
                                             int waitIndex);

    void StatesStatesPruningMulti(GPUPtr states,
                                  GPUPtr partials,
                                  GPUPtr matrices,
                                  GPUPtr scalingFactors,
                                  GPUPtr ptrOffsets,
                                  unsigned int patternCount,
                                  int gridStartOp,
                                  int gridSize,
                                  int doRescaling);

    void StatesStatesPruningDynamicScaling(GPUPtr states1,
                                           GPUPtr states2,
                                           GPUPtr partials3,
                                           GPUPtr matrices1,
                                           GPUPtr matrices2,
                                           GPUPtr scalingFactors,
                                           GPUPtr cumulativeScaling,
                                           unsigned int startPattern,
                                           unsigned int endPattern,
                                           unsigned int patternCount,
                                           unsigned int categoryCount,
                                           int doRescaling,
                                           int streamIndex,
                                           int waitIndex);

	void PartialsStatesGrowing(GPUPtr partials1,
                               GPUPtr partials2,
                               GPUPtr partials3,
                               GPUPtr matrices1,
                               GPUPtr matrices2,
                               unsigned int patternCount,
                               unsigned int categoryCount,
                               int sizeReal);

    void PartialsPartialsGrowing(GPUPtr partials1,
                                 GPUPtr partials2,
                                 GPUPtr partials3,
                                 GPUPtr matrices1,
                                 GPUPtr matrices2,
                                 unsigned int patternCount,
                                 unsigned int categoryCount,
                                 int sizeReal);

	void PartialsStatesEdgeFirstDerivatives(GPUPtr out,
											  GPUPtr states0,
											  GPUPtr partials0,
											  GPUPtr matrices0,
											  GPUPtr weights,
											  GPUPtr instructions,
											  unsigned int instructionOffset,
											  unsigned int nodeCount,
											  unsigned int patternCount,
											  unsigned int categoryCount,
											  bool synchronize);

	void PartialsPartialsEdgeFirstDerivatives(GPUPtr out,
											  GPUPtr partials0,
											  GPUPtr matrices0,
											  GPUPtr weights,
											  GPUPtr instructions,
											  unsigned int instructionOffset,
											  unsigned int nodeCount,
											  unsigned int patternCount,
											  unsigned int categoryCount,
											  bool synchronize);

	void PartialsStatesCrossProducts(GPUPtr out,
									 GPUPtr states0,
                                                   GPUPtr partials,
                                                   GPUPtr lengths,
                                                   GPUPtr instructions,
                                                   GPUPtr categoryWeights,
                                                   GPUPtr patternWeights,
                                                   unsigned int instructionOffset,
                                                   unsigned int nodeCount,
                                                   unsigned int rateOffset,
                                                   unsigned int patternCount,
                                                   unsigned int categoryCount,
                                                   bool synchronize,
                                                   unsigned int nodeBlocks,
                                                   unsigned int patterBlocks,
                                                   unsigned int missingState);

	void PartialsPartialsCrossProducts(GPUPtr out,
                                                   GPUPtr partials,
                                                   GPUPtr lengths,
                                                   GPUPtr instructions,
                                                   GPUPtr categoryWeights,
                                                   GPUPtr patternWeights,
                                                   unsigned int instructionOffset,
                                                   unsigned int nodeCount,
                                                   unsigned int rateOffset,
                                                   unsigned int patternCount,
                                                   unsigned int categoryCount,
                                                   bool synchronize,
                                                   unsigned int nodeBlocks,
                                                   unsigned int patterBlocks);

	void MultipleNodeSiteReduction(GPUPtr outSiteValues,
								   GPUPtr inSiteValues,
								   GPUPtr weights,
								   unsigned int outOffset,
								   unsigned int stride,
								   unsigned int count);

	void MultipleNodeSiteSquaredReduction(GPUPtr outSiteValues,
								          GPUPtr inSiteValues,
								          GPUPtr weights,
								          unsigned int outOffset,
								          unsigned int stride,
								          unsigned int count);

    void IntegrateLikelihoodsDynamicScaling(GPUPtr dResult,
                                            GPUPtr dRootPartials,
                                            GPUPtr dWeights,
                                            GPUPtr dFrequencies,
                                            GPUPtr dRootScalingFactors,
                                            unsigned int patternCount,
                                            unsigned int categoryCount);

    void IntegrateLikelihoodsDynamicScalingPartition(GPUPtr dResult,
                                                     GPUPtr dRootPartials,
                                                     GPUPtr dWeights,
                                                     GPUPtr dFrequencies,
                                                     GPUPtr dRootScalingFactors,
                                                     GPUPtr dPtrOffsets,
                                                     unsigned int patternCount,
                                                     unsigned int categoryCount,
                                                     int gridSize);

    void IntegrateLikelihoodsAutoScaling(GPUPtr dResult,
                                            GPUPtr dRootPartials,
                                            GPUPtr dWeights,
                                            GPUPtr dFrequencies,
                                            GPUPtr dRootScalingFactors,
                                            unsigned int patternCount,
                                            unsigned int categoryCount);

    void IntegrateLikelihoodsDynamicScalingSecondDeriv(GPUPtr dResult,
                                                       GPUPtr dFirstDerivResult,
                                                       GPUPtr dSecondDerivResult,
                                                       GPUPtr dRootPartials,
                                                       GPUPtr dRootFirstDeriv,
                                                       GPUPtr dRootSecondDeriv,
                                                       GPUPtr dWeights,
                                                       GPUPtr dFrequencies,
                                                       GPUPtr dRootScalingFactors,
                                                       unsigned int patternCount,
                                                       unsigned int categoryCount);

    void PartialsPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
                                         GPUPtr dParentPartials,
                                         GPUPtr dChildParials,
                                         GPUPtr dTransMatrix,
                                         unsigned int patternCount,
                                         unsigned int categoryCount);

    void PartialsPartialsEdgeLikelihoodsByPartition(GPUPtr dPartialsTmp,
                                                    GPUPtr dPartialsOrigin,
                                                    GPUPtr dMatricesOrigin,
                                                    GPUPtr dPtrOffsets,
                                                    unsigned int patternCount,
                                                    int gridSize);

    void PartialsPartialsEdgeLikelihoodsSecondDeriv(GPUPtr dPartialsTmp,
                                                    GPUPtr dFirstDerivTmp,
                                                    GPUPtr dSecondDerivTmp,
                                                    GPUPtr dParentPartials,
                                                    GPUPtr dChildParials,
                                                    GPUPtr dTransMatrix,
                                                    GPUPtr dFirstDerivMatrix,
                                                    GPUPtr dSecondDerivMatrix,
                                                    unsigned int patternCount,
                                                    unsigned int categoryCount);


    void StatesPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
                                       GPUPtr dParentPartials,
                                       GPUPtr dChildStates,
                                       GPUPtr dTransMatrix,
                                       unsigned int patternCount,
                                       unsigned int categoryCount);

    void StatesPartialsEdgeLikelihoodsByPartition(GPUPtr dPartialsTmp,
                                                  GPUPtr dPartialsOrigin,
                                                  GPUPtr dStatesOrigin,
                                                  GPUPtr dMatricesOrigin,
                                                  GPUPtr dPtrOffsets,
                                                  unsigned int patternCount,
                                                  int gridSize);

    void StatesPartialsEdgeLikelihoodsSecondDeriv(GPUPtr dPartialsTmp,
                                                  GPUPtr dFirstDerivTmp,
                                                  GPUPtr dSecondDerivTmp,
                                                  GPUPtr dParentPartials,
                                                  GPUPtr dChildStates,
                                                  GPUPtr dTransMatrix,
                                                  GPUPtr dFirstDerivMatrix,
                                                  GPUPtr dSecondDerivMatrix,
                                                  unsigned int patternCount,
                                                  unsigned int categoryCount);

    void AccumulateFactorsDynamicScaling(GPUPtr dScalingFactors,
                                         GPUPtr dNodePtrQueue,
                                         GPUPtr dRootScalingFactors,
                                         unsigned int nodeCount,
                                         unsigned int patternCount);

    void AccumulateFactorsDynamicScalingByPartition(GPUPtr dScalingFactors,
                                                    GPUPtr dNodePtrQueue,
                                                    GPUPtr dRootScalingFactors,
                                                    unsigned int nodeCount,
                                                    int startPattern,
                                                    int endPattern);

    void AccumulateFactorsAutoScaling(GPUPtr dScalingFactors,
                                      GPUPtr dNodePtrQueue,
                                      GPUPtr dRootScalingFactors,
                                      unsigned int nodeCount,
                                      unsigned int patternCount,
                                      unsigned int scaleBufferSize);

    void RemoveFactorsDynamicScaling(GPUPtr dScalingFactors,
                                     GPUPtr dNodePtrQueue,
                                     GPUPtr dRootScalingFactors,
                                     unsigned int nodeCount,
                                     unsigned int patternCount);

    void RemoveFactorsDynamicScalingByPartition(GPUPtr dScalingFactors,
                                                GPUPtr dNodePtrQueue,
                                                GPUPtr dRootScalingFactors,
                                                unsigned int nodeCount,
                                                int startPattern,
                                                int endPattern);

    void ResetFactorsDynamicScalingByPartition(GPUPtr dScalingFactors,
                                               int startPattern,
                                               int endPattern);

    void RescalePartials(GPUPtr partials3,
                         GPUPtr scalingFactors,
                         GPUPtr cumulativeScaling,
                         unsigned int patternCount,
                         unsigned int categoryCount,
                         unsigned int fillWithOnes,
                         int streamIndex,
                         int waitIndex);

    void RescalePartialsByPartition(GPUPtr partials3,
                                    GPUPtr scalingFactors,
                                    GPUPtr cumulativeScaling,
                                    unsigned int patternCount,
                                    unsigned int categoryCount,
                                    unsigned int fillWithOnes,
                                    int streamIndex,
                                    int waitIndex,
                                    int startPattern,
                                    int endPattern);

    void IntegrateLikelihoods(GPUPtr dResult,
                              GPUPtr dRootPartials,
                              GPUPtr dWeights,
                              GPUPtr dFrequencies,
                              unsigned int patternCount,
                              unsigned int categoryCount);

    void IntegrateLikelihoodsPartition(GPUPtr dResult,
                                       GPUPtr dRootPartials,
                                       GPUPtr dWeights,
                                       GPUPtr dFrequencies,
                                       GPUPtr dPtrOffsets,
                                       unsigned int patternCount,
                                       unsigned int categoryCount,
                                       int gridSize);

    void IntegrateLikelihoodsSecondDeriv(GPUPtr dResult,
                                         GPUPtr dFirstDerivResult,
                                         GPUPtr dSecondDerivResult,
                                         GPUPtr dRootPartials,
                                         GPUPtr dRootFirstDeriv,
                                         GPUPtr dRootSecondDeriv,
                                         GPUPtr dWeights,
                                         GPUPtr dFrequencies,
                                         unsigned int patternCount,
                                         unsigned int categoryCount);

	void IntegrateLikelihoodsMulti(GPUPtr dResult,
								   GPUPtr dRootPartials,
								   GPUPtr dWeights,
								   GPUPtr dFrequencies,
								   unsigned int patternCount,
								   unsigned int categoryCount,
								   unsigned int takeLog);

	void IntegrateLikelihoodsFixedScaleMulti(GPUPtr dResult,
											 GPUPtr dRootPartials,
											 GPUPtr dWeights,
											 GPUPtr dFrequencies,
                                             GPUPtr dScalingFactors,
											 GPUPtr dPtrQueue,
											 GPUPtr dMaxScalingFactors,
											 GPUPtr dIndexMaxScalingFactors,
											 unsigned int patternCount,
											 unsigned int categoryCount,
											 unsigned int subsetCount,
											 unsigned int subsetIndex);

    void SumSites1(GPUPtr dArray1,
                  GPUPtr dSum1,
                  GPUPtr dPatternWeights,
                  unsigned int patternCount);

    void SumSites1Partition(GPUPtr dArray1,
                            GPUPtr dSum1,
                            GPUPtr dPatternWeights,
                            int startPattern,
                            int endPattern,
                            int blockCount);

    void SumSites2(GPUPtr dArray1,
                  GPUPtr dSum1,
                  GPUPtr dArray2,
                  GPUPtr dSum2,
                  GPUPtr dPatternWeights,
                  unsigned int patternCount);

    void SumSites3(GPUPtr dArray1,
                  GPUPtr dSum1,
                  GPUPtr dArray2,
                  GPUPtr dSum2,
                  GPUPtr dArray3,
                  GPUPtr dSum3,
                  GPUPtr dPatternWeights,
                  unsigned int patternCount);

	void InnerBastaPartialsCoalescent(GPUPtr partials, GPUPtr matrices,
	GPUPtr operations, GPUPtr sizes, GPUPtr coalescent, unsigned int start, unsigned int numOps, unsigned int patternCount);

    void reduceAcrossIntervals(GPUPtr dBastaMemory, GPUPtr distance, GPUPtr dLogL, GPUPtr sizes, GPUPtr coalescent, unsigned int intervalNUmber, unsigned int kCoalescentBufferLength);

    void SetupKernelBlocksAndGrids();
    void reduceWithinIntervalMerged(GPUPtr operations, GPUPtr partials, GPUPtr dBastaMemory, unsigned int numOps, unsigned int start, unsigned int end, unsigned int numBlocks, unsigned int kCoalescentBufferLength);


    void reduceWithinIntervalMergedSlab(GPUPtr reduceOps, GPUPtr partials,
                                        GPUPtr dBastaMemory,
                                        unsigned int start, unsigned int end,
                                        unsigned int numBlocks,
                                        unsigned int kCoalescentBufferLength);

    void ComputeHazardAdjoints(GPUPtr dBastaMemory, GPUPtr intervalLengths,
        GPUPtr intervalNumbers, GPUPtr sizes,
        GPUPtr dHazardAdjoints, GPUPtr dPopSizeGrad,
        unsigned int numIntervals, unsigned int kCoalescentBufferLength);


    void AdjointBastaPartials(GPUPtr partials, GPUPtr partialAdj,
        GPUPtr matrices,
        GPUPtr scratchYBar, GPUPtr scratchX,
        GPUPtr coalRightYBar, GPUPtr coalRightX,
        GPUPtr operations, GPUPtr sizes, GPUPtr coalescent,
        GPUPtr dHazardAdjoints, GPUPtr dPopSizeGrad,
        unsigned int interval, unsigned int start,
        unsigned int totalPatterns, unsigned int matTransIndex,
        unsigned int stateCount, int coalOp);

    void AccumulateMatrixAdjoints(GPUPtr scratchYBar, GPUPtr scratchX,
        GPUPtr coalRightYBar, GPUPtr coalRightX,
        GPUPtr intervalStarts, GPUPtr matTransIndices, GPUPtr coalOps,
        GPUPtr matrixAdj, unsigned int intervalCount);

    void AccumulateMatrixAdjointsSlabEigen(
        GPUPtr scratchYBar, GPUPtr partialsTilde,
        GPUPtr opInBufOff,
        GPUPtr intervalOpStart, GPUPtr intervalOpList,
        GPUPtr matrixAdj, unsigned int intervalCount);

    void TiledBatchGemmABt(GPUPtr A, GPUPtr B, GPUPtr C, unsigned int matrixCount);

    void TiledBatchGemmAtB(GPUPtr A, GPUPtr B, GPUPtr C, unsigned int matrixCount);

    void ReduceMatrices(GPUPtr src, GPUPtr dst, unsigned int matrixCount);

    void ApplyComplexLoewnerInPlace(GPUPtr transformed, GPUPtr eigenValues,
        GPUPtr branchLengths, GPUPtr blockStarts, GPUPtr blockDims,
        unsigned int matrixCount, unsigned int stateCount, unsigned int numBlocks);

    void LoewnerExpPrecompute(GPUPtr eigenValues, GPUPtr branchLengths,
        GPUPtr blockStarts, GPUPtr blockDims, GPUPtr loewnerExp,
        unsigned int matrixCount, unsigned int stateCount, unsigned int numBlocks);

    void ApplyComplexLoewnerInPlacePrecomp(GPUPtr transformed, GPUPtr eigenValues,
        GPUPtr branchLengths, GPUPtr blockStarts, GPUPtr blockDims, GPUPtr loewnerExp,
        unsigned int matrixCount, unsigned int stateCount, unsigned int numBlocks);

    void TiledSingleGemmAtB(GPUPtr A, GPUPtr B, GPUPtr C);

    void TiledSingleGemmABt(GPUPtr A, GPUPtr B, GPUPtr C);


    void ProjectPartialsToEigenbasis(GPUPtr partials, GPUPtr inverseEvec,
        GPUPtr partialsTilde, GPUPtr bufferIndices,
        unsigned int bufferCount, unsigned int stateCount, unsigned int stride);


    void ProjectHazardsToEigen(GPUPtr partials,
                                GPUPtr hazardAdjoints,
                                GPUPtr evecT,
                                GPUPtr opUOff, GPUPtr opKIn,
                                GPUPtr opKAcc, GPUPtr opHasAcc,
                                GPUPtr hazardEigenPerOp,
                                unsigned int opCount,
                                unsigned int stateCount);

    void CoalescentSlab(GPUPtr partials, GPUPtr partialAdj,
        GPUPtr sizes, GPUPtr coalescent, GPUPtr popSizeGrad,
        GPUPtr coalDestBufs, GPUPtr coalLeftAccBufs, GPUPtr coalRightAccBufs,
        GPUPtr coalIntervals,
        unsigned int coalSlabOffset,
        unsigned int coalCount, unsigned int stateCount, unsigned int stride);

    void AdjointBranchSlabLocal(
        GPUPtr partials, GPUPtr partialAdj, GPUPtr scratchYBar,
        GPUPtr evecT,
        GPUPtr eigenValues, GPUPtr blockStarts, GPUPtr blockDims,
        GPUPtr hazardAdjoints, GPUPtr hazardEigenPerOp,
        GPUPtr slabBlockBranchIdx, GPUPtr slabBlockChunkStart,
        GPUPtr slabBlockChunkLen,  GPUPtr slabBlockChunkIdx,
        GPUPtr branchKb, GPUPtr branchKTop,
        GPUPtr branchTopBuf, GPUPtr branchOpFirst,
        GPUPtr branchTimeStart, GPUPtr branchT,
        GPUPtr slabCarryOut, GPUPtr slabAStash, GPUPtr slabYBottomEigen,
        GPUPtr slabAlpha,
        int useAnchor,
        unsigned int blockBase, unsigned int numBlocks,
        unsigned int numEvBlocks,
        unsigned int stateCount, unsigned int stride);

    void AdjointBranchSlabSpinePerBranch(
        GPUPtr slabCarryOut, GPUPtr slabCarryPrefix,
        GPUPtr slabAlpha, GPUPtr blockStarts, GPUPtr blockDims,
        GPUPtr branchSlabList, GPUPtr branchKb, GPUPtr branchFirstBlock,
        int useAnchor,
        unsigned int branchSlabOffset, unsigned int numBranches,
        unsigned int opsPerBlock, unsigned int stateCount);

    void AdjointBranchSlabApply(
        GPUPtr partialAdj, GPUPtr scratchYBar,
        GPUPtr inverseEvecT,
        GPUPtr slabAStash, GPUPtr slabCarryPrefix, GPUPtr slabYBottomEigen,
        GPUPtr slabBlockBranchIdx, GPUPtr slabBlockChunkStart,
        GPUPtr slabBlockChunkLen,  GPUPtr slabBlockChunkIdx,
        GPUPtr branchKb, GPUPtr branchBotBuf, GPUPtr branchOpFirst,
        GPUPtr blockStarts, GPUPtr blockDims,
        unsigned int blockBase, unsigned int numBlocks,
        unsigned int numEvBlocks, unsigned int opsPerBlock,
        unsigned int stateCount, unsigned int stride);


    void ForwardBranchSlab(
        GPUPtr partials, GPUPtr partialsTilde,
        GPUPtr evecT,
        GPUPtr eigenValues, GPUPtr blockStarts, GPUPtr blockDims,
        GPUPtr slabBlockBranchIdx, GPUPtr slabBlockChunkStart,
        GPUPtr slabBlockChunkLen,  GPUPtr slabBlockChunkIdx,
        GPUPtr branchKb, GPUPtr branchTopBuf, GPUPtr branchBotBuf,
        GPUPtr branchOpFirst, GPUPtr branchTimeStart, GPUPtr branchT,
        GPUPtr opInBufOff,
        unsigned int blockBase, unsigned int numBlocks,
        unsigned int numEvBlocks,
        unsigned int stateCount, unsigned int stride);

    void ForwardCoalescentSlab(
        GPUPtr partials, GPUPtr partialsTilde,
        GPUPtr inverseEvecT,
        GPUPtr sizes, GPUPtr coalescent,
        GPUPtr coalDestBufs, GPUPtr coalLeftAccBufs, GPUPtr coalRightAccBufs,
        GPUPtr coalIntervals,
        unsigned int coalSlabOffset, unsigned int coalCount,
        unsigned int stateCount, unsigned int stride);

protected:
    void LoadKernels();

};

}; // namespace

#endif // __KernelLauncher__
