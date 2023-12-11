/*
 *  BeagleCPUImpl.h
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
 * @author Andrew Rambaut
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#ifndef __BeagleCPUImpl__
#define __BeagleCPUImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/BeagleImpl.h"
#include "libhmsbeagle/CPU/Precision.h"
#include "libhmsbeagle/CPU/EigenDecomposition.h"

#include <vector>
#include <thread>
#include <future>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <functional>

#define BEAGLE_CPU_GENERIC	REALTYPE, T_PAD, P_PAD
#define BEAGLE_CPU_TEMPLATE	template <typename REALTYPE, int T_PAD, int P_PAD>

#define BEAGLE_CPU_FACTORY_GENERIC	REALTYPE
#define BEAGLE_CPU_FACTORY_TEMPLATE	template <typename REALTYPE>

#define T_PAD_DEFAULT   1   // Pad transition matrix rows with an extra 1.0 for ambiguous characters
#define P_PAD_DEFAULT   0   // No partials padding necessary for non-SSE implementations

//  TODO: assess following cut-offs dynamically
#define BEAGLE_CPU_ASYNC_HW_THREAD_COUNT_THRESHOLD     16  // CPU category threshold
#define BEAGLE_CPU_ASYNC_MIN_PATTERN_COUNT_LOW        256  // do not use CPU auto-threading for problems with fewer patterns on CPUs with many cores
#define BEAGLE_CPU_ASYNC_MIN_PATTERN_COUNT_HIGH       768  // do not use CPU auto-threading for problems with fewer patterns on CPUs with few cores
#define BEAGLE_CPU_ASYNC_LIMIT_PATTERN_COUNT       262144  // do not use all CPU cores for problems with fewer patterns

namespace beagle {
namespace cpu {

BEAGLE_CPU_TEMPLATE
class BeagleCPUImpl : public BeagleImpl {

protected:
    int kBufferCount; /// after initialize this will be partials.size()
    ///   (we don't really need this field)
    int kTipCount; /// after initialize this will be tipStates.size()
    ///   (we don't really need this field, but it is handy)
    int kPatternCount; /// the number of data patterns in each partial and tipStates element
    int kPaddedPatternCount; /// the number of data patterns padded to be a multiple of 2 or 4
    int kExtraPatterns; /// kPaddedPatternCount - kPatternCount
    int kMatrixCount; /// the number of transition matrices to alloc and store
    int kStateCount; /// the number of states
    int kTransPaddedStateCount;
    int kPartialsPaddedStateCount;
    int kEigenDecompCount; /// the number of eigen solutions to alloc and store
    int kCategoryCount;
    int kScaleBufferCount;

    int kPartialsSize;  /// stored for convenience. kPartialsSize = kStateCount*kPatternCount
    int kMatrixSize; /// stored for convenience. kMatrixSize = kStateCount*(kStateCount + 1)

    int kInternalPartialsBufferCount;

    int kPartitionCount;
    int kMaxPartitionCount;
    bool kPartitionsInitialised;
    bool kPatternsReordered;
    int kMinPatternCount;

    long kFlags;

    REALTYPE realtypeMin;
    int scalingExponentThreshold;

    EigenDecomposition<BEAGLE_CPU_EIGEN_GENERIC>* gEigenDecomposition;

    double** gCategoryRates; // Kept in double-precision until multiplication by edgelength
    double* gPatternWeights;

    int* gPatternPartitions;
    int* gPatternPartitionsStartPatterns;
    int* gPatternsNewOrder;

    REALTYPE** gCategoryWeights;
    REALTYPE** gStateFrequencies;

    //@ the size of these pointers are known at alloc-time, so the partials and
    //      tipStates field should be switched to vectors of vectors (to make
    //      memory management less error prone
    REALTYPE** gPartials;
    int** gTipStates;
    REALTYPE** gScaleBuffers;

    signed short** gAutoScaleBuffers;

    int* gActiveScalingFactors;

    // There will be kMatrixCount transitionMatrices.
    // Each kStateCount x (kStateCount+1) matrix that is flattened
    //  into a single array
    REALTYPE** gTransitionMatrices;

    REALTYPE* integrationTmp;
    REALTYPE* firstDerivTmp;
    REALTYPE* secondDerivTmp;

//    REALTYPE* cLikelihoodTmp;
    REALTYPE* grandDenominatorDerivTmp;
    REALTYPE* grandNumeratorDerivTmp;
    REALTYPE* crossProductNumeratorTmp;
//    REALTYPE* grandNumeratorLowerBoundDerivTmp;
//    REALTYPE* grandNumeratorUpperBoundDerivTmp;

    REALTYPE* outLogLikelihoodsTmp;
    REALTYPE* outFirstDerivativesTmp;
    REALTYPE* outSecondDerivativesTmp;

    REALTYPE* ones;
    REALTYPE* zeros;

    struct threadData
    {
        std::thread t; // The thread object
        std::queue<std::packaged_task<void()>> jobs; // The job queue
        std::condition_variable cv; // The condition variable to wait for threads
        std::mutex m; // Mutex used for avoiding data races
        bool stop = false; // When set, this flag tells the thread that it should exit
    };

    int kNumThreads;
    bool kThreadingEnabled;
    bool kAutoPartitioningEnabled;
    bool kAutoRootPartitioningEnabled;

    threadData* gThreads;
    int** gThreadOperations;
    int* gThreadOpCounts;
    int* gAutoPartitionOperations;
    int* gAutoPartitionIndices;
    double* gAutoPartitionOutSumLogLikelihoods;
    std::shared_future<void>* gFutures;

public:
    virtual ~BeagleCPUImpl();

    // creation of instance
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

    // initialization of instance,  returnInfo can be null
    int getInstanceDetails(BeagleInstanceDetails* returnInfo);

    int setCPUThreadCount(int threadCount);

    // set the states for a given tip
    //
    // tipIndex the index of the tip
    // inStates the array of states: 0 to stateCount - 1, missing = stateCount
    int setTipStates(int tipIndex,
                     const int* inStates);

    // set the partials for a given tip
    //
    // tipIndex the index of the tip
    // inPartials the array of partials, stateCount x patternCount
    int setTipPartials(int tipIndex,
                       const double* inPartials);

    // set the pre-order partials for root node
    //
    // bufferIndices the indices of root nodes (may countain multiple)
    // stateFrequenciesIndices the indices of state frequencies at root
    // count number of root nodes
    int setRootPrePartials(const int* bufferIndices,
                           const int* stateFrequenciesIndices,
                           int count);

    int setPartials(int bufferIndex,
                    const double* inPartials);

    int getPartials(int bufferIndex,
					int scaleBuffer,
                    double* outPartials);

    // sets the Eigen decomposition for a given matrix
    //
    // matrixIndex the matrix index to update
    // eigenVectors an array containing the Eigen Vectors
    // inverseEigenVectors an array containing the inverse Eigen Vectors
    // eigenValues an array containing the Eigen Values
    int setEigenDecomposition(int eigenIndex,
                              const double* inEigenVectors,
                              const double* inInverseEigenVectors,
                              const double* inEigenValues);

    int setStateFrequencies(int stateFrequenciesIndex,
                            const double* inStateFrequencies);

    int setCategoryWeights(int categoryWeightsIndex,
                           const double* inCategoryWeights);

    int setPatternWeights(const double* inPatternWeights);

    int setPatternPartitions(int partitionCount,
                             const int* inPatternPartitions);

    // set the vector of category rates
    //
    // categoryRates an array containing categoryCount rate scalers
    int setCategoryRates(const double* inCategoryRates);

    int setCategoryRatesWithIndex(int categoryRatesIndex,
                                  const double* inCategoryRates);

    int setTransitionMatrix(int matrixIndex,
                            const double* inMatrix,
                            double paddedValue);

    int setDifferentialMatrix(int matrixIndex,
                              const double* inMatrix);

    int setTransitionMatrices(const int* matrixIndices,
                              const double* inMatrices,
                              const double* paddedValues,
                              int count);

    int getTransitionMatrix(int matrixIndex,
    						double* outMatrix);

    int convolveTransitionMatrices(const int* firstIndices,
                                   const int* secondIndices,
                                   const int* resultIndices,
                                   int count);

	int addTransitionMatrices(const int* firstIndices,
	                          const int* secondIndices,
	                          const int* resultIndices,
	                          int matrixCount);

    int transposeTransitionMatrices(const int* inputIndices,
                                    const int* resultIndices,
                                    int matrixCount);

    // calculate a transition probability matrices for a given list of node. This will
    // calculate for all categories (and all matrices if more than one is being used).
    //
    // nodeIndices an array of node indices that require transition probability matrices
    // edgeLengths an array of expected lengths in substitutions per site
    // count the number of elements in the above arrays
    int updateTransitionMatrices(int eigenIndex,
                                 const int* probabilityIndices,
                                 const int* firstDerivativeIndices,
                                 const int* secondDerivativeIndices,
                                 const double* edgeLengths,
                                 int count);

    int updateTransitionMatricesWithModelCategories(int* eigenIndices,
                                 const int* probabilityIndices,
                                 const int* firstDerivativeIndices,
                                 const int* secondDerivativeIndices,
                                 const double* edgeLengths,
                                 int count);


    int updateTransitionMatricesWithMultipleModels(const int* eigenIndices,
                                                   const int* categoryRateIndices,
                                                   const int* probabilityIndices,
                                                   const int* firstDerivativeIndices,
                                                   const int* secondDerivativeIndices,
                                                   const double* edgeLengths,
                                                   int count);

    // calculate or queue for calculation partials using an array of operations
    //
    // operations an array of triplets of indices: the two source partials and the destination
    // dependencies an array of indices specify which operations are dependent on which (optional)
    // count the number of operations
    // rescale indicate if partials should be rescaled during peeling
    int updatePartials(const int* operations,
                       int operationCount,
                       int cumulativeScalingIndex);

    int updatePrePartials(const int *operations,
                          int operationCount,
                          int cumulativeScalingIndex);

    int updatePartialsByPartition(const int* operations,
                                  int operationCount);

    int updatePrePartialsByPartition(const int* operations,
                                     int operationCount);

    // Block until all calculations that write to the specified partials have completed.
    //
    // This function is optional and only has to be called by clients that "recycle" partials.
    //
    // If used, this function must be called after an updatePartials call and must refer to
    //  indices of "destinationPartials" that were used in a previous updatePartials
    // call.  The library will block until those partials have been calculated.
    //
    // destinationPartials - List of the indices of destinationPartials that must be calculated
    //                         before the function returns
    // destinationPartialsCount - Number of destinationPartials (input)
    //
    // return error code
    int waitForPartials(const int* destinationPartials,
                        int destinationPartialsCount);


    int accumulateScaleFactors(const int* scalingIndices,
							  int count,
							  int cumulativeScalingIndex);

    int accumulateScaleFactorsByPartition(const int* scalingIndices,
                                          int count,
                                          int cumulativeScalingIndex,
                                          int partitionIndex);

    int removeScaleFactors(const int* scalingIndices,
                           int count,
                           int cumulativeScalingIndex);

    int removeScaleFactorsByPartition(const int* scalingIndices,
                                      int count,
                                      int cumulativeScalingIndex,
                                      int partitionIndex);

    int resetScaleFactors(int cumulativeScalingIndex);

    int resetScaleFactorsByPartition(int cumulativeScalingIndex, int partitionIndex);

    int copyScaleFactors(int destScalingIndex,
                         int srcScalingIndex);

	int getScaleFactors(int srcScalingIndex,
                        double* scaleFactors);

    // calculate the site log likelihoods at a particular node
    //
    // rootNodeIndex the index of the root
    // outLogLikelihoods an array into which the site log likelihoods will be put
    int calculateRootLogLikelihoods(const int* bufferIndices,
                                    const int* categoryWeightsIndices,
                                    const int* stateFrequenciesIndices,
                                    const int* cumulativeScaleIndices,
                                    int count,
                                    double* outSumLogLikelihood);

    int calculateRootLogLikelihoodsByPartition(const int* bufferIndices,
                                               const int* categoryWeightsIndices,
                                               const int* stateFrequenciesIndices,
                                               const int* cumulativeScaleIndices,
                                               const int* partitionIndices,
                                               int partitionCount,
                                               int count,
                                               double* outSumLogLikelihoodByPartition,
                                               double* outSumLogLikelihood);

    // possible nulls: firstDerivativeIndices, secondDerivativeIndices,
    //                 outFirstDerivatives, outSecondDerivatives
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
                                    double* outSumSecondDerivative);

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
                                               double* outSumSecondDerivative);

    int calculateEdgeDerivatives(const int *postBufferIndices,
                                 const int *preBufferIndices,
                                 const int *derivativeMatrixIndices,
                                 const int *categoryWeightsIndices,
                                 const int *categoryRatesIndices,
                                 const int *cumulativeScaleIndices,
                                 int count,
                                 double *outDerivatives,
                                 double *outSumDerivatives,
                                 double *outSumSquaredDerivatives);

    int calculateCrossProducts(const int *postBufferIndices,
                               const int *preBufferIndices,
                               const int *categoryRatesIndices,
                               const int *categoryWeightsIndices,
                               const double *edgeLengths,
                               int count,
                               double *outSumDerivatives,
                               double *outSumSquaredDerivatives);

    int getLogLikelihood(double* outSumLogLikelihood);

    int getDerivatives(double* outSumFirstDerivative,
                       double* outSumSecondDerivative);

    int getSiteLogLikelihoods(double* outLogLikelihoods);

    int getSiteDerivatives(double* outFirstDerivatives,
                           double* outSecondDerivatives);
                           
	int updateBastaPartials(const int* operations,
  							int operationCount,
  							int populationSizesIndex);                             
                           
	int accumulateBastaPartials(const int* operations,
	     				  		int operationCount,
	     				  		const int* segments,
	     				  		int segmentCount);

    int block(void);

	virtual const char* getName();

	virtual const long getFlags();

protected:
    virtual int upPartials(bool byPartition,
                           const int* operations,
                           int operationCount,
                           int cumulativeScalingIndex);

    virtual int upPrePartials(bool byPartition,
                              const int* operations,
                              int count,
                              int cumulativeScaleIndex);

    virtual int calcEdgeLogDerivatives(const int *postBufferIndices,
                                       const int *preBufferIndices,
                                       const int *firstDerivativeIndices,
                                       const int *secondDerivativeIndices,
                                       const int *categoryWeightsIndices,
                                       const int *categoryRatesIndices,
                                       const int *cumulativeScaleIndices,
                                       int count,
                                       double *siteLogLikelihoods,
                                       double *outLogFirstDerivatives,
                                       double *outLogDiagonalSecondDerivatives);

    virtual void calcEdgeLogDerivativesStates(const int *tipStates,
                                              const REALTYPE *preOrderPartial,
                                              const int firstDerivativeIndex,
                                              const int secondDerivativeIndex,
                                              const double *categoryRates,
                                              const REALTYPE *categoryWeights,
//                                              const REALTYPE *cumulativeScaleBuffer,
                                              double *siteLogLikelihoods,
                                              double *outLogFirstDerivatives,
                                              double *outLogDiagonalSecondDerivatives);

    virtual void calcEdgeLogDerivativesPartials(const REALTYPE *postOrderPartial,
                                                const REALTYPE *preOrderPartial,
                                                const int firstDerivativeIndex,
                                                const int secondDerivativeIndex,
                                                const double *categoryRates,
                                                const REALTYPE *categoryWeights,
                                                const int scalingFactorsIndex,
//                                                const REALTYPE *cumulativeScaleBuffer,
                                                double *siteLogLikelihoods,
                                                double *outLogFirstDerivatives,
                                                double *outLogDiagonalSecondDerivatives);

    virtual int calcCrossProducts(const int *postBufferIndices,
                                  const int *preBufferIndices,
                                  const int *categoryRatesIndices,
                                  const int *categoryWeightsIndices,
                                  const double *edgeLengths,
                                  int count,
                                  double *outSumDerivatives,
                                  double *outSumSquaredDerivatives);

    virtual void calcCrossProductsStates(const int *tipStates,
                                         const REALTYPE *preOrderPartial,
                                         const double *categoryRates,
                                         const REALTYPE *categoryWeights,
                                         const double edgeLength,
                                         double *outCrossProducts,
                                         double *outSumSquaredDerivatives);

    virtual void calcCrossProductsPartials(const REALTYPE *postOrderPartial,
                                           const REALTYPE *preOrderPartial,
                                           const double *categoryRates,
                                           const REALTYPE *categoryWeights,
                                           const double edgeLength,
                                           double *outCrossProducts,
                                           double *outSumSquaredDerivatives);

    virtual void resetDerivativeTemporaries();

    virtual void accumulateDerivatives(double* outDerivatives,
                               double* outSumDerivatives,
                               double* outSumSquaredDerivatives);

    virtual void autoPartitionPartialsOperations(const int* operations,
                                                 int* partitionOperations,
                                                 int count,
                                                 int cumulativeScaleIndex);

    virtual int upPartialsByPartitionAsync(const int* operations,
                                           int operationCount);

    virtual int reorderPatternsByPartition();

    virtual void calcStatesStates(REALTYPE* destP,
                                  const int* states1,
                                  const REALTYPE* matrices1,
                                  const int* states2,
                                  const REALTYPE* matrices2,
                                  int startPattern,
                                  int endPattern);


    virtual void calcStatesPartials(REALTYPE* destP,
                                    const int* states1,
                                    const REALTYPE* matrices1,
                                    const REALTYPE* partials2,
                                    const REALTYPE* matrices2,
                                    int startPattern,
                                    int endPatternd);

    virtual void calcPartialsPartials(REALTYPE* destP,
                                      const REALTYPE* partials1,
                                      const REALTYPE* matrices1,
                                      const REALTYPE* partials2,
                                      const REALTYPE* matrices2,
                                      int startPattern,
                                      int endPattern);

    virtual void calcPrePartialsPartials(REALTYPE* destP,
                                         const REALTYPE* partials1,
                                         const REALTYPE* matrices1,
                                         const REALTYPE* partials2,
                                         const REALTYPE* matrices2,
                                         int startPattern,
                                         int endPattern);

    virtual void calcPrePartialsStates(REALTYPE* destP,
                                         const REALTYPE* partials1,
                                         const REALTYPE* matrices1,
                                         const int* states2,
                                         const REALTYPE* matrices2,
                                         int startPattern,
                                         int endPattern);

    virtual int calcRootLogLikelihoods(const int bufferIndex,
                                        const int categoryWeightsIndex,
                                        const int stateFrequenciesIndex,
                                        const int scaleBufferIndex,
                                        double* outSumLogLikelihood);

    virtual int calcRootLogLikelihoodsPerCategory(const int bufferIndex,
                                                  const int stateFrequenciesIndex,
                                                  const int scaleBufferIndex,
                                                  double* outSumLogLikelihood);

    virtual void calcRootLogLikelihoodsByPartitionAsync(const int* bufferIndices,
                                                       const int* categoryWeightsIndices,
                                                       const int* stateFrequenciesIndices,
                                                       const int* cumulativeScaleIndices,
                                                       const int* partitionIndices,
                                                       int partitionCount,
                                                       double* outSumLogLikelihoodByPartition);

    virtual void calcRootLogLikelihoodsByAutoPartitionAsync(const int* bufferIndices,
                                                            const int* categoryWeightsIndices,
                                                            const int* stateFrequenciesIndices,
                                                            const int* cumulativeScaleIndices,
                                                            const int* partitionIndices,
                                                            double* outSumLogLikelihoodByPartition);

    virtual void calcRootLogLikelihoodsByPartition(const int* bufferIndices,
                                                  const int* categoryWeightsIndices,
                                                  const int* stateFrequenciesIndices,
                                                  const int* cumulativeScaleIndices,
                                                  const int* partitionIndices,
                                                  int partitionCount,
                                                  double* outSumLogLikelihoodByPartition);

    virtual int calcRootLogLikelihoodsMulti(const int* bufferIndices,
                                             const int* categoryWeightsIndices,
                                             const int* stateFrequenciesIndices,
                                             const int* scaleBufferIndices,
                                             int count,
                                             double* outSumLogLikelihood);

    virtual int calcEdgeLogLikelihoods(const int parentBufferIndex,
                                        const int childBufferIndex,
                                        const int probabilityIndex,
                                        const int categoryWeightsIndex,
                                        const int stateFrequenciesIndex,
                                        const int scalingFactorsIndex,
                                        double* outSumLogLikelihood);

    virtual void calcEdgeLogLikelihoodsByPartitionAsync(const int* parentBufferIndices,
                                                        const int* childBufferIndices,
                                                        const int* probabilityIndices,
                                                        const int* categoryWeightsIndices,
                                                        const int* stateFrequenciesIndices,
                                                        const int* cumulativeScaleIndices,
                                                        const int* partitionIndices,
                                                        int partitionCount,
                                                        double* outSumLogLikelihoodByPartition);

    virtual void calcEdgeLogLikelihoodsByAutoPartitionAsync(
                                                        const int* parentBufferIndices,
                                                        const int* childBufferIndices,
                                                        const int* probabilityIndices,
                                                        const int* categoryWeightsIndices,
                                                        const int* stateFrequenciesIndices,
                                                        const int* cumulativeScaleIndices,
                                                        const int* partitionIndices,
                                                        double* outSumLogLikelihoodByPartition);

    virtual void calcEdgeLogLikelihoodsByPartition(const int* parentBufferIndices,
                                                  const int* childBufferIndices,
                                                  const int* probabilityIndices,
                                                  const int* categoryWeightsIndices,
                                                  const int* stateFrequenciesIndices,
                                                  const int* cumulativeScaleIndices,
                                                  const int* partitionIndices,
                                                  int partitionCount,
                                                  double* outSumLogLikelihoodByPartition);

    virtual void calcEdgeLogLikelihoodsSecondDerivByPartition(const int* parentBufferIndices,
                                                  const int* childBufferIndices,
                                                  const int* probabilityIndices,
                                                  const int* firstDerivativeIndices,
                                                  const int* secondDerivativeIndices,
                                                  const int* categoryWeightsIndices,
                                                  const int* stateFrequenciesIndices,
                                                  const int* cumulativeScaleIndices,
                                                  const int* partitionIndices,
                                                  int partitionCount,
                                                  double* outSumLogLikelihoodByPartition,
                                                  double* outSumFirstDerivativeByPartition,
                                                  double* outSumSecondDerivativeByPartition);

    virtual int calcEdgeLogLikelihoodsMulti(const int* parentBufferIndices,
                                            const int* childBufferIndices,
                                            const int* probabilityIndices,
                                            const int* categoryWeightsIndices,
                                            const int* stateFrequenciesIndices,
                                            const int* scalingFactorsIndices,
                                            int count,
                                            double* outSumLogLikelihood);

    virtual int calcEdgeLogLikelihoodsFirstDeriv(const int parentBufferIndex,
                                                  const int childBufferIndex,
                                                  const int probabilityIndex,
                                                  const int firstDerivativeIndex,
                                                  const int categoryWeightsIndex,
                                                  const int stateFrequenciesIndex,
                                                  const int scalingFactorsIndex,
                                                  double* outSumLogLikelihood,
                                                  double* outSumFirstDerivative);

    virtual int calcEdgeLogLikelihoodsSecondDeriv(const int parentBufferIndex,
                                                   const int childBufferIndex,
                                                   const int probabilityIndex,
                                                   const int firstDerivativeIndex,
                                                   const int secondDerivativeIndex,
                                                   const int categoryWeightsIndex,
                                                   const int stateFrequenciesIndex,
                                                   const int scalingFactorsIndex,
                                                   double* outSumLogLikelihood,
                                                   double* outSumFirstDerivative,
                                                   double* outSumSecondDerivative);

    virtual void calcStatesStatesFixedScaling(REALTYPE *destP,
                                              const int *child0States,
                                              const REALTYPE *child0TransMat,
                                              const int *child1States,
                                              const REALTYPE *child1TransMat,
                                              const REALTYPE *scaleFactors,
                                              int startPattern,
                                              int endPattern);

    virtual void calcStatesPartialsFixedScaling(REALTYPE *destP,
                                                const int *child0States,
                                                const REALTYPE *child0TransMat,
                                                const REALTYPE *child1Partials,
                                                const REALTYPE *child1TransMat,
                                                const REALTYPE *scaleFactors,
                                                int startPattern,
                                                int endPattern);

    virtual void calcPreStatesPartialsFixedScaling(REALTYPE* destP,
                                                   const int* states1,
                                                   const REALTYPE* matrices1,
                                                   const REALTYPE* partials2,
                                                   const REALTYPE* matrices2,
                                                   const REALTYPE* scaleFactors,
                                                   int startPattern,
                                                   int endPattern);

    virtual void calcPartialsPartialsFixedScaling(REALTYPE *destP,
                                            const REALTYPE *child0States,
                                            const REALTYPE *child0TransMat,
                                            const REALTYPE *child1Partials,
                                            const REALTYPE *child1TransMat,
                                            const REALTYPE *scaleFactors,
                                            int startPattern,
                                            int endPattern);

    virtual void calcPartialsPartialsAutoScaling(REALTYPE* destP,
                                                  const REALTYPE* partials1,
                                                  const REALTYPE* matrices1,
                                                  const REALTYPE* partials2,
                                                  const REALTYPE* matrices2,
                                                  int* activateScaling);

    virtual void rescalePartials(REALTYPE *destP,
    		                     REALTYPE *scaleFactors,
                                 REALTYPE *cumulativeScaleFactors,
                                 const int  fillWithOnes);

    virtual void rescalePartialsByPartition(REALTYPE *destP,
                                            REALTYPE *scaleFactors,
                                            REALTYPE *cumulativeScaleFactors,
                                            const int fillWithOnes,
                                            const int partitionIndex);

    virtual void autoRescalePartials(REALTYPE *destP,
    		                     signed short *scaleFactors);

    virtual int getPaddedPatternsModulus();

    void* mallocAligned(size_t size);

    void threadWaiting(threadData* tData);

private:

    template <bool DoDerivatives>
    void accumulateDerivativesDispatch1(double* outDerivatives,
                               double* outSumDerivatives,
                               double* outSumSquaredDerivatives);

    template <bool DoDerivatives, bool DoSum>
    void accumulateDerivativesDispatch2(double* outDerivatives,
                               double* outSumDerivatives,
                               double* outSumSquaredDerivatives);

    template <bool DoDerivatives, bool DoSum, bool DoSumSquared>
    void accumulateDerivativesImpl(double* outDerivatives,
                               double* outSumDerivatives,
                               double* outSumSquaredDerivatives);
};

BEAGLE_CPU_FACTORY_TEMPLATE
class BeagleCPUImplFactory : public BeagleImplFactory {
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

//typedef BeagleCPUImplGeneral<double> BeagleCPUImpl;

}	// namespace cpu
}	// namespace beagle

// now that the interface is defined, include the implementation of template functions
#include "libhmsbeagle/CPU/BeagleCPUImpl.hpp"

#endif // __BeagleCPUImpl__
