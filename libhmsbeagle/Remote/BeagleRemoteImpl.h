/*
 *  BeagleRemoteImpl.h
 *  BEAGLE
 *
 * An implementation whose instance lives in a beagle-worker process on another machine. Each call is written
 * to a batch; the batch is sent when a call has to return something (see RemoteProtocol.h).
 */

#ifndef __BEAGLE_REMOTE_IMPL_H__
#define __BEAGLE_REMOTE_IMPL_H__

#include <string>
#include <vector>

#include "libhmsbeagle/BeagleImpl.h"
#include "libhmsbeagle/Remote/RemoteChannel.h"
#include "libhmsbeagle/Remote/RemoteProtocol.h"

namespace beagle {
namespace remote {

/** One resource of one worker, as it appears in this process's resource list. */
struct RemoteResource {
    std::string host;
    int port;
    int workerResource;       /**< The resource's number in the worker's own resource list */
    std::string name;
    std::string description;
    long long supportFlags;
    long long requiredFlags;
};

/**
 * Connects to a worker and exchanges hellos. Returns the open channel, or NULL with the reason in error.
 * If resources is not NULL it receives the worker's resource list (host and port filled in, the flags as the
 * worker reports them).
 */
Channel* openWorker(const std::string& host, int port, int timeoutMilliseconds,
                    std::vector<RemoteResource>* resources, std::string* error);

class BeagleRemoteImpl : public BeagleImpl {
public:
    explicit BeagleRemoteImpl(const RemoteResource& resource);
    virtual ~BeagleRemoteImpl();

    int createInstance(int tipCount, int partialsBufferCount, int compactBufferCount, int stateCount,
                       int patternCount, int eigenBufferCount, int matrixBufferCount, int categoryCount,
                       int scaleBufferCount, int resourceNumber, int pluginResourceNumber,
                       long long preferenceFlags, long long requirementFlags);

    int getInstanceDetails(BeagleInstanceDetails* returnInfo);
    int setCPUThreadCount(int threadCount);
    int ensureBufferCounts(int partialsBufferCount, int matrixBufferCount, int scaleBufferCount);

    int setTipStates(int tipIndex, const int* inStates);
    int setTipPartials(int tipIndex, const double* inPartials);
    int setPartials(int bufferIndex, const double* inPartials);
    int setRootPrePartials(const int* bufferIndices, const int* stateFrequenciesIndices, int count);
    int getPartials(int bufferIndex, int scaleIndex, double* outPartials);
    int getPartialsBatch(const int* bufferIndices, const int* scaleIndices, int count, double* outPartials);

    int setEigenDecomposition(int eigenIndex, const double* inEigenVectors, const double* inInverseEigenVectors,
                              const double* inEigenValues);
    int setStateFrequencies(int stateFrequenciesIndex, const double* inStateFrequencies);
    int setCategoryWeights(int categoryWeightsIndex, const double* inCategoryWeights);
    int setPatternWeights(const double* inPatternWeights);
    int setPatternPartitions(int partitionCount, const int* inPatternPartitions);
    int setCategoryRates(const double* inCategoryRates);
    int setCategoryRatesWithIndex(int categoryRatesIndex, const double* inCategoryRates);

    int setTransitionMatrix(int matrixIndex, const double* inMatrix, double paddedValue);
    int setDifferentialMatrix(int matrixIndex, const double* inMatrix);
    int setTransitionMatrices(const int* matrixIndices, const double* inMatrices, const double* paddedValues,
                              int count);
    int getTransitionMatrix(int matrixIndex, double* outMatrix);
    int convolveTransitionMatrices(const int* firstIndices, const int* secondIndices, const int* resultIndices,
                                   int matrixCount);
    int addTransitionMatrices(const int* firstIndices, const int* secondIndices, const int* resultIndices,
                              int matrixCount);
    int transposeTransitionMatrices(const int* inputIndices, const int* resultIndices, int matrixCount);

    int updateTransitionMatrices(int eigenIndex, const int* probabilityIndices, const int* firstDerivativeIndices,
                                 const int* secondDerivativeIndices, const double* edgeLengths, int count);
    int updateTransitionMatricesWithModelCategories(int* eigenIndices, const int* probabilityIndices,
                                                    const int* firstDerivativeIndices,
                                                    const int* secondDerivativeIndices, const double* edgeLengths,
                                                    int count);
    int updateTransitionMatricesWithMultipleModels(const int* eigenIndices, const int* categoryRateIndices,
                                                   const int* probabilityIndices,
                                                   const int* firstDerivativeIndices,
                                                   const int* secondDerivativeIndices, const double* edgeLengths,
                                                   int count);

    int updatePartials(const int* operations, int operationCount, int cumulativeScalingIndex,
                       BeaglePartialsType partialsType);
    int updatePrePartials(const int* operations, int count, int cumulativeScaleIndex,
                          BeaglePartialsType partialsType);
    int updatePartialsByPartition(const int* operations, int operationCount);
    int updatePrePartialsByPartition(const int* operations, int operationCount, BeaglePartialsType partialsType);
    int waitForPartials(const int* destinationPartials, int destinationPartialsCount);

    int accumulateScaleFactors(const int* scalingIndices, int count, int cumulativeScalingIndex);
    int accumulateScaleFactorsByPartition(const int* scaleIndices, int count, int cumulativeScaleIndex,
                                          int partitionIndex);
    int removeScaleFactors(const int* scalingIndices, int count, int cumulativeScalingIndex);
    int removeScaleFactorsByPartition(const int* scaleIndices, int count, int cumulativeScaleIndex,
                                      int partitionIndex);
    int resetScaleFactors(int cumulativeScalingIndex);
    int resetScaleFactorsByPartition(int cumulativeScaleIndex, int partitionIndex);
    int copyScaleFactors(int destScalingIndex, int srcScalingIndex);
    int getScaleFactors(int srcScalingIndex, double* scaleFactors);

    int calculateRootLogLikelihoods(const int* bufferIndices, const int* categoryWeightsIndices,
                                    const int* stateFrequenciesIndices, const int* scalingFactorsIndices,
                                    int count, double* outSumLogLikelihood);
    int calculateRootLogLikelihoodsByPartition(const int* bufferIndices, const int* categoryWeightsIndices,
                                               const int* stateFrequenciesIndices,
                                               const int* cumulativeScaleIndices, const int* partitionIndices,
                                               int partitionCount, int count,
                                               double* outSumLogLikelihoodByPartition,
                                               double* outSumLogLikelihood);
    int calculateEdgeLogLikelihoods(const int* parentBufferIndices, const int* childBufferIndices,
                                    const int* probabilityIndices, const int* firstDerivativeIndices,
                                    const int* secondDerivativeIndices, const int* categoryWeightsIndices,
                                    const int* stateFrequenciesIndices, const int* scalingFactorsIndices,
                                    int count, double* outSumLogLikelihood, double* outSumFirstDerivative,
                                    double* outSumSecondDerivative);
    int calculateEdgeLogLikelihoodsByPartition(const int* parentBufferIndices, const int* childBufferIndices,
                                               const int* probabilityIndices, const int* firstDerivativeIndices,
                                               const int* secondDerivativeIndices,
                                               const int* categoryWeightsIndices,
                                               const int* stateFrequenciesIndices,
                                               const int* cumulativeScaleIndices, const int* partitionIndices,
                                               int partitionCount, int count,
                                               double* outSumLogLikelihoodByPartition,
                                               double* outSumLogLikelihood,
                                               double* outSumFirstDerivativeByPartition,
                                               double* outSumFirstDerivative,
                                               double* outSumSecondDerivativeByPartition,
                                               double* outSumSecondDerivative);

    int getLogLikelihood(double* outSumLogLikelihood);
    int getDerivatives(double* outSumFirstDerivative, double* outSumSecondDerivative);
    int getSiteLogLikelihoods(double* outLogLikelihoods);
    int getSiteDerivatives(double* outFirstDerivatives, double* outSecondDerivatives);

    // Not carried by the protocol yet: these return BEAGLE_ERROR_NO_IMPLEMENTATION
    int updateTransitionMatricesGrad(const int*, const double*, int);
    int calculateEdgeDerivatives(const int*, const int*, const int*, const int*, const int*, const int*, int,
                                 double*, double*, double*);
    int calculateCrossProducts(const int*, const int*, const int*, const int*, const double*, int, double*,
                               double*);
    int calculateAdjointCrossProducts(const int*, const int*, const int*, const int*, const int*, const int,
                                      const int, int, double*, double*, const int*, const int*, const int);
    int calculateAdjointDerivative(const BeagleBranchOperation*, const int, const int, const int, const int,
                                   const int, double*, double*);
    int updateBastaPartials(const int*, int, const int*, int, int, int);
    int updateBastaPartialsGrad(const int*, int, const int*, int, int, int);
    int accumulateBastaPartials(const int*, int, const int*, int, const double*, const int, int, double*);
    int accumulateBastaPartialsGrad(const int*, int, const int*, int, const double*, const int, int, int, int,
                                    int, double*, double*);
    int uploadBastaSlabMetadata(const int*, int);
    int getBastaSlabConstants(int*, int*);
    int allocateBastaBuffers(int, int, int, int, int);
    int allocateBastaGradBuffers(int);
    int getBastaBuffer(int, double*);
    int accumulateEigenBasisGradient(int, int, const double*, int, int, double*);
    int setBastaPopulationSizes(const double*, int);
    int setBastaPopulationSizesBuffer(const double*, int, int);
    int setCurrentPopulationSizeBuffer(int);
    int updateBastaPartials_v5(const int*, int, const int*, int, int, int);
    int updateBastaPartialsGrad_v5(const int*, int, const int*, int, int, int);
    int accumulateBastaPartials_v5(const int*, int, const int*, int, const double*, const int, int, double*);
    int accumulateBastaPartialsGrad_v5(const int*, int, const int*, int, const double*, const int, int, int, int,
                                       int, double*, double*);

private:
    void begin(Opcode opcode);
    int defer();
    int exchange();
    int fail(const char* reason);

    // Sends the batch, then hands the outputs of its last command to readOutputs if that command succeeded
    template <typename ReadOutputs>
    int call(ReadOutputs readOutputs);

    RemoteResource resource_;
    Channel* channel_;
    Buffer out_;            // the batch being built
    Buffer in_;             // the last response
    bool broken_;           // the link has failed; every call now returns an error
    bool hasOutputs_;       // the last response carries the outputs of its last command

    int kStateCount;
    int kPatternCount;
    int kCategoryCount;
    long long kFlags;       // the worker instance's flags
    std::string implName_;
};

class BeagleRemoteImplFactory : public BeagleImplFactory {
public:
    BeagleRemoteImplFactory(const std::vector<RemoteResource>& resources, long long flags)
        : resources_(resources), flags_(flags) {}

    virtual BeagleImpl* createImpl(int tipCount, int partialsBufferCount, int compactBufferCount, int stateCount,
                                   int patternCount, int eigenBufferCount, int matrixBufferCount,
                                   int categoryCount, int scaleBufferCount, int resourceNumber,
                                   int pluginResourceNumber, long long preferenceFlags,
                                   long long requirementFlags, int* errorCode);

    virtual const char* getName() { return "Remote"; }
    virtual const long long getFlags() { return flags_; }

private:
    std::vector<RemoteResource> resources_;
    long long flags_;
};

} // namespace remote
} // namespace beagle

#endif // __BEAGLE_REMOTE_IMPL_H__
