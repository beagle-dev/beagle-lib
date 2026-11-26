/*
 *  BeagleRemoteImpl.cpp
 *  BEAGLE
 */

#include "libhmsbeagle/Remote/BeagleRemoteImpl.h"

#include <iostream>

#include "libhmsbeagle/Remote/RemoteAuth.h"

namespace beagle {
namespace remote {

namespace {

// A batch of queued calls is sent once it reaches this size, so that loading the data of a large analysis
// does not hold all of it in memory twice
const size_t kFlushBytes = 4 << 20;

const int kConnectTimeoutMilliseconds = 2000;

} // anonymous namespace

Channel* openWorker(const std::string& host, int port, int timeoutMilliseconds,
                    std::vector<RemoteResource>* resources, std::string* error) {
    Channel* channel = connectTcp(host, port, timeoutMilliseconds, error);
    if (channel == NULL) return NULL;

    try {
        std::string secret;
        std::string secretError;
        if (!secretFromEnvironment(&secret, &secretError)) throw ProtocolError(secretError);

        std::string clientNonce = randomNonce();
        Buffer hello;
        hello.putI64(kMagic);
        hello.putI64(kProtocolVersion);
        hello.putString(clientNonce);
        Buffer answer;
        if (!channel->send(hello) || !channel->receive(answer)) throw ProtocolError("connection closed");

        Reader challenge(answer);
        if (challenge.i64() != kMagic) throw ProtocolError("not a beagle-worker");
        if (challenge.i64() != kProtocolVersion)
            throw ProtocolError("beagle-worker speaks another protocol version");
        bool secretRequired = challenge.i64() != 0;
        std::string workerNonce = challenge.string();
        if (workerNonce.size() != kNonceBytes) throw ProtocolError("malformed hello");

        if (secretRequired && secret.empty())
            throw ProtocolError(std::string("the worker requires a secret: set ") + kSecretFileVariable + " or " +
                                kSecretVariable);
        // a worker that asks for no secret cannot show that it is the one meant, and would be sent the data
        if (!secretRequired && !secret.empty())
            throw ProtocolError("a secret is set here but the worker was started without one");

        Buffer proof;
        proof.putString(secret.empty() ? std::string() : proofOfSecret(secret, "client", clientNonce, workerNonce));
        if (!channel->send(proof) || !channel->receive(answer)) throw ProtocolError("connection closed");

        Reader reader(answer);
        if (reader.i64() != 0) throw ProtocolError("the worker has a different secret");
        std::string workerProof = reader.string();
        if (!secret.empty() && !sameBytes(workerProof, proofOfSecret(secret, "worker", clientNonce, workerNonce)))
            throw ProtocolError("the worker does not know the secret");

        size_t resourceCount = reader.count();
        for (size_t i = 0; i < resourceCount; i++) {
            RemoteResource resource;
            resource.host = host;
            resource.port = port;
            resource.workerResource = (int) i;
            resource.name = reader.string();
            resource.description = reader.string();
            resource.supportFlags = reader.i64();
            resource.requiredFlags = reader.i64();
            if (resources) resources->push_back(resource);
        }
    } catch (const ProtocolError& e) {
        if (error) *error = e.what();
        delete channel;
        return NULL;
    }
    return channel;
}

BeagleRemoteImpl::BeagleRemoteImpl(const RemoteResource& resource)
    : resource_(resource), channel_(NULL), broken_(false), hasOutputs_(false),
      kStateCount(0), kPatternCount(0), kCategoryCount(0), kFlags(0) {}

BeagleRemoteImpl::~BeagleRemoteImpl() {
    // closing the connection is what tells the worker to finalize the instance
    delete channel_;
}

void BeagleRemoteImpl::begin(Opcode opcode) {
    out_.putI64(opcode);
}

int BeagleRemoteImpl::fail(const char* reason) {
    if (!broken_) {
        std::cerr << "BEAGLE remote: lost the worker at " << resource_.host << ":" << resource_.port
                  << " (" << reason << ")" << std::endl;
        broken_ = true;
    }
    out_.clear();
    hasOutputs_ = false;
    return BEAGLE_ERROR_GENERAL;
}

/*
 * Sends the batch and waits for the response. Returns the last command's return code, or, if that succeeded,
 * the first error among the queued commands before it.
 */
int BeagleRemoteImpl::exchange() {
    hasOutputs_ = false;
    if (broken_ || channel_ == NULL) return fail("not connected");
    bool sent = channel_->send(out_);
    out_.clear();
    if (!sent) return fail("send failed");
    if (!channel_->receive(in_)) return fail("connection closed");
    if (in_.size() < 2 * sizeof(int64_t)) return fail("short response");

    Reader reader(in_);
    int deferredError = reader.i32();
    int status = reader.i32();
    hasOutputs_ = (status == BEAGLE_SUCCESS);
    return status != BEAGLE_SUCCESS ? status : deferredError;
}

template <typename ReadOutputs>
int BeagleRemoteImpl::call(ReadOutputs readOutputs) {
    int returnCode = exchange();
    if (hasOutputs_) {
        try {
            Reader reader(in_.data() + 2 * sizeof(int64_t), in_.size() - 2 * sizeof(int64_t));
            readOutputs(reader);
        } catch (const ProtocolError& e) {
            return fail(e.what());
        }
    }
    return returnCode;
}

/* Ends a call that returns nothing: it stays in the batch unless the batch has grown large. */
int BeagleRemoteImpl::defer() {
    if (broken_) return fail("not connected");
    if (out_.size() < kFlushBytes) return BEAGLE_SUCCESS;
    begin(OP_SYNC);
    return exchange();
}

int BeagleRemoteImpl::createInstance(int tipCount, int partialsBufferCount, int compactBufferCount,
                                     int stateCount, int patternCount, int eigenBufferCount,
                                     int matrixBufferCount, int categoryCount, int scaleBufferCount,
                                     int globalResourceNumber, int pluginResourceNumber,
                                     long long preferenceFlags, long long requirementFlags) {
    std::string error;
    channel_ = openWorker(resource_.host, resource_.port, kConnectTimeoutMilliseconds, NULL, &error);
    if (channel_ == NULL) {
        std::cerr << "BEAGLE remote: cannot reach the worker at " << resource_.host << ":" << resource_.port
                  << " (" << error << ")" << std::endl;
        return BEAGLE_ERROR_NO_RESOURCE;
    }

    resourceNumber = globalResourceNumber;
    kStateCount = stateCount;
    kPatternCount = patternCount;
    kCategoryCount = categoryCount;

    begin(OP_CREATE_INSTANCE);
    out_.putI64(tipCount);
    out_.putI64(partialsBufferCount);
    out_.putI64(compactBufferCount);
    out_.putI64(stateCount);
    out_.putI64(patternCount);
    out_.putI64(eigenBufferCount);
    out_.putI64(matrixBufferCount);
    out_.putI64(categoryCount);
    out_.putI64(scaleBufferCount);
    out_.putI64(resource_.workerResource);
    // the worker's own resources are not remote ones
    out_.putI64(preferenceFlags & ~BEAGLE_FLAG_FRAMEWORK_REMOTE);
    out_.putI64(requirementFlags & ~BEAGLE_FLAG_FRAMEWORK_REMOTE);
    return call([this](Reader& reader) {
        kFlags = reader.i64();
        implName_ = "Remote-" + reader.string();
    });
}

int BeagleRemoteImpl::getInstanceDetails(BeagleInstanceDetails* returnInfo) {
    if (returnInfo != NULL) {
        returnInfo->resourceNumber = resourceNumber;
        returnInfo->flags = kFlags | BEAGLE_FLAG_FRAMEWORK_REMOTE;
        returnInfo->implName = (char*) implName_.c_str();
    }
    return BEAGLE_SUCCESS;
}

int BeagleRemoteImpl::setCPUThreadCount(int threadCount) {
    begin(OP_SET_CPU_THREAD_COUNT);
    out_.putI64(threadCount);
    return defer();
}

int BeagleRemoteImpl::ensureBufferCounts(int partialsBufferCount, int matrixBufferCount, int scaleBufferCount) {
    // not queued: the caller acts on the return code (an implementation that cannot grow says so)
    begin(OP_ENSURE_BUFFER_COUNTS);
    out_.putI64(partialsBufferCount);
    out_.putI64(matrixBufferCount);
    out_.putI64(scaleBufferCount);
    return call([](Reader&) {});
}

int BeagleRemoteImpl::setTipStates(int tipIndex, const int* inStates) {
    begin(OP_SET_TIP_STATES);
    out_.putI64(tipIndex);
    out_.putInts(inStates, kPatternCount);
    return defer();
}

int BeagleRemoteImpl::setTipPartials(int tipIndex, const double* inPartials) {
    begin(OP_SET_TIP_PARTIALS);
    out_.putI64(tipIndex);
    out_.putDoubles(inPartials, (size_t) kPatternCount * kStateCount);
    return defer();
}

int BeagleRemoteImpl::setPartials(int bufferIndex, const double* inPartials) {
    begin(OP_SET_PARTIALS);
    out_.putI64(bufferIndex);
    out_.putDoubles(inPartials, (size_t) kPatternCount * kStateCount * kCategoryCount);
    return defer();
}

int BeagleRemoteImpl::setRootPrePartials(const int* bufferIndices, const int* stateFrequenciesIndices, int count) {
    begin(OP_SET_ROOT_PRE_PARTIALS);
    out_.putI64(count);
    out_.putInts(bufferIndices, count);
    out_.putInts(stateFrequenciesIndices, count);
    return defer();
}

int BeagleRemoteImpl::getPartials(int bufferIndex, int scaleIndex, double* outPartials) {
    begin(OP_GET_PARTIALS);
    out_.putI64(bufferIndex);
    out_.putI64(scaleIndex);
    const size_t size = (size_t) kPatternCount * kStateCount * kCategoryCount;
    return call([=](Reader& reader) { reader.copyDoubles(outPartials, size); });
}

int BeagleRemoteImpl::getPartialsBatch(const int* bufferIndices, const int* scaleIndices, int count,
                                       double* outPartials) {
    begin(OP_GET_PARTIALS_BATCH);
    out_.putI64(count);
    out_.putInts(bufferIndices, count);
    out_.putInts(scaleIndices, count);
    const size_t size = (size_t) count * kPatternCount * kStateCount * kCategoryCount;
    return call([=](Reader& reader) { reader.copyDoubles(outPartials, size); });
}

int BeagleRemoteImpl::setEigenDecomposition(int eigenIndex, const double* inEigenVectors,
                                            const double* inInverseEigenVectors, const double* inEigenValues) {
    begin(OP_SET_EIGEN_DECOMPOSITION);
    out_.putI64(eigenIndex);
    out_.putDoubles(inEigenVectors, (size_t) kStateCount * kStateCount);
    out_.putDoubles(inInverseEigenVectors, (size_t) kStateCount * kStateCount);
    // complex eigenvalues are passed as the real parts followed by the imaginary parts
    out_.putDoubles(inEigenValues, (kFlags & BEAGLE_FLAG_EIGEN_COMPLEX) ? 2 * (size_t) kStateCount : kStateCount);
    return defer();
}

int BeagleRemoteImpl::setStateFrequencies(int stateFrequenciesIndex, const double* inStateFrequencies) {
    begin(OP_SET_STATE_FREQUENCIES);
    out_.putI64(stateFrequenciesIndex);
    out_.putDoubles(inStateFrequencies, kStateCount);
    return defer();
}

int BeagleRemoteImpl::setCategoryWeights(int categoryWeightsIndex, const double* inCategoryWeights) {
    begin(OP_SET_CATEGORY_WEIGHTS);
    out_.putI64(categoryWeightsIndex);
    out_.putDoubles(inCategoryWeights, kCategoryCount);
    return defer();
}

int BeagleRemoteImpl::setPatternWeights(const double* inPatternWeights) {
    begin(OP_SET_PATTERN_WEIGHTS);
    out_.putDoubles(inPatternWeights, kPatternCount);
    return defer();
}

int BeagleRemoteImpl::setPatternPartitions(int partitionCount, const int* inPatternPartitions) {
    begin(OP_SET_PATTERN_PARTITIONS);
    out_.putI64(partitionCount);
    out_.putInts(inPatternPartitions, kPatternCount);
    return defer();
}

int BeagleRemoteImpl::setCategoryRates(const double* inCategoryRates) {
    begin(OP_SET_CATEGORY_RATES);
    out_.putDoubles(inCategoryRates, kCategoryCount);
    return defer();
}

int BeagleRemoteImpl::setCategoryRatesWithIndex(int categoryRatesIndex, const double* inCategoryRates) {
    begin(OP_SET_CATEGORY_RATES_WITH_INDEX);
    out_.putI64(categoryRatesIndex);
    out_.putDoubles(inCategoryRates, kCategoryCount);
    return defer();
}

int BeagleRemoteImpl::setTransitionMatrix(int matrixIndex, const double* inMatrix, double paddedValue) {
    begin(OP_SET_TRANSITION_MATRIX);
    out_.putI64(matrixIndex);
    out_.putDoubles(inMatrix, (size_t) kStateCount * kStateCount * kCategoryCount);
    out_.putF64(paddedValue);
    return defer();
}

int BeagleRemoteImpl::setDifferentialMatrix(int matrixIndex, const double* inMatrix) {
    begin(OP_SET_DIFFERENTIAL_MATRIX);
    out_.putI64(matrixIndex);
    out_.putDoubles(inMatrix, (size_t) kStateCount * kStateCount * kCategoryCount);
    return defer();
}

int BeagleRemoteImpl::setTransitionMatrices(const int* matrixIndices, const double* inMatrices,
                                            const double* paddedValues, int count) {
    begin(OP_SET_TRANSITION_MATRICES);
    out_.putI64(count);
    out_.putInts(matrixIndices, count);
    out_.putDoubles(inMatrices, (size_t) count * kStateCount * kStateCount * kCategoryCount);
    out_.putDoubles(paddedValues, count);
    return defer();
}

int BeagleRemoteImpl::getTransitionMatrix(int matrixIndex, double* outMatrix) {
    begin(OP_GET_TRANSITION_MATRIX);
    out_.putI64(matrixIndex);
    const size_t size = (size_t) kStateCount * kStateCount * kCategoryCount;
    return call([=](Reader& reader) { reader.copyDoubles(outMatrix, size); });
}

int BeagleRemoteImpl::convolveTransitionMatrices(const int* firstIndices, const int* secondIndices,
                                                 const int* resultIndices, int matrixCount) {
    begin(OP_CONVOLVE_TRANSITION_MATRICES);
    out_.putI64(matrixCount);
    out_.putInts(firstIndices, matrixCount);
    out_.putInts(secondIndices, matrixCount);
    out_.putInts(resultIndices, matrixCount);
    return defer();
}

int BeagleRemoteImpl::addTransitionMatrices(const int* firstIndices, const int* secondIndices,
                                            const int* resultIndices, int matrixCount) {
    begin(OP_ADD_TRANSITION_MATRICES);
    out_.putI64(matrixCount);
    out_.putInts(firstIndices, matrixCount);
    out_.putInts(secondIndices, matrixCount);
    out_.putInts(resultIndices, matrixCount);
    return defer();
}

int BeagleRemoteImpl::transposeTransitionMatrices(const int* inputIndices, const int* resultIndices,
                                                  int matrixCount) {
    begin(OP_TRANSPOSE_TRANSITION_MATRICES);
    out_.putI64(matrixCount);
    out_.putInts(inputIndices, matrixCount);
    out_.putInts(resultIndices, matrixCount);
    return defer();
}

int BeagleRemoteImpl::updateTransitionMatrices(int eigenIndex, const int* probabilityIndices,
                                               const int* firstDerivativeIndices,
                                               const int* secondDerivativeIndices, const double* edgeLengths,
                                               int count) {
    begin(OP_UPDATE_TRANSITION_MATRICES);
    out_.putI64(eigenIndex);
    out_.putI64(count);
    out_.putInts(probabilityIndices, count);
    out_.putInts(firstDerivativeIndices, count);
    out_.putInts(secondDerivativeIndices, count);
    out_.putDoubles(edgeLengths, count);
    return defer();
}

int BeagleRemoteImpl::updateTransitionMatricesWithModelCategories(int* eigenIndices,
                                                                  const int* probabilityIndices,
                                                                  const int* firstDerivativeIndices,
                                                                  const int* secondDerivativeIndices,
                                                                  const double* edgeLengths, int count) {
    begin(OP_UPDATE_TRANSITION_MATRICES_WITH_MODEL_CATEGORIES);
    out_.putI64(count);
    out_.putInts(eigenIndices, kCategoryCount);
    out_.putInts(probabilityIndices, count);
    out_.putInts(firstDerivativeIndices, count);
    out_.putInts(secondDerivativeIndices, count);
    out_.putDoubles(edgeLengths, count);
    return defer();
}

int BeagleRemoteImpl::updateTransitionMatricesWithMultipleModels(const int* eigenIndices,
                                                                 const int* categoryRateIndices,
                                                                 const int* probabilityIndices,
                                                                 const int* firstDerivativeIndices,
                                                                 const int* secondDerivativeIndices,
                                                                 const double* edgeLengths, int count) {
    begin(OP_UPDATE_TRANSITION_MATRICES_WITH_MULTIPLE_MODELS);
    out_.putI64(count);
    out_.putInts(eigenIndices, count);
    out_.putInts(categoryRateIndices, count);
    out_.putInts(probabilityIndices, count);
    out_.putInts(firstDerivativeIndices, count);
    out_.putInts(secondDerivativeIndices, count);
    out_.putDoubles(edgeLengths, count);
    return defer();
}

int BeagleRemoteImpl::updatePartials(const int* operations, int operationCount, int cumulativeScalingIndex,
                                     BeaglePartialsType partialsType) {
    begin(OP_UPDATE_PARTIALS);
    out_.putI64(operationCount);
    out_.putInts(operations, (size_t) operationCount * BEAGLE_OP_COUNT);
    out_.putI64(cumulativeScalingIndex);
    out_.putI64(partialsType);
    return defer();
}

int BeagleRemoteImpl::updatePrePartials(const int* operations, int count, int cumulativeScaleIndex,
                                        BeaglePartialsType partialsType) {
    begin(OP_UPDATE_PRE_PARTIALS);
    out_.putI64(count);
    out_.putInts(operations, (size_t) count * BEAGLE_OP_COUNT);
    out_.putI64(cumulativeScaleIndex);
    out_.putI64(partialsType);
    return defer();
}

int BeagleRemoteImpl::updatePartialsByPartition(const int* operations, int operationCount) {
    begin(OP_UPDATE_PARTIALS_BY_PARTITION);
    out_.putI64(operationCount);
    out_.putInts(operations, (size_t) operationCount * BEAGLE_PARTITION_OP_COUNT);
    return defer();
}

int BeagleRemoteImpl::updatePrePartialsByPartition(const int* operations, int operationCount,
                                                   BeaglePartialsType partialsType) {
    begin(OP_UPDATE_PRE_PARTIALS_BY_PARTITION);
    out_.putI64(operationCount);
    out_.putInts(operations, (size_t) operationCount * BEAGLE_PARTITION_OP_COUNT);
    out_.putI64(partialsType);
    return defer();
}

int BeagleRemoteImpl::waitForPartials(const int* destinationPartials, int destinationPartialsCount) {
    // a call that blocks until partials are computed is what a round trip is
    begin(OP_WAIT_FOR_PARTIALS);
    out_.putI64(destinationPartialsCount);
    out_.putInts(destinationPartials, destinationPartialsCount);
    return call([](Reader&) {});
}

int BeagleRemoteImpl::accumulateScaleFactors(const int* scalingIndices, int count, int cumulativeScalingIndex) {
    begin(OP_ACCUMULATE_SCALE_FACTORS);
    out_.putI64(count);
    out_.putInts(scalingIndices, count);
    out_.putI64(cumulativeScalingIndex);
    return defer();
}

int BeagleRemoteImpl::accumulateScaleFactorsByPartition(const int* scaleIndices, int count,
                                                        int cumulativeScaleIndex, int partitionIndex) {
    begin(OP_ACCUMULATE_SCALE_FACTORS_BY_PARTITION);
    out_.putI64(count);
    out_.putInts(scaleIndices, count);
    out_.putI64(cumulativeScaleIndex);
    out_.putI64(partitionIndex);
    return defer();
}

int BeagleRemoteImpl::removeScaleFactors(const int* scalingIndices, int count, int cumulativeScalingIndex) {
    begin(OP_REMOVE_SCALE_FACTORS);
    out_.putI64(count);
    out_.putInts(scalingIndices, count);
    out_.putI64(cumulativeScalingIndex);
    return defer();
}

int BeagleRemoteImpl::removeScaleFactorsByPartition(const int* scaleIndices, int count, int cumulativeScaleIndex,
                                                    int partitionIndex) {
    begin(OP_REMOVE_SCALE_FACTORS_BY_PARTITION);
    out_.putI64(count);
    out_.putInts(scaleIndices, count);
    out_.putI64(cumulativeScaleIndex);
    out_.putI64(partitionIndex);
    return defer();
}

int BeagleRemoteImpl::resetScaleFactors(int cumulativeScalingIndex) {
    begin(OP_RESET_SCALE_FACTORS);
    out_.putI64(cumulativeScalingIndex);
    return defer();
}

int BeagleRemoteImpl::resetScaleFactorsByPartition(int cumulativeScaleIndex, int partitionIndex) {
    begin(OP_RESET_SCALE_FACTORS_BY_PARTITION);
    out_.putI64(cumulativeScaleIndex);
    out_.putI64(partitionIndex);
    return defer();
}

int BeagleRemoteImpl::copyScaleFactors(int destScalingIndex, int srcScalingIndex) {
    begin(OP_COPY_SCALE_FACTORS);
    out_.putI64(destScalingIndex);
    out_.putI64(srcScalingIndex);
    return defer();
}

int BeagleRemoteImpl::getScaleFactors(int srcScalingIndex, double* scaleFactors) {
    begin(OP_GET_SCALE_FACTORS);
    out_.putI64(srcScalingIndex);
    const size_t size = kPatternCount;
    return call([=](Reader& reader) { reader.copyDoubles(scaleFactors, size); });
}

int BeagleRemoteImpl::calculateRootLogLikelihoods(const int* bufferIndices, const int* categoryWeightsIndices,
                                                  const int* stateFrequenciesIndices,
                                                  const int* scalingFactorsIndices, int count,
                                                  double* outSumLogLikelihood) {
    begin(OP_CALCULATE_ROOT_LOG_LIKELIHOODS);
    out_.putI64(count);
    out_.putInts(bufferIndices, count);
    out_.putInts(categoryWeightsIndices, count);
    out_.putInts(stateFrequenciesIndices, count);
    out_.putInts(scalingFactorsIndices, count);
    return call([=](Reader& reader) { *outSumLogLikelihood = reader.f64(); });
}

int BeagleRemoteImpl::calculateRootLogLikelihoodsByPartition(const int* bufferIndices,
                                                             const int* categoryWeightsIndices,
                                                             const int* stateFrequenciesIndices,
                                                             const int* cumulativeScaleIndices,
                                                             const int* partitionIndices, int partitionCount,
                                                             int count, double* outSumLogLikelihoodByPartition,
                                                             double* outSumLogLikelihood) {
    const size_t size = (size_t) partitionCount * count;
    begin(OP_CALCULATE_ROOT_LOG_LIKELIHOODS_BY_PARTITION);
    out_.putI64(partitionCount);
    out_.putI64(count);
    out_.putInts(bufferIndices, size);
    out_.putInts(categoryWeightsIndices, size);
    out_.putInts(stateFrequenciesIndices, size);
    out_.putInts(cumulativeScaleIndices, size);
    out_.putInts(partitionIndices, size);
    return call([=](Reader& reader) {
        reader.copyDoubles(outSumLogLikelihoodByPartition, size);
        *outSumLogLikelihood = reader.f64();
    });
}

int BeagleRemoteImpl::calculateEdgeLogLikelihoods(const int* parentBufferIndices, const int* childBufferIndices,
                                                  const int* probabilityIndices,
                                                  const int* firstDerivativeIndices,
                                                  const int* secondDerivativeIndices,
                                                  const int* categoryWeightsIndices,
                                                  const int* stateFrequenciesIndices,
                                                  const int* scalingFactorsIndices, int count,
                                                  double* outSumLogLikelihood, double* outSumFirstDerivative,
                                                  double* outSumSecondDerivative) {
    begin(OP_CALCULATE_EDGE_LOG_LIKELIHOODS);
    out_.putI64(count);
    out_.putInts(parentBufferIndices, count);
    out_.putInts(childBufferIndices, count);
    out_.putInts(probabilityIndices, count);
    out_.putInts(firstDerivativeIndices, count);
    out_.putInts(secondDerivativeIndices, count);
    out_.putInts(categoryWeightsIndices, count);
    out_.putInts(stateFrequenciesIndices, count);
    out_.putInts(scalingFactorsIndices, count);
    return call([=](Reader& reader) {
        *outSumLogLikelihood = reader.f64();
        double first = reader.f64();
        double second = reader.f64();
        if (outSumFirstDerivative != NULL) *outSumFirstDerivative = first;
        if (outSumSecondDerivative != NULL) *outSumSecondDerivative = second;
    });
}

int BeagleRemoteImpl::calculateEdgeLogLikelihoodsByPartition(const int* parentBufferIndices,
                                                             const int* childBufferIndices,
                                                             const int* probabilityIndices,
                                                             const int* firstDerivativeIndices,
                                                             const int* secondDerivativeIndices,
                                                             const int* categoryWeightsIndices,
                                                             const int* stateFrequenciesIndices,
                                                             const int* cumulativeScaleIndices,
                                                             const int* partitionIndices, int partitionCount,
                                                             int count, double* outSumLogLikelihoodByPartition,
                                                             double* outSumLogLikelihood,
                                                             double* outSumFirstDerivativeByPartition,
                                                             double* outSumFirstDerivative,
                                                             double* outSumSecondDerivativeByPartition,
                                                             double* outSumSecondDerivative) {
    const size_t size = (size_t) partitionCount * count;
    begin(OP_CALCULATE_EDGE_LOG_LIKELIHOODS_BY_PARTITION);
    out_.putI64(partitionCount);
    out_.putI64(count);
    out_.putInts(parentBufferIndices, size);
    out_.putInts(childBufferIndices, size);
    out_.putInts(probabilityIndices, size);
    out_.putInts(firstDerivativeIndices, size);
    out_.putInts(secondDerivativeIndices, size);
    out_.putInts(categoryWeightsIndices, size);
    out_.putInts(stateFrequenciesIndices, size);
    out_.putInts(cumulativeScaleIndices, size);
    out_.putInts(partitionIndices, size);
    return call([=](Reader& reader) {
        reader.copyDoubles(outSumLogLikelihoodByPartition, size);
        *outSumLogLikelihood = reader.f64();
        reader.copyDoubles(outSumFirstDerivativeByPartition, size);
        double first = reader.f64();
        reader.copyDoubles(outSumSecondDerivativeByPartition, size);
        double second = reader.f64();
        if (outSumFirstDerivative != NULL) *outSumFirstDerivative = first;
        if (outSumSecondDerivative != NULL) *outSumSecondDerivative = second;
    });
}

int BeagleRemoteImpl::getLogLikelihood(double* outSumLogLikelihood) {
    begin(OP_GET_LOG_LIKELIHOOD);
    return call([=](Reader& reader) { *outSumLogLikelihood = reader.f64(); });
}

int BeagleRemoteImpl::getDerivatives(double* outSumFirstDerivative, double* outSumSecondDerivative) {
    begin(OP_GET_DERIVATIVES);
    return call([=](Reader& reader) {
        double first = reader.f64();
        double second = reader.f64();
        if (outSumFirstDerivative != NULL) *outSumFirstDerivative = first;
        if (outSumSecondDerivative != NULL) *outSumSecondDerivative = second;
    });
}

int BeagleRemoteImpl::getSiteLogLikelihoods(double* outLogLikelihoods) {
    begin(OP_GET_SITE_LOG_LIKELIHOODS);
    const size_t size = kPatternCount;
    return call([=](Reader& reader) { reader.copyDoubles(outLogLikelihoods, size); });
}

int BeagleRemoteImpl::getSiteDerivatives(double* outFirstDerivatives, double* outSecondDerivatives) {
    begin(OP_GET_SITE_DERIVATIVES);
    out_.putI64(outSecondDerivatives != NULL);
    const size_t size = kPatternCount;
    return call([=](Reader& reader) {
        reader.copyDoubles(outFirstDerivatives, size);
        if (outSecondDerivatives != NULL) reader.copyDoubles(outSecondDerivatives, size);
    });
}

// The calls below are not carried by the protocol yet

#define BEAGLE_REMOTE_NOT_CARRIED { return BEAGLE_ERROR_NO_IMPLEMENTATION; }

int BeagleRemoteImpl::updateTransitionMatricesGrad(const int*, const double*, int) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::calculateEdgeDerivatives(const int*, const int*, const int*, const int*, const int*,
                                               const int*, int, double*, double*,
                                               double*) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::calculateCrossProducts(const int*, const int*, const int*, const int*, const double*, int,
                                             double*, double*) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::calculateAdjointCrossProducts(const int*, const int*, const int*, const int*, const int*,
                                                    const int, const int, int, double*, double*, const int*,
                                                    const int*, const int) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::calculateAdjointDerivative(const BeagleBranchOperation*, const int, const int, const int,
                                                 const int, const int, double*,
                                                 double*) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::updateBastaPartials(const int*, int, const int*, int, int, int) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::updateBastaPartialsGrad(const int*, int, const int*, int, int,
                                              int) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::accumulateBastaPartials(const int*, int, const int*, int, const double*, const int, int,
                                              double*) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::accumulateBastaPartialsGrad(const int*, int, const int*, int, const double*, const int,
                                                  int, int, int, int, double*,
                                                  double*) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::uploadBastaSlabMetadata(const int*, int) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::getBastaSlabConstants(int*, int*) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::allocateBastaBuffers(int, int, int, int, int) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::allocateBastaGradBuffers(int) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::getBastaBuffer(int, double*) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::accumulateEigenBasisGradient(int, int, const double*, int, int,
                                                   double*) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::setBastaPopulationSizes(const double*, int) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::setBastaPopulationSizesBuffer(const double*, int, int) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::setCurrentPopulationSizeBuffer(int) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::updateBastaPartials_v5(const int*, int, const int*, int, int,
                                             int) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::updateBastaPartialsGrad_v5(const int*, int, const int*, int, int,
                                                 int) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::accumulateBastaPartials_v5(const int*, int, const int*, int, const double*, const int, int,
                                                 double*) BEAGLE_REMOTE_NOT_CARRIED
int BeagleRemoteImpl::accumulateBastaPartialsGrad_v5(const int*, int, const int*, int, const double*, const int,
                                                     int, int, int, int, double*,
                                                     double*) BEAGLE_REMOTE_NOT_CARRIED

#undef BEAGLE_REMOTE_NOT_CARRIED

BeagleImpl* BeagleRemoteImplFactory::createImpl(int tipCount, int partialsBufferCount, int compactBufferCount,
                                                int stateCount, int patternCount, int eigenBufferCount,
                                                int matrixBufferCount, int categoryCount, int scaleBufferCount,
                                                int resourceNumber, int pluginResourceNumber,
                                                long long preferenceFlags, long long requirementFlags,
                                                int* errorCode) {
    if (pluginResourceNumber < 0 || pluginResourceNumber >= (int) resources_.size()) {
        *errorCode = BEAGLE_ERROR_NO_RESOURCE;
        return NULL;
    }

    BeagleRemoteImpl* impl = new BeagleRemoteImpl(resources_[pluginResourceNumber]);
    try {
        *errorCode = impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                          patternCount, eigenBufferCount, matrixBufferCount, categoryCount,
                                          scaleBufferCount, resourceNumber, pluginResourceNumber,
                                          preferenceFlags, requirementFlags);
    } catch (...) {
        *errorCode = BEAGLE_ERROR_GENERAL;
    }
    if (*errorCode != BEAGLE_SUCCESS) {
        delete impl;
        return NULL;
    }
    return impl;
}

} // namespace remote
} // namespace beagle
