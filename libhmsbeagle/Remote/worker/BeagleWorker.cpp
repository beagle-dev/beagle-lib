/*
 *  BeagleWorker.cpp
 *  BEAGLE
 *
 * beagle-worker: lends this machine's BEAGLE resources to clients on other machines. A client that names this
 * worker in BEAGLE_REMOTE_HOSTS sees each of its resources in its own resource list (the hmsbeagle-remote
 * plugin), and an instance it creates on one of them is created here, through the public BEAGLE API, and
 * driven over the connection.
 *
 * Each connection has its own thread and owns at most one instance, which is finalized when the connection
 * closes, however it closes.
 *
 * Who can use the worker is limited in two ways. Unless told otherwise (--bind) it listens only on this machine's
 * Thunderbolt links and loopback addresses, so that a client has to be on the other end of a cable. And if it
 * is given a secret, a client has to show that it knows the same one (RemoteAuth.h); listening anywhere else
 * requires one.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include <signal.h>
#include <unistd.h>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/Remote/RemoteAuth.h"
#include "libhmsbeagle/Remote/RemoteChannel.h"
#include "libhmsbeagle/Remote/RemoteProtocol.h"
#include "libhmsbeagle/Remote/worker/LinkAddresses.h"

using namespace beagle::remote;

namespace {

bool verbose = false;

// What a client must show it knows before it is served; empty if this worker asks for nothing
std::string secret;

// The hello is a few hundred bytes. Nothing larger is read from a client that has not finished it.
const uint64_t kHelloLimitBytes = 4096;

// beagleCreateInstance and beagleFinalizeInstance change the library's table of instances, which every other
// call reads: they exclude all other calls, which otherwise run concurrently, one thread per instance
std::shared_mutex instanceTableMutex;

std::mutex logMutex;

void logLine(const std::string& peer, const std::string& text) {
    std::lock_guard<std::mutex> lock(logMutex);
    std::cerr << "[" << peer << "] " << text << std::endl;
}

/** The instance of one connection, with the sizes that say how long each array of a command must be. */
struct Session {
    int instance;
    size_t stateCount;
    size_t patternCount;
    size_t categoryCount;
    long long flags;

    Session() : instance(-1), stateCount(0), patternCount(0), categoryCount(0), flags(0) {}

    size_t partialsSize() const { return patternCount * stateCount * categoryCount; }
    size_t matrixSize() const { return stateCount * stateCount * categoryCount; }
};

int createInstance(Session& session, Reader& in, Buffer& out) {
    if (session.instance >= 0) throw ProtocolError("the connection already has an instance");

    int tipCount = in.i32();
    int partialsBufferCount = in.i32();
    int compactBufferCount = in.i32();
    int stateCount = in.i32();
    int patternCount = in.i32();
    int eigenBufferCount = in.i32();
    int matrixBufferCount = in.i32();
    int categoryCount = in.i32();
    int scaleBufferCount = in.i32();
    int resource = in.i32();
    long long preferenceFlags = in.i64();
    long long requirementFlags = in.i64();

    if (stateCount < 0 || patternCount < 0 || categoryCount < 0) return BEAGLE_ERROR_OUT_OF_RANGE;
    BeagleResourceList* resources = beagleGetResourceList();
    if (resource < 0 || resource >= resources->length) return BEAGLE_ERROR_NO_RESOURCE;

    BeagleInstanceDetails details;
    memset(&details, 0, sizeof(details));
    int instance;
    {
        std::unique_lock<std::shared_mutex> lock(instanceTableMutex);
        instance = beagleCreateInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                        patternCount, eigenBufferCount, matrixBufferCount, categoryCount,
                                        scaleBufferCount, &resource, 1, preferenceFlags, requirementFlags,
                                        &details);
    }
    if (instance < 0) return instance;

    session.instance = instance;
    session.stateCount = (size_t) stateCount;
    session.patternCount = (size_t) patternCount;
    session.categoryCount = (size_t) categoryCount;
    session.flags = details.flags;

    out.putI64(details.flags);
    out.putString(details.implName ? details.implName : "");
    return BEAGLE_SUCCESS;
}

/*
 * Runs one command: reads its arguments, calls BEAGLE, writes its outputs. Returns BEAGLE's return code.
 * The arrays are used where they lie in the message, having been checked to be as long as BEAGLE will read.
 */
int runCommand(Session& s, int opcode, Reader& in, Buffer& out) {
    if (opcode == OP_SYNC) return BEAGLE_SUCCESS;
    if (opcode == OP_CREATE_INSTANCE) return createInstance(s, in, out);
    if (s.instance < 0) throw ProtocolError("command before an instance was created");

    std::shared_lock<std::shared_mutex> lock(instanceTableMutex);
    const int instance = s.instance;

    switch (opcode) {
    case OP_SET_CPU_THREAD_COUNT: {
        int threadCount = in.i32();
        return beagleSetCPUThreadCount(instance, threadCount);
    }
    case OP_ENSURE_BUFFER_COUNTS: {
        int partialsBufferCount = in.i32();
        int matrixBufferCount = in.i32();
        int scaleBufferCount = in.i32();
        return beagleEnsureBufferCounts(instance, partialsBufferCount, matrixBufferCount, scaleBufferCount);
    }
    case OP_SET_TIP_STATES: {
        int tipIndex = in.i32();
        const int* states = in.ints(s.patternCount);
        return beagleSetTipStates(instance, tipIndex, states);
    }
    case OP_SET_TIP_PARTIALS: {
        int tipIndex = in.i32();
        const double* partials = in.doubles(s.patternCount * s.stateCount);
        return beagleSetTipPartials(instance, tipIndex, partials);
    }
    case OP_SET_PARTIALS: {
        int bufferIndex = in.i32();
        const double* partials = in.doubles(s.partialsSize());
        return beagleSetPartials(instance, bufferIndex, partials);
    }
    case OP_SET_ROOT_PRE_PARTIALS: {
        size_t count = in.count();
        const int* bufferIndices = in.ints(count);
        const int* stateFrequenciesIndices = in.ints(count);
        return beagleSetRootPrePartials(instance, bufferIndices, stateFrequenciesIndices, (int) count);
    }
    case OP_GET_PARTIALS: {
        int bufferIndex = in.i32();
        int scaleIndex = in.i32();
        std::vector<double> partials(s.partialsSize());
        int returnCode = beagleGetPartials(instance, bufferIndex, scaleIndex, partials.data());
        out.putDoubles(partials.data(), partials.size());
        return returnCode;
    }
    case OP_GET_PARTIALS_BATCH: {
        size_t count = in.count();
        const int* bufferIndices = in.ints(count);
        const int* scaleIndices = in.ints(count);
        std::vector<double> partials(count * s.partialsSize());
        int returnCode = beagleGetPartialsBatch(instance, bufferIndices, scaleIndices, (int) count,
                                                partials.data());
        out.putDoubles(partials.data(), partials.size());
        return returnCode;
    }
    case OP_SET_EIGEN_DECOMPOSITION: {
        int eigenIndex = in.i32();
        const double* eigenVectors = in.doubles(s.stateCount * s.stateCount);
        const double* inverseEigenVectors = in.doubles(s.stateCount * s.stateCount);
        const double* eigenValues =
                in.doubles((s.flags & BEAGLE_FLAG_EIGEN_COMPLEX) ? 2 * s.stateCount : s.stateCount);
        return beagleSetEigenDecomposition(instance, eigenIndex, eigenVectors, inverseEigenVectors, eigenValues);
    }
    case OP_SET_STATE_FREQUENCIES: {
        int index = in.i32();
        const double* frequencies = in.doubles(s.stateCount);
        return beagleSetStateFrequencies(instance, index, frequencies);
    }
    case OP_SET_CATEGORY_WEIGHTS: {
        int index = in.i32();
        const double* weights = in.doubles(s.categoryCount);
        return beagleSetCategoryWeights(instance, index, weights);
    }
    case OP_SET_PATTERN_WEIGHTS: {
        const double* weights = in.doubles(s.patternCount);
        return beagleSetPatternWeights(instance, weights);
    }
    case OP_SET_PATTERN_PARTITIONS: {
        int partitionCount = in.i32();
        const int* partitions = in.ints(s.patternCount);
        return beagleSetPatternPartitions(instance, partitionCount, partitions);
    }
    case OP_SET_CATEGORY_RATES: {
        const double* rates = in.doubles(s.categoryCount);
        return beagleSetCategoryRates(instance, rates);
    }
    case OP_SET_CATEGORY_RATES_WITH_INDEX: {
        int index = in.i32();
        const double* rates = in.doubles(s.categoryCount);
        return beagleSetCategoryRatesWithIndex(instance, index, rates);
    }
    case OP_SET_TRANSITION_MATRIX: {
        int matrixIndex = in.i32();
        const double* matrix = in.doubles(s.matrixSize());
        double paddedValue = in.f64();
        return beagleSetTransitionMatrix(instance, matrixIndex, matrix, paddedValue);
    }
    case OP_SET_DIFFERENTIAL_MATRIX: {
        int matrixIndex = in.i32();
        const double* matrix = in.doubles(s.matrixSize());
        return beagleSetDifferentialMatrix(instance, matrixIndex, matrix);
    }
    case OP_SET_TRANSITION_MATRICES: {
        size_t count = in.count();
        const int* matrixIndices = in.ints(count);
        const double* matrices = in.doubles(count * s.matrixSize());
        const double* paddedValues = in.doubles(count);
        return beagleSetTransitionMatrices(instance, matrixIndices, matrices, paddedValues, (int) count);
    }
    case OP_GET_TRANSITION_MATRIX: {
        int matrixIndex = in.i32();
        std::vector<double> matrix(s.matrixSize());
        int returnCode = beagleGetTransitionMatrix(instance, matrixIndex, matrix.data());
        out.putDoubles(matrix.data(), matrix.size());
        return returnCode;
    }
    case OP_CONVOLVE_TRANSITION_MATRICES: {
        size_t count = in.count();
        const int* firstIndices = in.ints(count);
        const int* secondIndices = in.ints(count);
        const int* resultIndices = in.ints(count);
        return beagleConvolveTransitionMatrices(instance, firstIndices, secondIndices, resultIndices, (int) count);
    }
    case OP_ADD_TRANSITION_MATRICES: {
        size_t count = in.count();
        const int* firstIndices = in.ints(count);
        const int* secondIndices = in.ints(count);
        const int* resultIndices = in.ints(count);
        return beagleAddTransitionMatrices(instance, firstIndices, secondIndices, resultIndices, (int) count);
    }
    case OP_TRANSPOSE_TRANSITION_MATRICES: {
        size_t count = in.count();
        const int* inputIndices = in.ints(count);
        const int* resultIndices = in.ints(count);
        return beagleTransposeTransitionMatrices(instance, inputIndices, resultIndices, (int) count);
    }
    case OP_UPDATE_TRANSITION_MATRICES: {
        int eigenIndex = in.i32();
        size_t count = in.count();
        const int* probabilityIndices = in.ints(count);
        const int* firstDerivativeIndices = in.optionalInts(count);
        const int* secondDerivativeIndices = in.optionalInts(count);
        const double* edgeLengths = in.doubles(count);
        return beagleUpdateTransitionMatrices(instance, eigenIndex, probabilityIndices, firstDerivativeIndices,
                                              secondDerivativeIndices, edgeLengths, (int) count);
    }
    case OP_UPDATE_TRANSITION_MATRICES_WITH_MODEL_CATEGORIES: {
        size_t count = in.count();
        const int* eigenIndices = in.ints(s.categoryCount);
        const int* probabilityIndices = in.ints(count);
        const int* firstDerivativeIndices = in.optionalInts(count);
        const int* secondDerivativeIndices = in.optionalInts(count);
        const double* edgeLengths = in.doubles(count);
        // the API takes the eigen indices without const
        std::vector<int> eigenIndicesCopy(eigenIndices, eigenIndices + s.categoryCount);
        return beagleUpdateTransitionMatricesWithModelCategories(instance, eigenIndicesCopy.data(),
                                                                 probabilityIndices, firstDerivativeIndices,
                                                                 secondDerivativeIndices, edgeLengths,
                                                                 (int) count);
    }
    case OP_UPDATE_TRANSITION_MATRICES_WITH_MULTIPLE_MODELS: {
        size_t count = in.count();
        const int* eigenIndices = in.ints(count);
        const int* categoryRateIndices = in.ints(count);
        const int* probabilityIndices = in.ints(count);
        const int* firstDerivativeIndices = in.optionalInts(count);
        const int* secondDerivativeIndices = in.optionalInts(count);
        const double* edgeLengths = in.doubles(count);
        return beagleUpdateTransitionMatricesWithMultipleModels(instance, eigenIndices, categoryRateIndices,
                                                                probabilityIndices, firstDerivativeIndices,
                                                                secondDerivativeIndices, edgeLengths,
                                                                (int) count);
    }
    case OP_UPDATE_PARTIALS: {
        size_t count = in.count();
        const int* operations = in.ints(count * BEAGLE_OP_COUNT);
        int cumulativeScaleIndex = in.i32();
        BeaglePartialsType partialsType = (BeaglePartialsType) in.i64();
        return beagleUpdatePartials_v5(instance, (const BeagleOperation*) operations, (int) count,
                                       cumulativeScaleIndex, partialsType);
    }
    case OP_UPDATE_PRE_PARTIALS: {
        size_t count = in.count();
        const int* operations = in.ints(count * BEAGLE_OP_COUNT);
        int cumulativeScaleIndex = in.i32();
        BeaglePartialsType partialsType = (BeaglePartialsType) in.i64();
        return beagleUpdatePrePartials_v5(instance, (const BeagleOperation*) operations, (int) count,
                                          cumulativeScaleIndex, partialsType);
    }
    case OP_UPDATE_PARTIALS_BY_PARTITION: {
        size_t count = in.count();
        const int* operations = in.ints(count * BEAGLE_PARTITION_OP_COUNT);
        return beagleUpdatePartialsByPartition(instance, (const BeagleOperationByPartition*) operations,
                                               (int) count);
    }
    case OP_UPDATE_PRE_PARTIALS_BY_PARTITION: {
        size_t count = in.count();
        const int* operations = in.ints(count * BEAGLE_PARTITION_OP_COUNT);
        BeaglePartialsType partialsType = (BeaglePartialsType) in.i64();
        return beagleUpdatePrePartialsByPartition_v5(instance, (const BeagleOperationByPartition*) operations,
                                                     (int) count, partialsType);
    }
    case OP_WAIT_FOR_PARTIALS: {
        size_t count = in.count();
        const int* destinationPartials = in.ints(count);
        return beagleWaitForPartials(instance, destinationPartials, (int) count);
    }
    case OP_ACCUMULATE_SCALE_FACTORS: {
        size_t count = in.count();
        const int* scaleIndices = in.ints(count);
        int cumulativeScaleIndex = in.i32();
        return beagleAccumulateScaleFactors(instance, scaleIndices, (int) count, cumulativeScaleIndex);
    }
    case OP_ACCUMULATE_SCALE_FACTORS_BY_PARTITION: {
        size_t count = in.count();
        const int* scaleIndices = in.ints(count);
        int cumulativeScaleIndex = in.i32();
        int partitionIndex = in.i32();
        return beagleAccumulateScaleFactorsByPartition(instance, scaleIndices, (int) count,
                                                       cumulativeScaleIndex, partitionIndex);
    }
    case OP_REMOVE_SCALE_FACTORS: {
        size_t count = in.count();
        const int* scaleIndices = in.ints(count);
        int cumulativeScaleIndex = in.i32();
        return beagleRemoveScaleFactors(instance, scaleIndices, (int) count, cumulativeScaleIndex);
    }
    case OP_REMOVE_SCALE_FACTORS_BY_PARTITION: {
        size_t count = in.count();
        const int* scaleIndices = in.ints(count);
        int cumulativeScaleIndex = in.i32();
        int partitionIndex = in.i32();
        return beagleRemoveScaleFactorsByPartition(instance, scaleIndices, (int) count, cumulativeScaleIndex,
                                                   partitionIndex);
    }
    case OP_RESET_SCALE_FACTORS: {
        int cumulativeScaleIndex = in.i32();
        return beagleResetScaleFactors(instance, cumulativeScaleIndex);
    }
    case OP_RESET_SCALE_FACTORS_BY_PARTITION: {
        int cumulativeScaleIndex = in.i32();
        int partitionIndex = in.i32();
        return beagleResetScaleFactorsByPartition(instance, cumulativeScaleIndex, partitionIndex);
    }
    case OP_COPY_SCALE_FACTORS: {
        int destScalingIndex = in.i32();
        int srcScalingIndex = in.i32();
        return beagleCopyScaleFactors(instance, destScalingIndex, srcScalingIndex);
    }
    case OP_GET_SCALE_FACTORS: {
        int srcScalingIndex = in.i32();
        std::vector<double> scaleFactors(s.patternCount);
        int returnCode = beagleGetScaleFactors(instance, srcScalingIndex, scaleFactors.data());
        out.putDoubles(scaleFactors.data(), scaleFactors.size());
        return returnCode;
    }
    case OP_CALCULATE_ROOT_LOG_LIKELIHOODS: {
        size_t count = in.count();
        const int* bufferIndices = in.ints(count);
        const int* categoryWeightsIndices = in.ints(count);
        const int* stateFrequenciesIndices = in.ints(count);
        const int* cumulativeScaleIndices = in.ints(count);
        double sumLogLikelihood = 0.0;
        int returnCode = beagleCalculateRootLogLikelihoods(instance, bufferIndices, categoryWeightsIndices,
                                                           stateFrequenciesIndices, cumulativeScaleIndices,
                                                           (int) count, &sumLogLikelihood);
        out.putF64(sumLogLikelihood);
        return returnCode;
    }
    case OP_CALCULATE_ROOT_LOG_LIKELIHOODS_BY_PARTITION: {
        size_t partitionCount = in.count();
        size_t count = in.count();
        size_t size = partitionCount * count;
        const int* bufferIndices = in.ints(size);
        const int* categoryWeightsIndices = in.ints(size);
        const int* stateFrequenciesIndices = in.ints(size);
        const int* cumulativeScaleIndices = in.ints(size);
        const int* partitionIndices = in.ints(size);
        std::vector<double> byPartition(size);
        double sumLogLikelihood = 0.0;
        int returnCode = beagleCalculateRootLogLikelihoodsByPartition(
                instance, bufferIndices, categoryWeightsIndices, stateFrequenciesIndices, cumulativeScaleIndices,
                partitionIndices, (int) partitionCount, (int) count, byPartition.data(), &sumLogLikelihood);
        out.putDoubles(byPartition.data(), byPartition.size());
        out.putF64(sumLogLikelihood);
        return returnCode;
    }
    case OP_CALCULATE_EDGE_LOG_LIKELIHOODS: {
        size_t count = in.count();
        const int* parentBufferIndices = in.ints(count);
        const int* childBufferIndices = in.ints(count);
        const int* probabilityIndices = in.ints(count);
        const int* firstDerivativeIndices = in.optionalInts(count);
        const int* secondDerivativeIndices = in.optionalInts(count);
        const int* categoryWeightsIndices = in.ints(count);
        const int* stateFrequenciesIndices = in.ints(count);
        const int* cumulativeScaleIndices = in.ints(count);
        double sumLogLikelihood = 0.0, sumFirstDerivative = 0.0, sumSecondDerivative = 0.0;
        int returnCode = beagleCalculateEdgeLogLikelihoods(
                instance, parentBufferIndices, childBufferIndices, probabilityIndices, firstDerivativeIndices,
                secondDerivativeIndices, categoryWeightsIndices, stateFrequenciesIndices, cumulativeScaleIndices,
                (int) count, &sumLogLikelihood, &sumFirstDerivative, &sumSecondDerivative);
        out.putF64(sumLogLikelihood);
        out.putF64(sumFirstDerivative);
        out.putF64(sumSecondDerivative);
        return returnCode;
    }
    case OP_CALCULATE_EDGE_LOG_LIKELIHOODS_BY_PARTITION: {
        size_t partitionCount = in.count();
        size_t count = in.count();
        size_t size = partitionCount * count;
        const int* parentBufferIndices = in.ints(size);
        const int* childBufferIndices = in.ints(size);
        const int* probabilityIndices = in.ints(size);
        const int* firstDerivativeIndices = in.optionalInts(size);
        const int* secondDerivativeIndices = in.optionalInts(size);
        const int* categoryWeightsIndices = in.ints(size);
        const int* stateFrequenciesIndices = in.ints(size);
        const int* cumulativeScaleIndices = in.ints(size);
        const int* partitionIndices = in.ints(size);
        std::vector<double> logLikelihoods(size), firstDerivatives(size), secondDerivatives(size);
        double sumLogLikelihood = 0.0, sumFirstDerivative = 0.0, sumSecondDerivative = 0.0;
        int returnCode = beagleCalculateEdgeLogLikelihoodsByPartition(
                instance, parentBufferIndices, childBufferIndices, probabilityIndices, firstDerivativeIndices,
                secondDerivativeIndices, categoryWeightsIndices, stateFrequenciesIndices, cumulativeScaleIndices,
                partitionIndices, (int) partitionCount, (int) count, logLikelihoods.data(), &sumLogLikelihood,
                firstDerivatives.data(), &sumFirstDerivative, secondDerivatives.data(), &sumSecondDerivative);
        out.putDoubles(logLikelihoods.data(), size);
        out.putF64(sumLogLikelihood);
        out.putDoubles(firstDerivatives.data(), size);
        out.putF64(sumFirstDerivative);
        out.putDoubles(secondDerivatives.data(), size);
        out.putF64(sumSecondDerivative);
        return returnCode;
    }
    case OP_GET_LOG_LIKELIHOOD: {
        double sumLogLikelihood = 0.0;
        int returnCode = beagleGetLogLikelihood(instance, &sumLogLikelihood);
        out.putF64(sumLogLikelihood);
        return returnCode;
    }
    case OP_GET_DERIVATIVES: {
        double sumFirstDerivative = 0.0, sumSecondDerivative = 0.0;
        int returnCode = beagleGetDerivatives(instance, &sumFirstDerivative, &sumSecondDerivative);
        out.putF64(sumFirstDerivative);
        out.putF64(sumSecondDerivative);
        return returnCode;
    }
    case OP_GET_SITE_LOG_LIKELIHOODS: {
        std::vector<double> logLikelihoods(s.patternCount);
        int returnCode = beagleGetSiteLogLikelihoods(instance, logLikelihoods.data());
        out.putDoubles(logLikelihoods.data(), logLikelihoods.size());
        return returnCode;
    }
    case OP_GET_SITE_DERIVATIVES: {
        bool wantSecond = in.i64() != 0;
        std::vector<double> firstDerivatives(s.patternCount), secondDerivatives(wantSecond ? s.patternCount : 0);
        int returnCode = beagleGetSiteDerivatives(instance, firstDerivatives.data(),
                                                  wantSecond ? secondDerivatives.data() : NULL);
        out.putDoubles(firstDerivatives.data(), firstDerivatives.size());
        if (wantSecond) out.putDoubles(secondDerivatives.data(), secondDerivatives.size());
        return returnCode;
    }
    default:
        throw ProtocolError("unknown command");
    }
}

void serve(Channel* channel, std::string peer) {
    Session session;
    Buffer request, response, outputs;
    size_t batches = 0;

    try {
        channel->limitReceive(kHelloLimitBytes);
        if (!channel->receive(request)) throw ProtocolError("closed before hello");
        {
            Reader hello(request);
            if (hello.i64() != kMagic) throw ProtocolError("not a BEAGLE client");
            if (hello.i64() != kProtocolVersion) {
                // answered, so that the client can say what is wrong
                response.putI64(kMagic);
                response.putI64(kProtocolVersion);
                channel->send(response);
                throw ProtocolError("client speaks another protocol version");
            }
            std::string clientNonce = hello.string();
            if (clientNonce.size() != kNonceBytes) throw ProtocolError("malformed hello");
            std::string workerNonce = randomNonce();

            response.putI64(kMagic);
            response.putI64(kProtocolVersion);
            response.putI64(secret.empty() ? 0 : 1);
            response.putString(workerNonce);
            if (!channel->send(response)) throw ProtocolError("closed during hello");

            if (!channel->receive(request)) throw ProtocolError("closed during hello");
            Reader proof(request);
            bool known = secret.empty() ||
                         sameBytes(proof.string(), proofOfSecret(secret, "client", clientNonce, workerNonce));

            // the resource list, and this worker's own proof, only for a client that knows the secret
            response.clear();
            response.putI64(known ? 0 : 1);
            if (known) {
                response.putString(secret.empty() ? std::string()
                                                  : proofOfSecret(secret, "worker", clientNonce, workerNonce));
                BeagleResourceList* resources = beagleGetResourceList();
                response.putI64(resources->length);
                for (int i = 0; i < resources->length; i++) {
                    response.putString(resources->list[i].name ? resources->list[i].name : "");
                    response.putString(resources->list[i].description ? resources->list[i].description : "");
                    response.putI64(resources->list[i].supportFlags);
                    response.putI64(resources->list[i].requiredFlags);
                }
            }
            if (!channel->send(response)) throw ProtocolError("closed during hello");
            if (!known) throw ProtocolError("client does not have the secret");
        }
        channel->limitReceive(Channel::kMaxMessageBytes);

        while (channel->receive(request)) {
            Reader in(request);
            int deferredError = BEAGLE_SUCCESS;
            int status = BEAGLE_SUCCESS;
            while (!in.atEnd()) {
                int opcode = in.i32();
                outputs.clear();
                bool creating = (opcode == OP_CREATE_INSTANCE);
                status = runCommand(session, opcode, in, outputs);
                if (!in.atEnd() && status != BEAGLE_SUCCESS && deferredError == BEAGLE_SUCCESS)
                    deferredError = status;
                if (creating && verbose)
                    logLine(peer, status == BEAGLE_SUCCESS
                                    ? "created instance " + std::to_string(session.instance)
                                    : "could not create an instance (" + std::to_string(status) + ")");
            }

            response.clear();
            response.putI64(deferredError);
            response.putI64(status);
            if (status == BEAGLE_SUCCESS && outputs.size() > 0) {
                size_t offset = response.size();
                response.resize(offset + outputs.size());
                memcpy(response.data() + offset, outputs.data(), outputs.size());
            }
            if (!channel->send(response)) break;
            batches++;
        }
    } catch (const ProtocolError& e) {
        logLine(peer, std::string("dropped: ") + e.what());
    } catch (const std::exception& e) {
        logLine(peer, std::string("dropped: ") + e.what());
    }

    if (session.instance >= 0) {
        std::unique_lock<std::shared_mutex> lock(instanceTableMutex);
        beagleFinalizeInstance(session.instance);
    }
    if (verbose && session.instance >= 0)
        logLine(peer, "finalized instance " + std::to_string(session.instance) + " after " +
                      std::to_string(batches) + " batches");
    delete channel;
}

void usage() {
    std::cerr << "Usage: beagle-worker [--port <port>] [--bind <address>]... [--secret-file <file>]\n"
              << "                     [--allow-unauthenticated] [--verbose]\n"
              << "\n"
              << "Lends this machine's BEAGLE resources to clients on other machines. On the client, name\n"
              << "this worker in the environment variable " << kHostsVariable << " (host[:port],host[:port],...);\n"
              << "its resources then appear in the client's BEAGLE resource list.\n"
              << "\n"
              << "  --port <port>         port to listen on (default " << kDefaultPort << ")\n"
              << "  --bind <address>      listen on this address instead of the default, which is this\n"
              << "                        machine's Thunderbolt links and loopback addresses only. 'any' is\n"
              << "                        every interface. Needs a secret. May be given more than once.\n"
              << "  --secret-file <file>  serve only clients that know the secret in this file, which only\n"
              << "                        its owner may be able to read. The client is given the same secret\n"
              << "                        in " << kSecretFileVariable << " or " << kSecretVariable << ", which\n"
              << "                        this worker also reads if there is no --secret-file.\n"
              << "  --allow-unauthenticated\n"
              << "                        let --bind be used without a secret: anyone who can reach the\n"
              << "                        address can then use this machine\n"
              << "  --verbose             report each instance as it is created and finalized\n"
              << "\n"
              << "The secret says who may connect. It does not encrypt the connection.\n";
}

std::string join(const std::vector<std::string>& items) {
    std::string text;
    for (size_t i = 0; i < items.size(); i++) text += (i ? ", " : "") + (items[i].empty() ? "*" : items[i]);
    return text.empty() ? "nothing" : text;
}

} // anonymous namespace

int main(int argc, char** argv) {
    int port = kDefaultPort;
    std::vector<std::string> bound;
    std::string secretFile;
    bool allowUnauthenticated = false;

    for (int i = 1; i < argc; i++) {
        std::string argument = argv[i];
        if (argument == "--port" && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (argument == "--bind" && i + 1 < argc) {
            std::string address = argv[++i];
            bound.push_back(address == "any" ? "" : address);
        } else if (argument == "--secret-file" && i + 1 < argc) {
            secretFile = argv[++i];
        } else if (argument == "--allow-unauthenticated") {
            allowUnauthenticated = true;
        } else if (argument == "--verbose") {
            verbose = true;
        } else {
            usage();
            return (argument == "--help" || argument == "-h") ? 0 : 1;
        }
    }

    std::string error;
    if (!(secretFile.empty() ? secretFromEnvironment(&secret, &error) : readSecretFile(secretFile, &secret, &error))) {
        std::cerr << "beagle-worker: " << error << std::endl;
        return 1;
    }
    // the instances' threads and plugins have no need of it
    unsetenv(kSecretVariable);
    unsetenv(kSecretFileVariable);

    bool cableOnly = bound.empty();
    if (!cableOnly && secret.empty() && !allowUnauthenticated) {
        std::cerr << "beagle-worker: --bind needs a secret (--secret-file), since anyone who can reach the\n"
                  << "address could otherwise use this machine. --allow-unauthenticated overrides this." << std::endl;
        return 1;
    }

    // a client that disappears is found by its connection failing, not by a signal
    signal(SIGPIPE, SIG_IGN);

    // this process lends its own resources; it must not list those of other workers and lend them on
    unsetenv(kHostsVariable);

    // loads the plugins, before there are threads
    BeagleResourceList* resources = beagleGetResourceList();

    TcpListener listener;
    std::vector<std::string> listening;
    if (cableOnly) {
        listener.listenOn(linkAddresses(thunderboltInterfaces()), port, NULL);
        listening = listener.addresses();
    } else {
        listener.listenOn(bound, port, &error);
        listening = listener.addresses();
        if (listening.size() != bound.size()) {
            std::cerr << "beagle-worker: cannot listen on port " << port << " of " << error << std::endl;
            return 1;
        }
    }

    std::cerr << "beagle-worker (BEAGLE " << beagleGetVersion() << "), port " << port << ", lending:" << std::endl;
    for (int i = 0; i < resources->length; i++)
        std::cerr << "  " << i << " : " << resources->list[i].name << std::endl;
    std::cerr << (secret.empty() ? "Serving any client that can connect (no secret)."
                                 : "Serving only clients that know the secret.") << std::endl;
    if (cableOnly) {
        if (canFindThunderboltInterfaces()) {
            std::set<std::string> links = thunderboltInterfaces();
            std::cerr << "Listening only on loopback and Thunderbolt links ("
                      << join(std::vector<std::string>(links.begin(), links.end()))
                      << "); their addresses are picked up as cables are connected." << std::endl;
        } else
            std::cerr << "Listening only on loopback: Thunderbolt links cannot be identified on this system.\n"
                      << "Use --bind with a secret to serve other machines." << std::endl;
    }
    std::cerr << "Listening on: " << join(listening) << std::endl;

    for (;;) {
        std::string peer;
        // in cable-only mode, come round every two seconds to see whether a link has come or gone
        Channel* channel = listener.accept(&peer, cableOnly ? 2000 : -1);
        if (channel != NULL) {
            std::thread(serve, channel, peer).detach();
            continue;
        }
        if (listener.addresses().empty() && !cableOnly) {
            std::cerr << "beagle-worker: no longer listening" << std::endl;
            return 1;
        }
        if (cableOnly) {
            listener.listenOn(linkAddresses(thunderboltInterfaces()), port, NULL);
            std::vector<std::string> now = listener.addresses();
            if (now != listening) {
                listening = now;
                std::cerr << "Listening on: " << join(listening) << std::endl;
            }
        }
    }
}
