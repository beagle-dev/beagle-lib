/*
 *  beagle.cpp
 *  BEAGLE
 *
 * @author Andrew Rambaut
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>    // for exception, bad_exception
#include <stdexcept>    // for std exception hierarchy
#include <list>
#include <vector>
#include <iostream>

#include "beagle.h"
#include "BeagleImpl.h"

#ifdef CUDA
    #include "CUDA/BeagleCUDAImpl.h"
#endif
#include "CPU/BeagleCPUImpl.h"

//@CHANGED make this a std::vector<BeagleImpl *> and use at to reference.
std::vector<BeagleImpl*> instances;

/// returns an initialized instance or NULL if the index refers to an invalid instance
BeagleImpl* getBeagleInstance(int instanceIndex);


BeagleImpl* getBeagleInstance(int instanceIndex) {
    if (instanceIndex > instances.size())
        return NULL;
    return instances[instanceIndex];
}

std::list<BeagleImplFactory*> implFactory;

ResourceList* getResourceList() {
    return NULL;
}

int createInstance(int tipCount,
                   int partialsBufferCount,
                   int compactBufferCount,
                   int stateCount,
                   int patternCount,
                   int eigenBufferCount,
                   int matrixBufferCount,
                   int* resourceList,
                   int resourceCount,
                   long preferenceFlags,
                   long requirementFlags) {
    try {
        // Set-up a list of implementation factories in trial-order
        if (implFactory.size() == 0) {
#ifdef CUDA
            implFactory.push_back(new BeagleCUDAImplFactory());
#endif
            implFactory.push_back(new BeagleCPUImplFactory());
        }

        // Try each implementation
        for(std::list<BeagleImplFactory*>::iterator factory = implFactory.begin();
            factory != implFactory.end(); factory++) {
            fprintf(stderr, "BEAGLE bootstrap: %s - ", (*factory)->getName());

            BeagleImpl* beagle = (*factory)->createImpl(tipCount, partialsBufferCount,
                                                        compactBufferCount, stateCount,
                                                        patternCount, eigenBufferCount,
                                                        matrixBufferCount);

            if (beagle != NULL) {
                fprintf(stderr, "Success\n");
                int instance = instances.size();
                instances.push_back(beagle);
                return instance;
            }
            fprintf(stderr, "Failed\n");
        }

        // No implementations found or appropriate
        return GENERAL_ERROR;
    }
    catch (std::bad_alloc &) {
        return OUT_OF_MEMORY_ERROR;
    }
    catch (std::out_of_range &) {
        return OUT_OF_RANGE_ERROR;
    }
    catch (...) {
        return UNIDENTIFIED_EXCEPTION_ERROR;
    }
}

int initializeInstance(int instance,
                       InstanceDetails* returnInfo) {
    try {
        BeagleImpl* beagleInstance = getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return UNINITIALIZED_INSTANCE_ERROR;
        return beagleInstance->initializeInstance(returnInfo);
    }
    catch (std::bad_alloc &) {
        return OUT_OF_MEMORY_ERROR;
    }
    catch (std::out_of_range &) {
        return OUT_OF_RANGE_ERROR;
    }
    catch (...) {
        return UNIDENTIFIED_EXCEPTION_ERROR;
    }
}

int finalize(int instance) {
    try {
        BeagleImpl* beagleInstance = getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return UNINITIALIZED_INSTANCE_ERROR;
        delete beagleInstance;
        instances[instance] = 0L;
        return NO_ERROR;
    }
    catch (std::bad_alloc &) {
        return OUT_OF_MEMORY_ERROR;
    }
    catch (std::out_of_range &) {
        return OUT_OF_RANGE_ERROR;
    }
    catch (...) {
        return UNIDENTIFIED_EXCEPTION_ERROR;
    }
}

int setPartials(int instance,
                int bufferIndex,
                const double* inPartials) {
    try {
        BeagleImpl* beagleInstance = getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return UNINITIALIZED_INSTANCE_ERROR;
        return beagleInstance->setPartials(bufferIndex, inPartials);
    }
    catch (std::bad_alloc &) {
        return OUT_OF_MEMORY_ERROR;
    }
    catch (std::out_of_range &) {
        return OUT_OF_RANGE_ERROR;
    }
    catch (...) {
        return UNIDENTIFIED_EXCEPTION_ERROR;
    }
}

int getPartials(int instance, int bufferIndex, double* outPartials) {
    try {
        BeagleImpl* beagleInstance = getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return UNINITIALIZED_INSTANCE_ERROR;
        return beagleInstance->getPartials(bufferIndex, outPartials);
    }
    catch (std::bad_alloc &) {
        return OUT_OF_MEMORY_ERROR;
    }
    catch (std::out_of_range &) {
        return OUT_OF_RANGE_ERROR;
    }
    catch (...) {
        return UNIDENTIFIED_EXCEPTION_ERROR;
    }
}

int setTipStates(int instance,
                 int tipIndex,
                 const int* inStates) {
    try {
        BeagleImpl* beagleInstance = getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return UNINITIALIZED_INSTANCE_ERROR;

        return beagleInstance->setTipStates(tipIndex, inStates);
    }
    catch (std::bad_alloc &) {
        return OUT_OF_MEMORY_ERROR;
    }
    catch (std::out_of_range &) {
        return OUT_OF_RANGE_ERROR;
    }
    catch (...) {
        return UNIDENTIFIED_EXCEPTION_ERROR;
    }
}

int setEigenDecomposition(int instance,
                          int eigenIndex,
                          const double* inEigenVectors,
                          const double* inInverseEigenVectors,
                          const double* inEigenValues) {
    try {
        BeagleImpl* beagleInstance = getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return UNINITIALIZED_INSTANCE_ERROR;

        return beagleInstance->setEigenDecomposition(eigenIndex, inEigenVectors,
                                                     inInverseEigenVectors, inEigenValues);
    }
    catch (std::bad_alloc &) {
        return OUT_OF_MEMORY_ERROR;
    }
    catch (std::out_of_range &) {
        return OUT_OF_RANGE_ERROR;
    }
    catch (...) {
        return UNIDENTIFIED_EXCEPTION_ERROR;
    }
}

int setTransitionMatrix(int instance,
                        int matrixIndex,
                        const double* inMatrix) {
    try {
        BeagleImpl* beagleInstance = getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return UNINITIALIZED_INSTANCE_ERROR;

        return beagleInstance->setTransitionMatrix(matrixIndex, inMatrix);
    }
    catch (std::bad_alloc &) {
        return OUT_OF_MEMORY_ERROR;
    }
    catch (std::out_of_range &) {
        return OUT_OF_RANGE_ERROR;
    }
    catch (...) {
        return UNIDENTIFIED_EXCEPTION_ERROR;
    }
}

int updateTransitionMatrices(int instance,
                             int eigenIndex,
                             const int* probabilityIndices,
                             const int* firstDerivativeIndices,
                             const int* secondDervativeIndices,
                             const double* edgeLengths,
                             int count) {
    try {
        BeagleImpl* beagleInstance = getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return UNINITIALIZED_INSTANCE_ERROR;
        return beagleInstance->updateTransitionMatrices(eigenIndex, probabilityIndices,
                                                        firstDerivativeIndices,
                                                        secondDervativeIndices, edgeLengths, count);
    }
    catch (std::bad_alloc &) {
        return OUT_OF_MEMORY_ERROR;
    }
    catch (std::out_of_range &) {
        return OUT_OF_RANGE_ERROR;
    }
    catch (...) {
        return UNIDENTIFIED_EXCEPTION_ERROR;
    }
}

int updatePartials(const int* instanceList,
                   int instanceCount,
                   const int* operations,
                   int operationCount,
                   int rescale) {
    try {
        int error_code = NO_ERROR;
        for (int i = 0; i < instanceCount; i++) {
            BeagleImpl* beagleInstance = getBeagleInstance(instanceList[i]);
            if (beagleInstance == NULL)
                return UNINITIALIZED_INSTANCE_ERROR;

            int err = beagleInstance->updatePartials(operations, operationCount, rescale);
            if (err != NO_ERROR) {
                error_code = err;
            }
        }
        return error_code;
    }
    catch (std::bad_alloc &) {
        return OUT_OF_MEMORY_ERROR;
    }
    catch (std::out_of_range &) {
        return OUT_OF_RANGE_ERROR;
    }
    catch (...) {
        return UNIDENTIFIED_EXCEPTION_ERROR;
    }
}

int waitForPartials(const int* instanceList,
                    int instanceCount,
                    const int* destinationPartials,
                    int destinationPartialsCount) {
    try {
        int error_code = NO_ERROR;
        for (int i = 0; i < instanceCount; i++) {
            BeagleImpl* beagleInstance = getBeagleInstance(instanceList[i]);
            if (beagleInstance == NULL)
                return UNINITIALIZED_INSTANCE_ERROR;
            
            int err = beagleInstance->waitForPartials(destinationPartials,
                                                      destinationPartialsCount);
            if (err != NO_ERROR) {
                error_code = err;
            }
        }
        return error_code;
    }
    catch (std::bad_alloc &) {
        return OUT_OF_MEMORY_ERROR;
    }
    catch (std::out_of_range &) {
        return OUT_OF_RANGE_ERROR;
    }
    catch (...) {
        return UNIDENTIFIED_EXCEPTION_ERROR;
    }
}

int calculateRootLogLikelihoods(int instance,
                                const int* bufferIndices,
                                const double* weights,
                                const double* stateFrequencies,
                                int count,
                                double* outLogLikelihoods) {
    try {
        BeagleImpl* beagleInstance = getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return UNINITIALIZED_INSTANCE_ERROR;

        return beagleInstance->calculateRootLogLikelihoods(bufferIndices, weights, stateFrequencies,
                                                           count, outLogLikelihoods);
    }
    catch (std::bad_alloc &) {
        return OUT_OF_MEMORY_ERROR;
    }
    catch (std::out_of_range &) {
        return OUT_OF_RANGE_ERROR;
    }
    catch (...) {
        return UNIDENTIFIED_EXCEPTION_ERROR;
    }


}

int calculateEdgeLogLikelihoods(int instance,
                                const int* parentBufferIndices,
                                const int* childBufferIndices,
                                const int* probabilityIndices,
                                const int* firstDerivativeIndices,
                                const int* secondDerivativeIndices,
                                const double* weights,
                                const double* stateFrequencies,
                                int count,
                                double* outLogLikelihoods,
                                double* outFirstDerivatives,
                                double* outSecondDerivatives) {
    try {
        BeagleImpl* beagleInstance = getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return UNINITIALIZED_INSTANCE_ERROR;

        return beagleInstance->calculateEdgeLogLikelihoods(parentBufferIndices, childBufferIndices,
                                                           probabilityIndices,
                                                           firstDerivativeIndices,
                                                           secondDerivativeIndices, weights,
                                                           stateFrequencies, count,
                                                           outLogLikelihoods, outFirstDerivatives,
                                                           outSecondDerivatives);
    }
    catch (std::bad_alloc &) {
        return OUT_OF_MEMORY_ERROR;
    }
    catch (std::out_of_range &) {
        return OUT_OF_RANGE_ERROR;
    }
    catch (...) {
        return UNIDENTIFIED_EXCEPTION_ERROR;
    }
}

