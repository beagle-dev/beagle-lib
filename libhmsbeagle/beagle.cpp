/*
 *  beagle.cpp
 *  BEAGLE
 *
 * @author Andrew Rambaut
 * @author Marc Suchard
 * @author Daniel Ayres
 * @author Aaron Darling
 */
#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>    // for exception, bad_exception
#include <stdexcept>    // for std exception hierarchy
#include <list>
#include <vector>
#include <iostream>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/BeagleImpl.h"

#if defined(CUDA) || defined(OPENCL)
    #include "libhmsbeagle/GPU/BeagleGPUImpl.h"
#endif
#include "libhmsbeagle/CPU/BeagleCPU4StateImpl.h"
#include "libhmsbeagle/CPU/BeagleCPUImpl.h"

//@CHANGED make this a std::vector<BeagleImpl *> and use at to reference.
std::vector<beagle::BeagleImpl*> instances;

/// returns an initialized instance or NULL if the index refers to an invalid instance
namespace beagle {
BeagleImpl* getBeagleInstance(int instanceIndex);


BeagleImpl* getBeagleInstance(int instanceIndex) {
    if (instanceIndex > instances.size())
        return NULL;
    return instances[instanceIndex];
}

}	// end namespace beagle

std::list<beagle::BeagleImplFactory*> implFactory;

ResourceList* getResourceList() {

    ResourceList* rList;
    rList = (ResourceList*) malloc(sizeof(ResourceList));
    rList->length = 1;
    
#if defined(CUDA) || defined(OPENCL)
    GPUInterface* gpu = new GPUInterface;
    int gpuDeviceCount = gpu->GetDeviceCount();
    rList->length += gpuDeviceCount;
    rList->list = (Resource*) malloc(sizeof(Resource) * rList->length); 
    for (int i = 0; i < gpuDeviceCount; i++) {
        char* dName = (char*) malloc(sizeof(char) * 100);
        gpu->GetDeviceName(i, dName, 100);
        rList->list[i + 1].name = dName;
        rList->list[i + 1].flags = SINGLE | ASYNCH | GPU;
    }   
    delete gpu;
#else
    rList->list = (Resource*) malloc(sizeof(Resource) * rList->length); 
#endif
    
    rList->list[0].name = "CPU";
    rList->list[0].flags = ASYNCH | CPU;
    if (sizeof(REAL) == 4)
        rList->list[0].flags |= SINGLE;
    else
        rList->list[0].flags |= DOUBLE;
    
    return rList;
}

int createInstance(int tipCount,
                   int partialsBufferCount,
                   int compactBufferCount,
                   int stateCount,
                   int patternCount,
                   int eigenBufferCount,
                   int matrixBufferCount,
                   int categoryCount,
                   int scaleBufferCount,
                   int* resourceList,
                   int resourceCount,
                   long preferenceFlags,
                   long requirementFlags) {
    try {
        // Set-up a list of implementation factories in trial-order
        if (implFactory.size() == 0) {
#if defined(CUDA) || defined(OPENCL)
            implFactory.push_back(new beagle::gpu::BeagleGPUImplFactory());
#endif
            implFactory.push_back(new beagle::cpu::BeagleCPU4StateImplFactory());
            implFactory.push_back(new beagle::cpu::BeagleCPUImplFactory());
        }

        // Try each implementation
        for(std::list<beagle::BeagleImplFactory*>::iterator factory = implFactory.begin();
            factory != implFactory.end(); factory++) {
            fprintf(stderr, "BEAGLE bootstrap: %s - ", (*factory)->getName());

            beagle::BeagleImpl* beagle = (*factory)->createImpl(tipCount, partialsBufferCount,
                                                        compactBufferCount, stateCount,
                                                        patternCount, eigenBufferCount,
                                                        matrixBufferCount, categoryCount, 
                                                        scaleBufferCount);

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
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
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
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
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
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
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

int getPartials(int instance, int bufferIndex, int scaleIndex, double* outPartials) {
    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return UNINITIALIZED_INSTANCE_ERROR;
        return beagleInstance->getPartials(bufferIndex, scaleIndex, outPartials);
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
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
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
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
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

int setCategoryRates(int instance,
                     const double* inCategoryRates) {
    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return UNINITIALIZED_INSTANCE_ERROR;
        
        return beagleInstance->setCategoryRates(inCategoryRates);
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
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
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
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
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
                   int cumulativeScalingIndex) {
    try {
        int error_code = NO_ERROR;
        for (int i = 0; i < instanceCount; i++) {
            beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instanceList[i]);
            if (beagleInstance == NULL)
                return UNINITIALIZED_INSTANCE_ERROR;

            int err = beagleInstance->updatePartials(operations, operationCount, cumulativeScalingIndex);
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
            beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instanceList[i]);
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

int accumulateScaleFactors(int instance,
						   const int* scalingIndices,
						   int count,
						   int cumulativeScalingIndex) {
    try {        
    	 beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
    	 if (beagleInstance == NULL)
    		 return UNINITIALIZED_INSTANCE_ERROR;
    	 return beagleInstance->accumulateScaleFactors(scalingIndices, count, cumulativeScalingIndex);
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

int subtractScaleFactors(int instance,
						   const int* scalingIndices,
						   int count,
						   int cumulativeScalingIndex) {
    try {        
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return UNINITIALIZED_INSTANCE_ERROR;
        return beagleInstance->subtractScaleFactors(scalingIndices, count, cumulativeScalingIndex);
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
                                const int* scalingFactorsIndices,
//                                int* scalingFactorsCount,
                                int count,
                                double* outLogLikelihoods) {
    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return UNINITIALIZED_INSTANCE_ERROR;

        return beagleInstance->calculateRootLogLikelihoods(bufferIndices, weights, stateFrequencies,
                                                           scalingFactorsIndices,
//                                                           scalingFactorsCount, 
                                                           count,
                                                           outLogLikelihoods);
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
                                const int* scalingFactorsIndices,
//                                int* scalingFactorsCount,
                                int count,
                                double* outLogLikelihoods,
                                double* outFirstDerivatives,
                                double* outSecondDerivatives) {
    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return UNINITIALIZED_INSTANCE_ERROR;

        return beagleInstance->calculateEdgeLogLikelihoods(parentBufferIndices, childBufferIndices,
                                                           probabilityIndices,
                                                           firstDerivativeIndices,
                                                           secondDerivativeIndices, weights,
                                                           stateFrequencies, scalingFactorsIndices,
//                                                           scalingFactorsCount, 
                                                           count,
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

