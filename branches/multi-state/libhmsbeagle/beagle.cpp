/*
 *  beagle.cpp
 *  BEAGLE
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * BEAGLE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * BEAGLE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAGLE.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @author Andrew Rambaut
 * @author Marc Suchard
 * @author Daniel Ayres
 * @author Aaron Darling
 * 
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
std::vector<beagle::BeagleImpl*> *instances = NULL;

/// returns an initialized instance or NULL if the index refers to an invalid instance
namespace beagle {
BeagleImpl* getBeagleInstance(int instanceIndex);


BeagleImpl* getBeagleInstance(int instanceIndex) {
    if (instanceIndex > instances->size())
        return NULL;
    return (*instances)[instanceIndex];
}

}	// end namespace beagle

std::list<beagle::BeagleImplFactory*>* implFactory = NULL;

BeagleResourceList* rsrcList = NULL;

int loaded = 0; // Indicates is the initial library constructors have been run
                // This patches a bug with JVM under Linux that calls the finalizer twice

std::list<beagle::BeagleImplFactory*>* beagleGetFactoryList(void) {	
	if (implFactory == NULL) {
		
		implFactory = new std::list<beagle::BeagleImplFactory*>;
	
		// Set-up a list of implementation factories in trial-order
#if defined(CUDA) || defined(OPENCL)
		if (rsrcList->length > 1)
			implFactory->push_back(new beagle::gpu::BeagleGPUImplFactory());
#endif
		implFactory->push_back(new beagle::cpu::BeagleCPU4StateImplFactory());
		implFactory->push_back(new beagle::cpu::BeagleCPUImplFactory());		
	}
	return implFactory;
}

void beagle_library_initialize(void) {
//	beagleGetResourceList(); // Generate resource list at library initialization, causes Bus error on Mac
//	beagleGetFactoryList(); // Generate factory list at library initialization, causes Bus error on Mac	
}

void beagle_library_finalize(void) {
	
	// Destroy implFactory
	if (implFactory && loaded) {	
		try {
		for(std::list<beagle::BeagleImplFactory*>::iterator factory = implFactory->begin();
				factory != implFactory->end(); factory++) {			
			delete (*factory); // Fixes 4 byte leak for each entry in implFactory    	
		}	
		delete implFactory;
		} catch (...) {
			
		}
	}
	
	// Destroy rsrcList
	if (rsrcList && loaded) {
		for(int i=1; i<rsrcList->length; i++) { // #0 is not needed		
			free(rsrcList->list[i].name);
			free(rsrcList->list[i].description);	
		}
		free(rsrcList->list);
		free(rsrcList);
	}	
	
	// Destroy instances
	if (instances && loaded) {
		delete instances;
	}
	loaded = 0;
}

#ifdef __GNUC__
void __attribute__ ((constructor)) beagle_gnu_init(void) {
	beagle_library_initialize();
}
void __attribute__ ((destructor)) beagle_gnu_finalize(void) {
	beagle_library_finalize();
}
#endif

BeagleResourceList* beagleGetResourceList() {
    
    if (rsrcList == NULL) {
        rsrcList = (BeagleResourceList*) malloc(sizeof(BeagleResourceList));
        rsrcList->length = 1;
        
#if defined(CUDA) || defined(OPENCL)
        GPUInterface* gpu = new GPUInterface;     
        if (gpu->Initialize()) {        
            int gpuDeviceCount = gpu->GetDeviceCount();
            rsrcList->length += gpuDeviceCount;
            rsrcList->list = (BeagleResource*) malloc(sizeof(BeagleResource) * rsrcList->length); 
            for (int i = 0; i < gpuDeviceCount; i++) {
                char* dName = (char*) malloc(sizeof(char) * 100);
                char* dDesc = (char*) malloc(sizeof(char) * 100);
                gpu->GetDeviceName(i, dName, 100);
                gpu->GetDeviceDescription(i, dDesc);
                rsrcList->list[i + 1].name = dName;
                rsrcList->list[i + 1].description = dDesc;
                rsrcList->list[i + 1].flags = BEAGLE_FLAG_SINGLE | BEAGLE_FLAG_ASYNCH | BEAGLE_FLAG_GPU;
            }   
        } else {        	
            rsrcList->list = (BeagleResource*) malloc(sizeof(BeagleResource) * rsrcList->length);             
        }
        delete gpu;
#else
        rsrcList->list = (BeagleResource*) malloc(sizeof(BeagleResource) * rsrcList->length); 
#endif
        
        rsrcList->list[0].name = (char*) "CPU";
        rsrcList->list[0].description = (char*) "";
        rsrcList->list[0].flags = BEAGLE_FLAG_ASYNCH | BEAGLE_FLAG_CPU;
        if (sizeof(REAL) == 4)
            rsrcList->list[0].flags |= BEAGLE_FLAG_SINGLE;
        else
            rsrcList->list[0].flags |= BEAGLE_FLAG_DOUBLE;
    }
    
    return rsrcList;
}

int beagleCreateInstance(int tipCount,
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
    	if (instances == NULL)
    		instances = new std::vector<beagle::BeagleImpl*>;
    	
        if (rsrcList == NULL)
            beagleGetResourceList();
        
        if (implFactory == NULL)
        	beagleGetFactoryList();
        
        loaded = 1;
        // Try each implementation
        for(std::list<beagle::BeagleImplFactory*>::iterator factory = implFactory->begin();
            factory != implFactory->end(); factory++) {
            
            if ((*factory)->getName() == "GPU" && (!(resourceList == NULL || (resourceList[0] < rsrcList->length
                 && rsrcList->list[resourceList[0]].flags & BEAGLE_FLAG_GPU)) || preferenceFlags & BEAGLE_FLAG_CPU || requirementFlags & BEAGLE_FLAG_CPU))
                continue;

            beagle::BeagleImpl* beagle = (*factory)->createImpl(tipCount, partialsBufferCount,
                                                        compactBufferCount, stateCount,
                                                        patternCount, eigenBufferCount,
                                                        matrixBufferCount, categoryCount, 
                                                        scaleBufferCount);

            if (beagle != NULL) {
                int instance = instances->size();
                instances->push_back(beagle);
                return instance;
            }
        }

        // No implementations found or appropriate
        return BEAGLE_ERROR_GENERAL;
    }
    catch (std::bad_alloc &) {
        return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
    catch (std::out_of_range &) {
        return BEAGLE_ERROR_OUT_OF_RANGE;
    }
    catch (...) {
        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
    }
    loaded = 1;
}

int beagleInitializeInstance(int instance,
                       BeagleInstanceDetails* returnInfo) {
    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
        return beagleInstance->initializeInstance(returnInfo);
    }
    catch (std::bad_alloc &) {
        return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
    catch (std::out_of_range &) {
        return BEAGLE_ERROR_OUT_OF_RANGE;
    }
    catch (...) {
        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
    }
}

int beagleFinalizeInstance(int instance) {
    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
        delete beagleInstance;
        (*instances)[instance] = NULL;
        return BEAGLE_SUCCESS;
    }
    catch (std::bad_alloc &) {
        return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
    catch (std::out_of_range &) {
        return BEAGLE_ERROR_OUT_OF_RANGE;
    }
    catch (...) {
        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
    }
}

int beagleSetTipStates(int instance,
                 int tipIndex,
                 const int* inStates) {
    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
        
        return beagleInstance->setTipStates(tipIndex, inStates);
    }
    catch (std::bad_alloc &) {
        return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
    catch (std::out_of_range &) {
        return BEAGLE_ERROR_OUT_OF_RANGE;
    }
    catch (...) {
        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
    }
}

int beagleSetTipPartials(int instance,
                   int tipIndex,
                   const double* inPartials) {
    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
        return beagleInstance->setTipPartials(tipIndex, inPartials);
    }
    catch (std::bad_alloc &) {
        return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
    catch (std::out_of_range &) {
        return BEAGLE_ERROR_OUT_OF_RANGE;
    }
    catch (...) {
        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
    }
}

int beagleSetPartials(int instance,
                int bufferIndex,
                const double* inPartials) {
    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
        return beagleInstance->setPartials(bufferIndex, inPartials);
    }
    catch (std::bad_alloc &) {
        return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
    catch (std::out_of_range &) {
        return BEAGLE_ERROR_OUT_OF_RANGE;
    }
    catch (...) {
        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
    }
}

int beagleGetPartials(int instance, int bufferIndex, int scaleIndex, double* outPartials) {
    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
        return beagleInstance->getPartials(bufferIndex, scaleIndex, outPartials);
    }
    catch (std::bad_alloc &) {
        return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
    catch (std::out_of_range &) {
        return BEAGLE_ERROR_OUT_OF_RANGE;
    }
    catch (...) {
        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
    }
}

int beagleSetEigenDecomposition(int instance,
                          int eigenIndex,
                          const double* inEigenVectors,
                          const double* inInverseEigenVectors,
                          const double* inEigenValues) {
    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;

        return beagleInstance->setEigenDecomposition(eigenIndex, inEigenVectors,
                                                     inInverseEigenVectors, inEigenValues);
    }
    catch (std::bad_alloc &) {
        return BEAGLE_ERROR_OUT_OF_MEMORY;
    }
    catch (std::out_of_range &) {
        return BEAGLE_ERROR_OUT_OF_RANGE;
    }
    catch (...) {
        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
    }
}

int beagleSetCategoryRates(int instance,
                     const double* inCategoryRates) {
//    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
        
        return beagleInstance->setCategoryRates(inCategoryRates);
//    }
//    catch (std::bad_alloc &) {
//        return BEAGLE_ERROR_OUT_OF_MEMORY;
//    }
//    catch (std::out_of_range &) {
//        return BEAGLE_ERROR_OUT_OF_RANGE;
//    }
//    catch (...) {
//        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
//    }
}

int beagleSetTransitionMatrix(int instance,
                        int matrixIndex,
                        const double* inMatrix) {
//    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;

        return beagleInstance->setTransitionMatrix(matrixIndex, inMatrix);
//    }
//    catch (std::bad_alloc &) {
//        return BEAGLE_ERROR_OUT_OF_MEMORY;
//    }
//    catch (std::out_of_range &) {
//        return BEAGLE_ERROR_OUT_OF_RANGE;
//    }
//    catch (...) {
//        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
//    }
}

int beagleUpdateTransitionMatrices(int instance,
                             int eigenIndex,
                             const int* probabilityIndices,
                             const int* firstDerivativeIndices,
                             const int* secondDervativeIndices,
                             const double* edgeLengths,
                             int count) {
//    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
        return beagleInstance->updateTransitionMatrices(eigenIndex, probabilityIndices,
                                                        firstDerivativeIndices,
                                                        secondDervativeIndices, edgeLengths, count);
//    }
//    catch (std::bad_alloc &) {
//        return BEAGLE_ERROR_OUT_OF_MEMORY;
//    }
//    catch (std::out_of_range &) {
//        return BEAGLE_ERROR_OUT_OF_RANGE;
//    }
//    catch (...) {
//        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
//    }
}

int beagleUpdatePartials(const int* instanceList,
                   int instanceCount,
                   const int* operations,
                   int operationCount,
                   int cumulativeScalingIndex) {
//    try {
        int error_code = BEAGLE_SUCCESS;
        for (int i = 0; i < instanceCount; i++) {
            beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instanceList[i]);
            if (beagleInstance == NULL)
                return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;

            int err = beagleInstance->updatePartials(operations, operationCount, cumulativeScalingIndex);
            if (err != BEAGLE_SUCCESS) {
                error_code = err;
            }
        }
        return error_code;
//    }
//    catch (std::bad_alloc &) {
//        return BEAGLE_ERROR_OUT_OF_MEMORY;
//    }
//    catch (std::out_of_range &) {
//        return BEAGLE_ERROR_OUT_OF_RANGE;
//    }
//    catch (...) {
//        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
//    }
}

int beagleWaitForPartials(const int* instanceList,
                    int instanceCount,
                    const int* destinationPartials,
                    int destinationPartialsCount) {
//    try {
        int error_code = BEAGLE_SUCCESS;
        for (int i = 0; i < instanceCount; i++) {
            beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instanceList[i]);
            if (beagleInstance == NULL)
                return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
            
            int err = beagleInstance->waitForPartials(destinationPartials,
                                                      destinationPartialsCount);
            if (err != BEAGLE_SUCCESS) {
                error_code = err;
            }
        }
        return error_code;
//    }
//    catch (std::bad_alloc &) {
//        return BEAGLE_ERROR_OUT_OF_MEMORY;
//    }
//    catch (std::out_of_range &) {
//        return BEAGLE_ERROR_OUT_OF_RANGE;
//    }
//    catch (...) {
//        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
//    }
}

int beagleAccumulateScaleFactors(int instance,
						   const int* scalingIndices,
						   int count,
						   int cumulativeScalingIndex) {
//    try {
    	 beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
    	 if (beagleInstance == NULL)
    		 return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
    	 return beagleInstance->accumulateScaleFactors(scalingIndices, count, cumulativeScalingIndex);
//    }
//    catch (std::bad_alloc &) {
//        return BEAGLE_ERROR_OUT_OF_MEMORY;
//    }
//    catch (std::out_of_range &) {
//        return BEAGLE_ERROR_OUT_OF_RANGE;
//    }
//    catch (...) {
//        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
//    }
}

int beagleRemoveScaleFactors(int instance,
						   const int* scalingIndices,
						   int count,
						   int cumulativeScalingIndex) {
//    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
        return beagleInstance->removeScaleFactors(scalingIndices, count, cumulativeScalingIndex);
//    }
//    catch (std::bad_alloc &) {
//        return BEAGLE_ERROR_OUT_OF_MEMORY;
//    }
//    catch (std::out_of_range &) {
//        return BEAGLE_ERROR_OUT_OF_RANGE;
//    }
//    catch (...) {
//        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
//    }
}

int beagleResetScaleFactors(int instance,
                      int cumulativeScalingIndex) {
//    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
        return beagleInstance->resetScaleFactors(cumulativeScalingIndex);
//    }
//    catch (std::bad_alloc &) {
//        return BEAGLE_ERROR_OUT_OF_MEMORY;
//    }
//    catch (std::out_of_range &) {
//        return BEAGLE_ERROR_OUT_OF_RANGE;
//    }
//    catch (...) {
//        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
//    }
}

int beagleCalculateRootLogLikelihoods(int instance,
                                const int* bufferIndices,
                                const double* weights,
                                const double* stateFrequencies,
                                const int* scalingFactorsIndices,
                                int count,
                                double* outLogLikelihoods) {
//    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;

        return beagleInstance->calculateRootLogLikelihoods(bufferIndices, weights, stateFrequencies,
                                                           scalingFactorsIndices,
                                                           count,
                                                           outLogLikelihoods);
//    }
//    catch (std::bad_alloc &) {
//        return BEAGLE_ERROR_OUT_OF_MEMORY;
//    }
//    catch (std::out_of_range &) {
//        return BEAGLE_ERROR_OUT_OF_RANGE;
//    }
//    catch (...) {
//        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
//    }


}

int beagleCalculateEdgeLogLikelihoods(int instance,
                                const int* parentBufferIndices,
                                const int* childBufferIndices,
                                const int* probabilityIndices,
                                const int* firstDerivativeIndices,
                                const int* secondDerivativeIndices,
                                const double* weights,
                                const double* stateFrequencies,
                                const int* scalingFactorsIndices,
                                int count,
                                double* outLogLikelihoods,
                                double* outFirstDerivatives,
                                double* outSecondDerivatives) {
//    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;

        return beagleInstance->calculateEdgeLogLikelihoods(parentBufferIndices, childBufferIndices,
                                                           probabilityIndices,
                                                           firstDerivativeIndices,
                                                           secondDerivativeIndices, weights,
                                                           stateFrequencies, scalingFactorsIndices,
                                                           count,
                                                           outLogLikelihoods, outFirstDerivatives,
                                                           outSecondDerivatives);
//    }
//    catch (std::bad_alloc &) {
//        return BEAGLE_ERROR_OUT_OF_MEMORY;
//    }
//    catch (std::out_of_range &) {
//        return BEAGLE_ERROR_OUT_OF_RANGE;
//    }
//    catch (...) {
//        return BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION;
//    }
}

