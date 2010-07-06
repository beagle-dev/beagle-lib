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
 */

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>    // for exception, bad_exception
#include <stdexcept>    // for std exception hierarchy
#include <list>
#include <utility>
#include <vector>
#include <iostream>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/BeagleImpl.h"

#if defined(CUDA) || defined(OPENCL)
#endif
#if defined(ENABLE_SSE)
    #include "libhmsbeagle/CPU/BeagleCPU4StateSSEImpl.h"
	#include "libhmsbeagle/CPU/BeagleCPUSSEImpl.h"
#endif

#include "libhmsbeagle/plugin/Plugin.h"

typedef std::list< std::pair<int,int> > PairedList;

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

/** The list of plugins that provide implementations of likelihood calculators */
std::list<beagle::plugin::Plugin*> plugins;

void beagleLoadPlugins(void) {
	beagle::plugin::PluginManager& pm = beagle::plugin::PluginManager::instance();

	try{
		beagle::plugin::Plugin* cpuplug = pm.findPlugin("hmsbeagle-cpu");
		plugins.push_back(cpuplug);
	}catch(beagle::plugin::SharedLibraryException sle){
		// this one should always work
		std::cerr << "Unable to load CPU plugin!\n";
		std::cerr << "Please check for proper libhmsbeagle installation.\n";
	}

	try{
		beagle::plugin::Plugin* gpuplug = pm.findPlugin("hmsbeagle-gpu");
		plugins.push_back(gpuplug);
	}catch(beagle::plugin::SharedLibraryException sle){}

	try{
		beagle::plugin::Plugin* sseplug = pm.findPlugin("hmsbeagle-sse");
		plugins.push_back(sseplug);
	}catch(beagle::plugin::SharedLibraryException sle){}
}

std::list<beagle::BeagleImplFactory*>* beagleGetFactoryList(void) {
	if (implFactory == NULL) {
		implFactory = new std::list<beagle::BeagleImplFactory*>;
		// Set-up a list of implementation factories in trial-order
		std::list<beagle::plugin::Plugin*>::iterator plugin_iter = plugins.begin();
		for(; plugin_iter != plugins.end(); plugin_iter++ ){
			std::list<beagle::BeagleImplFactory*> factories = (*plugin_iter)->getBeagleFactories();
			implFactory->insert(implFactory->end(), factories.begin(), factories.end());
		}
	}
	return implFactory;
}

void beagle_library_initialize(void) {
//	beagleGetResourceList(); // Generate resource list at library initialization, causes Bus error on Mac
//	beagleGetFactoryList(); // Generate factory list at library initialization, causes Bus error on Mac
}

void beagle_library_finalize(void) {

	// FIXME: need to destroy each plugin
	// the following code segfaults
/*	std::list<beagle::plugin::Plugin*>::iterator plugin_iter = plugins.begin();
	for(; plugin_iter != plugins.end(); plugin_iter++ ){
		delete *plugin_iter;
	}
	plugins.clear();	
*/
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

#ifdef _WIN32
BOOL WINAPI DllMain(HINSTANCE hInstance, DWORD fdwReason, LPVOID lpvReserved) {
    switch (fdwReason) {
    case DLL_PROCESS_ATTACH:
		beagle_library_initialize();
        break;
    case DLL_PROCESS_DETACH:
		beagle_library_finalize();
        break;
    }
    return TRUE;
}
#endif

int beagleFinalize() {
    if (loaded)
        beagle_library_finalize();
    return BEAGLE_SUCCESS;
}

BeagleResourceList* beagleGetResourceList() {

	// plugins must be loaded before resources
	if (plugins.size()==0)
	    beagleLoadPlugins();

    if (rsrcList == NULL) {
	// count the total resources across plugins
        rsrcList = (BeagleResourceList*) malloc(sizeof(BeagleResourceList));
        rsrcList->length = 0;
	std::list<beagle::plugin::Plugin*>::iterator plugin_iter = plugins.begin();
	for(; plugin_iter != plugins.end(); plugin_iter++ ){
		rsrcList->length += (*plugin_iter)->getBeagleResources().size();
	}
	// allocate space for a complete list of resources
        rsrcList->list = (BeagleResource*) malloc(sizeof(BeagleResource) * rsrcList->length);
	// copy in resource lists from each plugin
	int rI=0;
	for(plugin_iter = plugins.begin(); plugin_iter != plugins.end(); plugin_iter++ ){
		std::list<BeagleResource> rList = (*plugin_iter)->getBeagleResources();
		std::list<BeagleResource>::iterator r_iter = rList.begin();
		for(; r_iter != rList.end(); r_iter++){
			rsrcList->list[rI++] = *r_iter;
		}
	}
     }

#if defined(ENABLE_SSE)
        rsrcList->list[0].supportFlags |= BEAGLE_FLAG_VECTOR_SSE;
#endif
    return rsrcList;
}

int scoreFlags(long flags1, long flags2) {
    int score = 0;
    int trait = 1;
    for(int bits=0; bits<32; bits++) {
        if ( (flags1 & trait) &&
             (flags2 & trait) )
            score++;
        trait <<= 1;
    }
    return -score;
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
                         long requirementFlags,
                         BeagleInstanceDetails* returnInfo) {
    try {
        if (instances == NULL)
            instances = new std::vector<beagle::BeagleImpl*>;

        if (rsrcList == NULL)
            beagleGetResourceList();
        
        if (implFactory == NULL)
            beagleGetFactoryList();
        
        loaded = 1;
        
        // First determine a list of possible resources
        PairedList* possibleResources = new PairedList;
        if (resourceList == NULL || resourceCount == 0) { // No list given
            for(int i=0; i<rsrcList->length; i++)
                possibleResources->push_back(std::make_pair(
                    scoreFlags(preferenceFlags,rsrcList->list[i].supportFlags), // Score
                    i)); // ID
        } else {
            for(int i=0; i<resourceCount; i++)
                possibleResources->push_back(std::make_pair(
                    scoreFlags(preferenceFlags,rsrcList->list[resourceList[i]].supportFlags), // Score
                    resourceList[i])); // ID
        }
        if (requirementFlags != 0) { // If requirements given do restriction
            for(PairedList::iterator it = possibleResources->begin();
                it != possibleResources->end(); ++it) {
                int resource = (*it).second;
                long resourceFlag = rsrcList->list[resource].supportFlags;
                if ( (resourceFlag & requirementFlags) < requirementFlags) {
                    possibleResources->remove(*(it--));
                }
            }
        }
        
        if (possibleResources->size() == 0) {
            delete possibleResources;
            return BEAGLE_ERROR_NO_RESOURCE;
        }
        
        beagle::BeagleImpl* bestBeagle = NULL;
        int bestScore = +1;
        possibleResources->sort(); // Attempt in rank order, lowest score wins
        
        int errorCode;
        
        // Score each resource-implementation pair given preferences
        for(PairedList::iterator it = possibleResources->begin();
            it != possibleResources->end(); ++it) {
            int resource = (*it).second;
            long resourceRequiredFlags = rsrcList->list[resource].requiredFlags;
            int resourceScore = (*it).first;
#ifdef BEAGLE_DEBUG_FLOW
            fprintf(stderr,"Possible resource: %s (%d)\n",rsrcList->list[resource].name,resourceScore);
#endif
            
            for (std::list<beagle::BeagleImplFactory*>::iterator factory =
                 implFactory->begin(); factory != implFactory->end(); factory++) {
                long factoryFlags = (*factory)->getFlags();
#ifdef BEAGLE_DEBUG_FLOW
                fprintf(stderr,"\tExamining implementation: %s\n",(*factory)->getName());
#endif
                if ( ((requirementFlags & factoryFlags) >= requirementFlags) // Meets requirementFlags
                    && ((resourceRequiredFlags & factoryFlags) >= resourceRequiredFlags) // Meets resourceFlags
                    ) {
                    int implementationScore = scoreFlags(preferenceFlags,factoryFlags);
                    int totalScore = resourceScore + implementationScore;
#ifdef BEAGLE_DEBUG_FLOW
                    fprintf(stderr,"\tPossible implementation: %s (%d)\n",
                            (*factory)->getName(),totalScore);
#endif
                    if (totalScore < bestScore) { // Looking for lowest
                        // TODO: should only initialize impl after finding the best one
                        beagle::BeagleImpl* beagle = (*factory)->createImpl(tipCount, partialsBufferCount,
                                                                            compactBufferCount, stateCount,
                                                                            patternCount, eigenBufferCount,
                                                                            matrixBufferCount, categoryCount,
                                                                            scaleBufferCount,
                                                                            resource,
                                                                            preferenceFlags,
                                                                            requirementFlags,
                                                                            &errorCode);
                        if (beagle != NULL) {
                            //                            beagle->resourceNumber = resource;
                            // Found a better implementation
                            if (bestBeagle != NULL)
                                delete bestBeagle;
                            bestBeagle = beagle;
                            bestScore = totalScore;
                        }
                    }
                }
            }
        }
        
        delete possibleResources;
        
        if (bestBeagle != NULL) {
            int instance = instances->size();
            instances->push_back(bestBeagle);

            int returnValue = bestBeagle->getInstanceDetails(returnInfo);
            if (returnValue == BEAGLE_SUCCESS) {
                returnInfo->resourceName = rsrcList->list[returnInfo->resourceNumber].name;
                // TODO: move implDescription to inside the implementation
                returnInfo->implDescription = (char*) "none";
                
                returnValue = instance;
            }
            
            return returnValue;
        }
        
        // No implementations found or appropriate, return last error code
        return errorCode;
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
    	// BeagleImpl::createInstance should be called here
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
        int returnValue = beagleInstance->getInstanceDetails(returnInfo);
        returnInfo->resourceName = rsrcList->list[returnInfo->resourceNumber].name;
        return returnValue;
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

int beagleSetStateFrequencies(int instance,
                              int stateFrequenciesIndex,
                              const double* inStateFrequencies) {
    beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
    if (beagleInstance == NULL)
        return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
    
    return beagleInstance->setStateFrequencies(stateFrequenciesIndex, inStateFrequencies);    
}

int beagleSetCategoryWeights(int instance,
                             int categoryWeightsIndex,
                             const double* inCategoryWeights) {
    beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
    if (beagleInstance == NULL)
        return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
    
    return beagleInstance->setCategoryWeights(categoryWeightsIndex, inCategoryWeights);    
}

int beagleSetPatternWeights(int instance,
                            const double* inPatternWeights) {
    beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
    if (beagleInstance == NULL)
        return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
    
    return beagleInstance->setPatternWeights(inPatternWeights);    
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
                        const double* inMatrix,
                        double paddedValue) {
//    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;

        return beagleInstance->setTransitionMatrix(matrixIndex, inMatrix, paddedValue);
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

int beagleSetTransitionMatrices(int instance,
                              const int* matrixIndices,
                              const double* inMatrices,
                              const double* paddedValues,
                              int count) {
    //    try {
    beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
    if (beagleInstance == NULL)
        return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
    
    return beagleInstance->setTransitionMatrices(matrixIndices, inMatrices, paddedValues, count);
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

int beagleGetTransitionMatrix(int instance,
							  int matrixIndex,
							  double* outMatrix) {
	beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
	if (beagleInstance == NULL)
		return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
	return beagleInstance->getTransitionMatrix(matrixIndex,outMatrix);
}

int beagleUpdateTransitionMatrices(int instance,
                             int eigenIndex,
                             const int* probabilityIndices,
                             const int* firstDerivativeIndices,
                             const int* secondDerivativeIndices,
                             const double* edgeLengths,
                             int count) {
//    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
        return beagleInstance->updateTransitionMatrices(eigenIndex, probabilityIndices,
                                                        firstDerivativeIndices,
                                                        secondDerivativeIndices, edgeLengths, count);
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

int beagleUpdatePartials(const int instance,
                   const BeagleOperation* operations,
                   int operationCount,
                   int cumulativeScalingIndex) {
//    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;

        return beagleInstance->updatePartials((const int*)operations, operationCount, cumulativeScalingIndex);
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

int beagleWaitForPartials(const int instance,
                    const int* destinationPartials,
                    int destinationPartialsCount) {
//    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;

        return beagleInstance->waitForPartials(destinationPartials,
                                                  destinationPartialsCount);
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
                                      const int* categoryWeightsIndices,
                                      const int* stateFrequenciesIndices,
                                      const int* cumulativeScaleIndices,
                                      int count,
                                      double* outSumLogLikelihood) {
//    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;

        return beagleInstance->calculateRootLogLikelihoods(bufferIndices, categoryWeightsIndices,
                                                           stateFrequenciesIndices,
                                                           cumulativeScaleIndices,
                                                           count,
                                                           outSumLogLikelihood);
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
                                      const int* categoryWeightsIndices,
                                      const int* stateFrequenciesIndices,
                                      const int* cumulativeScaleIndices,
                                      int count,
                                      double* outSumLogLikelihood,
                                      double* outSumFirstDerivative,
                                      double* outSumSecondDerivative) {
//    try {
        beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
        if (beagleInstance == NULL)
            return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;

        return beagleInstance->calculateEdgeLogLikelihoods(parentBufferIndices, childBufferIndices,
                                                           probabilityIndices,
                                                           firstDerivativeIndices,
                                                           secondDerivativeIndices, categoryWeightsIndices,
                                                           stateFrequenciesIndices, cumulativeScaleIndices,
                                                           count,
                                                           outSumLogLikelihood, outSumFirstDerivative,
                                                           outSumSecondDerivative);
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

int beagleGetSiteLogLikelihoods(int instance,
                                double* outLogLikelihoods) {
    beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
    if (beagleInstance == NULL)
        return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
    return beagleInstance->getSiteLogLikelihoods(outLogLikelihoods);    
}

int beagleGetSiteDerivatives(int instance,
                             double* outFirstDerivatives,
                             double* outSecondDerivatives) {
    beagle::BeagleImpl* beagleInstance = beagle::getBeagleInstance(instance);
    if (beagleInstance == NULL)
        return BEAGLE_ERROR_UNINITIALIZED_INSTANCE;
    return beagleInstance->getSiteDerivatives(outFirstDerivatives, outSecondDerivatives);        
}

