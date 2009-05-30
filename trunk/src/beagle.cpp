/*
 *  beagle.cpp
 *  BEAGLE
 *
 * @author Andrew Rambaut
 * @author Marc Suchard
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <list>
#include <vector>

#include "beagle.h"
#include "BeagleImpl.h"

#ifdef CUDA
	#include "CUDA/BeagleCUDAImpl.h"
#endif
#include "CPU/BeagleCPUImpl.h"

BeagleImpl **instances = NULL;
int instanceCount = 0;

std::list<BeagleImplFactory*> implFactory;

ResourceList* getResourceList() {
	return NULL;
}

int createInstance(
				int bufferCount,
				int tipCount,
				int stateCount,
				int patternCount,
				int eigenDecompositionCount,
				int matrixCount,
				int* resourceList,
				int resourceCount,
				int preferenceFlags,
				int requirementFlags )
{
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
    	fprintf(stderr,"BEAGLE bootstrap: %s - ",(*factory)->getName());

    	BeagleImpl* beagle = (*factory)->createImpl(
    		bufferCount,
    		tipCount,
    		stateCount,
    		patternCount,
    		eigenDecompositionCount,
    		matrixCount);

    	if (beagle != NULL) {
    		fprintf(stderr,"Success\n");
    		int instance = instanceCount;
    	    instanceCount++;
    	    instances = (BeagleImpl **)realloc(instances, sizeof(BeagleImpl *) * instanceCount);
    	    instances[instance] = beagle;
    	    return instance;
    	}
    	fprintf(stderr,"Failed\n");
    }

    // No implementations found or appropriate
    return GENERAL_ERROR;
}

void initializeInstance(
						int *instance,
						int instanceCount,
						InstanceDetails* returnInfo)
{
	// TODO: Actual creation of instances should wait until here
}

void finalize(int instance)
{
    delete instances[instance];
   	instances[instance] = 0L;
}

int setPartials(
                    int instance,
					int bufferIndex,
					const double* inPartials)
{
	return instances[instance]->setPartials(bufferIndex, inPartials);
}

int getPartials(int instance, int bufferIndex, double *outPartials)
{
	return instances[instance]->getPartials(bufferIndex, outPartials);
}

int setTipStates(
                  int instance,
				  int tipIndex,
				  const int* inStates)
{
    return instances[instance]->setTipStates(tipIndex, inStates);
}

int setEigenDecomposition(
                           int instance,
						   int eigenIndex,
						   const double** inEigenVectors,
						   const double** inInverseEigenVectors,
						   const double* inEigenValues)
{
 	return instances[instance]->setEigenDecomposition(eigenIndex, inEigenVectors, inInverseEigenVectors, inEigenValues);
}

int setTransitionMatrix(	int instance,
                			int matrixIndex,
                			const double* inMatrix)
{
    return instances[instance]->setTransitionMatrix(matrixIndex, inMatrix);
}

int updateTransitionMatrices(
                              int instance,
                              int eigenIndex,
                              const int* probabilityIndices,
                              const int* firstDerivativeIndices,
                              const int* secondDervativeIndices,
                              const double* edgeLengths,
                              int count)
{
 	return instances[instance]->updateTransitionMatrices(eigenIndex, probabilityIndices, firstDerivativeIndices, secondDervativeIndices, edgeLengths, count);
}

int updatePartials(
                       int* instanceList,
                       int instanceCount,
					   int* operations,
					   int operationCount,
					   int rescale)
{
    int error_code = NO_ERROR;
 	for (int i = 0; i < instanceCount; i++) {
  		int err = instances[instanceList[i]]->updatePartials(operations, operationCount, rescale);
 	    if (err != NO_ERROR) {
 	        error_code = err;
        }
    }
	return error_code;
}

int calculateRootLogLikelihoods(
                             int instance,
		                     const int* bufferIndices,
		                     const double* weights,
		                     const double** stateFrequencies,
		                     int count,
			                 double* outLogLikelihoods)
{
    return instances[instance]->calculateRootLogLikelihoods(
                                            bufferIndices,
                                            weights,
                                            stateFrequencies,
                                            count,
                                            outLogLikelihoods);
}

int calculateEdgeLogLikelihoods(
							 int instance,
		                     const int* parentBufferIndices,
		                     const int* childBufferIndices,
		                     const int* probabilityIndices,
		                     const int* firstDerivativeIndices,
		                     const int* secondDerivativeIndices,
		                     const double* weights,
		                     const double** stateFrequencies,
		                     int count,
		                     double* outLogLikelihoods,
			                 double* outFirstDerivatives,
			                 double* outSecondDerivatives)
{
    return instances[instance]->calculateEdgeLogLikelihoods(
   											parentBufferIndices,
   											childBufferIndices,
   											probabilityIndices,
   											firstDerivativeIndices,
   											secondDerivativeIndices,
											weights,
											stateFrequencies,
											count,
											outLogLikelihoods,
											outFirstDerivatives,
											outSecondDerivatives);
}

