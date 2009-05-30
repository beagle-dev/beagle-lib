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
	return null;
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
    return ERROR;
}

void initializeInstance(
						int *instance, 
						int instanceCount,
						InstanceDetails* returnInfo)
{
	// TODO: Actual creation of instances should wait until here
}

void finalize(int *instance, int instanceCount)
{
	for (int i = 0; i < instanceCount; i++) 
	    instances[instance]->finalize();
}

int setPartials(
                    int* instance,
                    int instanceCount,
					int bufferIndex,
					const double* inPartials)
{
	for (int i = 0; i < instanceCount; i++) 
	    instances[instance]->setTipPartials(bufferIndex, inPartials);
	    
	return 0;
}

int getPartials(int* instance, int bufferIndex, double *outPartials)
{
	for (int i = 0; i < instanceCount; i++) 
	    instances[instance]->getPartials(bufferIndex, outPartials);
	    
	return 0;
}

int setTipStates(
                  int* instance,
                  int instanceCount,
				  int tipIndex,
				  const int* inStates)
{
	for (int i = 0; i < instanceCount; i++) 
  		instances[instance]->setTipStates(tipIndex, inStates);
	    
	return 0;
}

int setStateFrequencies(int* instance,
                         int instanceCount,
                         const double* inStateFrequencies)
{
 	for (int i = 0; i < instanceCount; i++) 
   		instances[instance]->setStateFrequencies(inStateFrequencies);
	return 0;
}

int setEigenDecomposition(
                           int* instance,
                           int instanceCount,
						   int eigenIndex,
						   const double** inEigenVectors,
						   const double** inInverseEigenVectors,
						   const double* inEigenValues)
{
 	for (int i = 0; i < instanceCount; i++) 
  		instances[instance]->setEigenDecomposition(eigenIndex, inEigenVectors, inInverseEigenVectors, inEigenValues);
	return 0;
}

int setTransitionMatrix(	int* instance,
                			int matrixIndex,
                			const double* inMatrix)
{
 	for (int i = 0; i < instanceCount; i++) 
  		instances[instance]->setTransitionMatrix(matrixIndex, inMatrix);
	return 0;
}

int updateTransitionMatrices(
                                            int* instance,
                                            int instanceCount,
                                            int eigenIndex,
                                            const int* probabilityIndices,
                                            const int* firstDerivativeIndices,
                                            const int* secondDervativeIndices,
                                            const double* edgeLengths,
                                            int count)
{
 	for (int i = 0; i < instanceCount; i++) 
  		instances[instance]->updateTransitionMatrices(eigenIndex, probabilityIndices, firstDerivativeIndices, secondDervativeIndices, edgeLengths, count);
	return 0;
}

int updatePartials(
                       int* instance,
                       int instanceCount,
					   int* operations,					
					   int operationCount,
					   int rescale)
{
 	for (int i = 0; i < instanceCount; i++) 
  		instances[instance]->calculatePartials(operations, operationCount, rescale);
	return 0;
}

int calculateRootLogLikelihoods(
                             int* instance,
                             int instanceCount,
		                     const int* bufferIndices,
		                     int count,
		                     const double* weights,
		                     const double** stateFrequencies,		                     
			                 double* outLogLikelihoods)
{
 	for (int i = 0; i < instanceCount; i++) 
   		instances[instance]->calculateLogLikelihoods(bufferIndices, count, weights, stateFrequencies, outLogLikelihoods);
	return 0;
}

int calculateEdgeLogLikelihoods(
							 int* instance,
							 int instanceCount,
		                     const int* parentBufferIndices,
		                     const int* childBufferIndices,		                   
		                     const int* probabilityIndices,
		                     const int* firstDerivativeIndices,
		                     const int* secondDerivativeIndices,
		                     int count,
		                     const double* weights,
		                     const double** stateFrequencies,
		                     double* outLogLikelihoods,
			                 double* outFirstDerivatives,
			                 double* outSecondDerivatives)
{
 	for (int i = 0; i < instanceCount; i++) 
   		instances[instance]->calculateEdgeLogLikelihoods(
   											parentBufferIndices,
   											childBufferIndices,
   											probabilityIndices,
   											firstDerivativeIndices,
   											secondDerivativeIndices,
											count, 
											weights, 
											stateFrequencies, 
											outLogLikelihoods,
											outFirstDerivatives,
											outSecondDerivatives);
	return 0;
}

