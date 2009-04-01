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
//	#include "CUDA/BeagleCUDAImplFactory.h"
#endif
#include "CPU/BeagleCPUImpl.h"
//#include "CPU/BeagleCPUImplFactory.h"

BeagleImpl **instances = NULL;
int instanceCount = 0;

std::list<BeagleImplFactory*> implFactory;

int initialize(
				int nodeCount,
				int tipCount,
				int stateCount,
				int patternCount,
				int categoryCount,
				int matrixCount)
{
	// Set-up a list of implementation factories in trial-order
	if (implFactory.size() == 0) {
#ifdef CUDA
		implFactory.push_back(new BeagleCUDAImplFactory());
#endif
		implFactory.push_back(new BeagleCPUImplFactory());
	}
fprintf(stderr,"starting up...\n");

	// Try each implementation
    for(std::list<BeagleImplFactory*>::iterator factory = implFactory.begin();
		factory != implFactory.end(); factory++) {
    	fprintf(stderr,"BEAGLE bootstrap: %s - ",(*factory)->getName());
    	BeagleImpl* beagle = (*factory)->createImpl(nodeCount,tipCount,stateCount,patternCount,categoryCount,matrixCount);
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

void finalize(int instance)
{
    instances[instance]->finalize();
}

void setTipPartials(
                    int instance,
					int tipIndex,
					double* inPartials)
{
    instances[instance]->setTipPartials(tipIndex, inPartials);
}

void setTipStates(
                  int instance,
				  int tipIndex,
				  int* inStates)
{
    instances[instance]->setTipStates(tipIndex, inStates);
}

void setStateFrequencies(int instance, double* inStateFrequencies)
{
    instances[instance]->setStateFrequencies(inStateFrequencies);
}

void setEigenDecomposition(
                           int instance,
						   int matrixIndex,
						   double** inEigenVectors,
						   double** inInverseEigenVectors,
						   double* inEigenValues)
{
    instances[instance]->setEigenDecomposition(matrixIndex, inEigenVectors, inInverseEigenVectors, inEigenValues);
}

void setCategoryRates(int instance, double* inCategoryRates)
{
    instances[instance]->setCategoryRates(inCategoryRates);
}

void setCategoryProportions(int instance, double* inCategoryProportions)
{
    instances[instance]->setCategoryProportions(inCategoryProportions);
}

void calculateProbabilityTransitionMatrices(
                                            int instance,
                                            int* nodeIndices,
                                            double* branchLengths,
                                            int count)
{
    instances[instance]->calculateProbabilityTransitionMatrices(nodeIndices, branchLengths, count);
}

void calculatePartials(
                       int instance,
					   int* operations,
					   int* dependencies,
					   int count,
					   int rescale)
{
    instances[instance]->calculatePartials(operations, dependencies, count, rescale);
}

void calculateLogLikelihoods(
                             int instance,
							 int rootNodeIndex,
							 double* outLogLikelihoods)
{
    instances[instance]->calculateLogLikelihoods(rootNodeIndex, outLogLikelihoods);
}

void storeState(int instance)
{
    instances[instance]->storeState();
}

void restoreState(int instance)
{
    instances[instance]->restoreState();
}

