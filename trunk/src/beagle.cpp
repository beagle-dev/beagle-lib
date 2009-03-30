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

int initialize(
				int nodeCount,
				int tipCount,
				int stateCount,
				int patternCount,
				int categoryCount,
				int matrixCount)
{
	// Set-up a list of implementations in trial-order
	std::list<BeagleImpl*> possibleBeagles;
#ifdef CUDA
	possibleBeagles.push_back(new BeagleCUDAImpl());
#endif
	possibleBeagles.push_back(new BeagleCPUImpl());

	// Try each implementation
    for(std::list<BeagleImpl*>::iterator beagle = possibleBeagles.begin();
		beagle != possibleBeagles.end(); beagle++) {
    	// Determine if appropriate
    	if ((*beagle)->initialize(nodeCount, tipCount, stateCount, patternCount,
    			           categoryCount, matrixCount)) {
    		// Add implementation to list of instances
    		int instance = instanceCount;
    		instanceCount++;
    		instances = (BeagleImpl **)realloc(instances, sizeof(BeagleImpl *) * instanceCount);
    		instances[instance] = *beagle;
    		return instance;
    	}
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

