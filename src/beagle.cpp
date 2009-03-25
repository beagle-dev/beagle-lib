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

#include "beagle.h"
#include "BeagleImpl.h"
#include "BeagleCPUImpl.h"

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
    int instance = instanceCount;
    instanceCount++;
    instances = (BeagleImpl **)realloc(instances, sizeof(BeagleImpl *) * instanceCount);
    instances[instance] = new BeagleCPUImpl();

    instances[instance]->initialize(nodeCount, tipCount, stateCount, patternCount, categoryCount, matrixCount);

    return instance;
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

