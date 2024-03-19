/*
 *  BeagleCPUActionImpl.hpp
 *  BEAGLE
 *
 * Copyright 2022 Phylogenetic Likelihood Working Group
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
 * @author Xiang Ji
 * @author Marc Suchard
 */

#ifndef BEAGLE_BEAGLECPUACTIONIMPL_HPP
#define BEAGLE_BEAGLECPUACTIONIMPL_HPP

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPUImpl.h"
#include "libhmsbeagle/CPU/BeagleCPUActionImpl.h"

using std::vector;
using std::tuple;

template <typename T>
double normP1(const T& matrix) {
    return (Eigen::RowVectorXd::Ones(matrix.rows()) * matrix.cwiseAbs()).maxCoeff();
}

template <typename T>
tuple<double,int> ArgNormP1(const T& matrix)
{
    int x=-1;
    double v = matrix.colwise().template lpNorm<1>().maxCoeff(&x);
    return {v,x};
}

template <typename T>
double normPInf(const T& matrix) {
    return matrix.template lpNorm<Eigen::Infinity>();
}

bool parallel_to_some(const MatrixXd& S1, const MatrixXd& S2, int i)
{
    // Check if column i of S1 is parallel to some column of S2.
    assert(S1.rows() == S2.rows());
    assert(S1.cols() == S2.cols());

    int n = S1.rows();

    for(int j=0;j<S2.cols();j++)
    {
	double x = S1.col(i).adjoint() * S2.col(j);
	if (x == n)
	    return true;
    }
    return false;
}

double all_parallel_to_some(const MatrixXd& S1, const MatrixXd& S2)
{
    assert(S1.rows() == S2.rows());
    assert(S1.cols() == S2.cols());

    for(int i=0;i<S1.cols();i++)
	if (not parallel_to_some(S1,S2,i))
	    return false;

    return true;
}

bool all(const vector<bool>& v, const vector<int>& idx)
{
    for(int i: idx)
	if (not v[i])
	    return false;

    return true;
}

bool all(const vector<bool>& v, const vector<int>& idx, int n)
{
    for(int i=0;i<n;i++)
	if (not v[idx[i]])
	    return false;

    return true;
}


// https://github.com/gnu-octave/octave/blob/default/scripts/linear-algebra/normest1.m
double normest1(const SpMatrix& A, int p, int t=2)
{
    assert(A.rows() == A.cols());
    assert(p >= 1);

    if (p == 0) return 1.0;


    // A is (n,n);
    int n = A.cols();
    t = std::min(n,t);
    if (p == 1 and (n <= 4 or t == n))
	return normP1(A);

    // 2. Create the initial vectors X to act on.
    //    The first one is all 1, and the rest have blocks of -1.
    //    Officially we should use random numbers to determine the sign for columns after the first.
    MatrixXd X(n,t);
    if (t >= n or t == -1)
    {
	X = MatrixXd::Zero(n, t);
	for(int i=0;i<n;i++)
	    X(i,i) = 1.0;
	MatrixXd Y = A*X;
	for(int i=1;i<p;i++)
	    Y = A*Y;
	auto [norm,j] = ArgNormP1(Y);
	std::cerr<<"norm = "<<norm<<"\n";
	return norm;
    }
    else
    {
	X = MatrixXd::Ones(n, t);
	int start = 0;
	for(int i=1;i<t;i++)
	{
	    int end   = i*n/t+1;
	    for(int j=start;j<end;j++)
		X(j,i) *= -1.0;

	    start = end;
	}
    }
    // The columns should have a 1-norm of 1.
    X /= n;

    // 3.
    int itmax = 5;
    std::vector<bool> idx_hist(n,0);
    std::vector<int> idx(n,0);
    int idx_best = 0;
    double nest_old = 0;
    double nestestold = 0;
    MatrixXd S = MatrixXd::Ones(n,t); // The paper and algorithm have (n,t)?
    MatrixXd Sold = MatrixXd::Ones(n,t);
    int iter[2] = {0,0};
    bool converged = false;
    MatrixXd Y(n,t);
    while(not converged and iter[0] < itmax)
    {
	iter[0]++;
	Y = A*X; // Y is (r,n) * (N,t) = (r,t)
	for(int i=1;i<p;i++)
	    Y = A*Y;
	iter[1]++;
	auto [nest,j] = ArgNormP1(Y);

	if (nest > nest_old or iter[0] == 2)
	{
	    idx_best = idx[j];
	    auto w = Y.col(j); // there is an error in Algorithm 2.4
	}

        // (1) of Algorithm 2.4
	if (nest <= nest_old and iter[0] >= 2)
	    return nest_old;

	nest_old = nest;
	Sold = S;

	// S = sign(Y), 0.0 -> 1.0
	for(int i=0;i<n;i++)
	    for(int j=0;j<t;j++)
		S(i,j) = (Y(i,j) >= 0)? 1.0: -1.0;
	bool possible_break = false;

	// (2)
	if (all_parallel_to_some(S, Sold))
	{
	    possible_break = true;
	    converged = true;
	}
	else if (t > 1)
	{
	    // Ensure that no column of S is parallel to another column of S or to a column of Sold by replacing columns of Sby rand{−1,1}.
	}

	if (possible_break) continue;

	// (3)
	auto Z = A.transpose() * S;
	iter[1]++;
	Eigen::VectorXd h = Z.rowwise().lpNorm<Eigen::Infinity>();
	for(int i=0;i<n;i++)
	    idx[i] = i;

	// (4) of Algorithm 2.4
	if (iter[0] >=2 and h.maxCoeff() == h[idx_best])
	    break;

	std::sort(idx.begin(), idx.end(), [&](int i,int j) {return h[i] > h[j];});

	// reorder idx correspondingly
	if (t > 1)
	{
	    if (all(idx_hist, idx, t))
		break;

	    // checking if we've seen idx[i], saving it to idx[k] if not.
	    int k=0;
	    for(int i=0;i<idx.size();i++)
	    {
		if (not idx_hist[idx[i]])
		{
		    idx[k] = idx[i];
		    k++;
		}
	    }
	    idx.resize( std::min(k,t) );

	    // if idx(1:t) is contained in idx_hist, break
	    // replace idx(1:t) by the first t idxices in idx(1:n) that are not in idx_hist
	}
	int tmax = std::min<int>(t, idx.size());

	X = MatrixXd::Zero(n, tmax);
	for(int j=0; j < tmax; j++)
	    X(idx[j], j) = 1; // X(:,j) = e(idx[j])

	for(int i: idx)
	    idx_hist[i] = true;
    }

    // v = e[idx_best]

    return nest_old;
}


class SimpleAction;
namespace beagle {
    namespace cpu {

        BEAGLE_CPU_ACTION_TEMPLATE
	MapType BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::partialsMap(int index, int category, int startPattern, int endPattern)
	{
	    double* start = gPartials[index] + category*kPaddedPatternCount*kStateCount;
	    start += startPattern*kStateCount;
	    return MapType(start, kStateCount, endPattern - startPattern);
	}

        BEAGLE_CPU_ACTION_TEMPLATE
	MapType BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::partialsMap(int index, int category)
	{
	    return partialsMap(index, category, 0, kPatternCount);
	}

        BEAGLE_CPU_ACTION_TEMPLATE
	MapType BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::partialsCacheMap(int index, int category, int startPattern, int endPattern)
	{
	    double* start = gPartials[index + kPartialsCacheOffset] + category*kPaddedPatternCount*kStateCount;
	    start += startPattern*kStateCount;
	    return MapType(start, kStateCount, endPattern - startPattern);
	}

        BEAGLE_CPU_ACTION_TEMPLATE
	MapType BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::partialsCacheMap(int index, int category)
	{
	    return partialsCacheMap(index, category, 0, kPatternCount);
	}

        BEAGLE_CPU_ACTION_TEMPLATE
        int BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::createInstance(int tipCount,
                                                                    int partialsBufferCount,
                                                                    int compactBufferCount,
                                                                    int stateCount,
                                                                    int patternCount,
                                                                    int eigenDecompositionCount,
                                                                    int matrixCount,
                                                                    int categoryCount,
                                                                    int scaleBufferCount,
                                                                    int resourceNumber,
                                                                    int pluginResourceNumber,
								    long long preferenceFlags,
                                                                    long long requirementFlags) {
            int parentCode = BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::createInstance(tipCount, 2 * partialsBufferCount, compactBufferCount,
                                                                               stateCount, patternCount, eigenDecompositionCount,
                                                                               matrixCount, categoryCount, scaleBufferCount,
                                                                               resourceNumber, pluginResourceNumber,
                                                                               preferenceFlags, requirementFlags);
            kPartialsCacheOffset = partialsBufferCount + compactBufferCount;

            gInstantaneousMatrices.resize(eigenDecompositionCount);
            gBs.resize(eigenDecompositionCount);
            gMuBs.resize(eigenDecompositionCount);
            gB1Norms.resize(eigenDecompositionCount);
            powerMatrices.resize(eigenDecompositionCount);
            ds.resize(eigenDecompositionCount);

            gEigenMaps.resize(kBufferCount);
            gEdgeMultipliers.resize(kBufferCount * categoryCount);
//            gSimpleActions = (SimpleAction**) malloc(sizeof(SimpleAction *) * eigenDecompositionCount);
//            for (int eigen = 0; eigen < eigenDecompositionCount; eigen++) {
//                gSimpleActions[eigen] = (SimpleAction *) malloc(sizeof(SimpleAction));
//                SimpleAction* action = new SimpleAction();
//                action->createInstance(categoryCount, patternCount, stateCount, &gInstantaneousMatrices[eigen],
//                                       gEdgeMultipliers);
////                SimpleAction action(categoryCount, patternCount, stateCount);
//                gSimpleActions[eigen] = action;
//            }
//            gScaledQs = new SpMatrix * [kBufferCount];
            identity = SpMatrix(kStateCount, kStateCount);
            identity.setIdentity();

            gIntegrationTmp = (double *) malloc(sizeof(double) * kStateCount * kPaddedPatternCount * kCategoryCount);

            gMappedIntegrationTmp = (MapType*) malloc(sizeof(MapType) * kCategoryCount);
            for (int category = 0; category < kCategoryCount; category++) {
                new (& gMappedIntegrationTmp[category]) MapType(gIntegrationTmp + category * kPaddedPatternCount * kStateCount, kStateCount, kPatternCount);
            }

            for (int i = 0; i < eigenDecompositionCount; i++) {
                SpMatrix matrix(kStateCount, kStateCount);
                gInstantaneousMatrices[i] = matrix;
            }

//            for (int i = 0; i < kBufferCount; i++) {
//                gScaledQs[i] = NULL;
//            }

            return BEAGLE_SUCCESS;
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::~BeagleCPUActionImpl() {
            free(gIntegrationTmp);
        }

        BEAGLE_CPU_FACTORY_TEMPLATE
        inline const char* getBeagleCPUActionName(){ return "CPU-Action-Unknown"; };

        template<>
        inline const char* getBeagleCPUActionName<double>(){ return "CPU-Action-Double"; };

        template<>
        inline const char* getBeagleCPUActionName<float>(){ return "CPU-Action-Single"; };


	BEAGLE_CPU_ACTION_TEMPLATE
        int BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::upPartials(bool byPartition,
								      const int *operations,
								      int operationCount,
								      int cumulativeScaleIndex) {
	    double* cumulativeScaleBuffer = NULL;
            if (cumulativeScaleIndex != BEAGLE_OP_NONE)
                cumulativeScaleBuffer = gScaleBuffers[cumulativeScaleIndex];

            for (int op = 0; op < operationCount; op++) {

                int numOps = BEAGLE_OP_COUNT;
		if (byPartition)
		    numOps = BEAGLE_PARTITION_OP_COUNT;

		if (DEBUGGING_OUTPUT) {
		    fprintf(stderr, "op[%d] = ", op);
		    for (int j = 0; j < numOps; j++) {
			std::cerr << operations[op*numOps+j] << " ";
		    }
		    fprintf(stderr, "\n");
		}

		const int destinationPartialIndex = operations[op * numOps];
                const int writeScalingIndex = operations[op * numOps + 1];
                const int readScalingIndex = operations[op * numOps + 2];
                const int firstChildPartialIndex = operations[op * numOps + 3];
                const int firstChildSubstitutionMatrixIndex = operations[op * numOps + 4];
                const int secondChildPartialIndex = operations[op * numOps + 5];
                const int secondChildSubstitutionMatrixIndex = operations[op * numOps + 6];
		int currentPartition = 0;
		if (byPartition) {
		    currentPartition = operations[op * numOps + 7];
		    cumulativeScaleIndex = operations[op * numOps + 8];
		    if (cumulativeScaleIndex != BEAGLE_OP_NONE)
			cumulativeScaleBuffer = gScaleBuffers[cumulativeScaleIndex];
		    else
			cumulativeScaleBuffer = NULL;
		}

		int startPattern = 0;
		int endPattern = kPatternCount;
		if (byPartition) {
		    startPattern = this->gPatternPartitionsStartPatterns[currentPartition];
		    endPattern = this->gPatternPartitionsStartPatterns[currentPartition + 1];

		    assert(startPattern >= 0 and startPattern <= kPatternCount);
		    assert(endPattern >= 0 and endPattern <= kPatternCount);
		    assert(startPattern <= endPattern);
		}

                int rescale = BEAGLE_OP_NONE;
                double* scalingFactors = NULL;
                if (writeScalingIndex >= 0) {
                    rescale = 1;
                    scalingFactors = gScaleBuffers[writeScalingIndex];
                } else if (readScalingIndex >= 0) {
                    rescale = 0;
                    scalingFactors = gScaleBuffers[readScalingIndex];
                } else {
                    rescale = 0;
                }


#ifdef BEAGLE_DEBUG_FLOW
                std::cerr<<"Updating partials for index: "<<destinationPartialIndex << std::endl;
#endif

//                calcPartialsPartials(destP, partials1, matrices1, partials2, matrices2);
                calcPartialsPartials2(destinationPartialIndex,
				      firstChildPartialIndex,
				      firstChildSubstitutionMatrixIndex,
				      secondChildPartialIndex,
				      secondChildSubstitutionMatrixIndex,
				      startPattern,
				      endPattern);

                if (rescale == 1) {
		    double* destPartials = gPartials[destinationPartialIndex];
		    if (byPartition) {
			this->rescalePartialsByPartition(destPartials,scalingFactors,cumulativeScaleBuffer,0, currentPartition);
		    } else {
			this->rescalePartials(destPartials,scalingFactors,cumulativeScaleBuffer,0);
		    }
                }
            }

            return BEAGLE_SUCCESS;
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        int BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::upPrePartials(bool byPartition,
									 const int *operations,
									 int operationCount,
									 int cumulativeScaleIndex) {
            double* cumulativeScaleBuffer = NULL;
            if (cumulativeScaleIndex != BEAGLE_OP_NONE)
                cumulativeScaleBuffer = gScaleBuffers[cumulativeScaleIndex];

            for (int op = 0; op < operationCount; op++) {
                int numOps = BEAGLE_OP_COUNT;

                // create a list of partial likelihood update operations
                // the order is [dest, destScaling, source1, matrix1, source2, matrix2]
                // destPartials point to the pre-order partials
                // partials1 = pre-order partials of the parent node
                // matrices1 = Ptr matrices of the current node (to the parent node)
                // partials2 = post-order partials of the sibling node
                // matrices2 = Ptr matrices of the sibling node (to the parent node)
                const int destinationPartialIndex = operations[op * numOps];
                const int writeScalingIndex = operations[op * numOps + 1];
                const int readScalingIndex = operations[op * numOps + 2];
                const int parentIndex = operations[op * numOps + 3];
                const int substitutionMatrixIndex = operations[op * numOps + 4];
                const int siblingIndex = operations[op * numOps + 5];
                const int siblingSubstitutionMatrixIndex = operations[op * numOps + 6];
		int currentPartition = 0;
		if (byPartition) {
		    currentPartition = operations[op * numOps + 7];
		    cumulativeScaleIndex = operations[op * numOps + 8];
//                    if (cumulativeScaleIndex != BEAGLE_OP_NONE)
//                        cumulativeScaleBuffer = gScaleBuffers[cumulativeScaleIndex];
//                    else
//                        cumulativeScaleBuffer = NULL;
		}

		double *destPartials = gPartials[destinationPartialIndex];

		int startPattern = 0;
		int endPattern = kPatternCount;
		if (byPartition) {
		    startPattern = this->gPatternPartitionsStartPatterns[currentPartition];
		    endPattern = this->gPatternPartitionsStartPatterns[currentPartition + 1];

		    assert(startPattern >= 0 and startPattern <= kPatternCount);
		    assert(endPattern >= 0 and endPattern <= kPatternCount);
		    assert(startPattern <= endPattern);
		}

                int rescale = BEAGLE_OP_NONE;
                double* scalingFactors = NULL;
                if (writeScalingIndex >= 0) {
                    rescale = 1;
                    scalingFactors = gScaleBuffers[writeScalingIndex];
                } else if (readScalingIndex >= 0) {
                    rescale = 0;
                    scalingFactors = gScaleBuffers[readScalingIndex];
                } else {
                    rescale = 0;
                }


#ifdef BEAGLE_DEBUG_FLOW
                std::cerr<<"Updating preorder partials for index: "<<destinationPartialIndex << std::endl;
#endif

//                calcPrePartialsPartials(destP, partials1, matrices1, partials2, matrices2);
                calcPrePartialsPartials2(destinationPartialIndex,
					 parentIndex,
					 substitutionMatrixIndex,
					 siblingIndex,
                                         siblingSubstitutionMatrixIndex,
					 startPattern,
					 endPattern);

                if (rescale == 1) {
		    double* destPartials = gPartials[destinationPartialIndex];
		    if (byPartition) {
			this->rescalePartialsByPartition(destPartials,scalingFactors,cumulativeScaleBuffer,0, currentPartition);
		    } else {
			this->rescalePartials(destPartials,scalingFactors,cumulativeScaleBuffer,0);
		    }
                }

		if (DEBUGGING_OUTPUT) {
		    if (scalingFactors != NULL && rescale == 0) {
			for (int i = 0; i < kPatternCount; i++)
			    fprintf(stderr, "old scaleFactor[%d] = %.5f\n", i, scalingFactors[i]);
		    }
		    fprintf(stderr, "Result partials:\n");
		    for (int i = 0; i < this->kPartialsSize; i++)
			fprintf(stderr, "destP[%d] = %.5f\n", i, destPartials[i]);
		}
            }

            return BEAGLE_SUCCESS;
        }


        BEAGLE_CPU_ACTION_TEMPLATE
        int BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::setEigenDecomposition(int eigenIndex,
                                                                                 const double *inEigenVectors,
                                                                                 const double *inInverseEigenVectors,
                                                                                 const double *inEigenValues) {

            const int numNonZeros = (int) inInverseEigenVectors[0];
//            gInstantaneousMatrices[eigenIndex].setZero();
            std::vector<Triplet> tripletList;
            for (int i = 0; i < numNonZeros; i++) {
                tripletList.push_back(Triplet((int) inEigenVectors[2 * i], (int) inEigenVectors[2 * i + 1], inEigenValues[i]));
            }
            gInstantaneousMatrices[eigenIndex].setFromTriplets(tripletList.begin(), tripletList.end());
            ds[eigenIndex].clear();
            powerMatrices[eigenIndex].clear();

            double mu_B = 0.0;
            for (int i = 0; i < kStateCount; i++) {
                mu_B += gInstantaneousMatrices[eigenIndex].coeff(i, i);
            }
            mu_B /= (double) kStateCount;
            gMuBs[eigenIndex] = mu_B;
            gBs[eigenIndex] = gInstantaneousMatrices[eigenIndex] - mu_B * identity;
            gB1Norms[eigenIndex] = normP1(gBs[eigenIndex]);

//            gSimpleActions[eigenIndex]->setInstantaneousMatrix(tripletList);
//            gSimpleActions[eigenIndex]->fireMatrixChanged();
#ifdef BEAGLE_DEBUG_FLOW
            std::cerr<<"In vlaues: \n";
            for (int i = 0; i < numNonZeros; i++) {
                std::cerr<< "("<<inEigenVectors[2 * i] << ", " << inEigenVectors[2 * i + 1] << ") = "<< inEigenValues[i]<< std::endl;
            }
            std::cerr<<"Instantaneous matrix " << std::endl << gInstantaneousMatrices[eigenIndex]<<std::endl;
#endif
            return BEAGLE_SUCCESS;
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        int BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::updateTransitionMatrices(int eigenIndex,
                                                                                    const int* probabilityIndices,
                                                                                    const int* firstDerivativeIndices,
                                                                                    const int* secondDerivativeIndices,
                                                                                    const double* edgeLengths,
                                                                                    int count) {

            for (int i = 0; i < count; i++) {
                const int nodeIndex = probabilityIndices[i];
                gEigenMaps[nodeIndex] = eigenIndex;

//                if (gScaledQs[nodeIndex] == NULL) {
//                    gScaledQs[nodeIndex] = new SpMatrix[kCategoryCount];
//                    for (int i = 0; i < kCategoryCount; i++) {
//                        SpMatrix matrix(kStateCount, kStateCount);
//                        gScaledQs[nodeIndex][i] = matrix;
//                    }
//                }
                for (int category = 0; category < kCategoryCount; category++) {
                    const double categoryRate = gCategoryRates[0][category];
                    gEdgeMultipliers[nodeIndex * kCategoryCount + category] = edgeLengths[i] * categoryRate;
//                    gScaledQs[nodeIndex][category] = gInstantaneousMatrices[eigenIndex] * (edgeLengths[i] * categoryRate);
#ifdef BEAGLE_DEBUG_FLOW
                    std::cerr<<"Transition matrix, rate category " << category << " rate multiplier: " << categoryRate
                    << " edge length multiplier: " << edgeLengths[i]
                    << "  edgeMultiplier: "<< gEdgeMultipliers[nodeIndex * kCategoryCount + category]
                    << "  nodeIndex: "<< nodeIndex
                    << std::endl << gScaledQs[nodeIndex][category]<<std::endl;
#endif
                }
            }
            return BEAGLE_SUCCESS;
        }


        BEAGLE_CPU_ACTION_TEMPLATE
        void BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::calcPartialsPartials2(int destPIndex,
										  int partials1Index,
										  int edgeIndex1,
										  int partials2Index,
										  int edgeIndex2,
										  int startPattern,
										  int endPattern) {
            for (int category = 0; category < kCategoryCount; category++)
	    {
		auto partials1 = partialsMap(partials1Index, category, startPattern, endPattern);
		auto partials1Cache = partialsCacheMap(partials1Index, category, startPattern, endPattern);
		simpleAction3(partials1Cache, partials1, edgeIndex1, category, false);

		auto partials2 = partialsMap(partials2Index, category, startPattern, endPattern);
		auto partials2Cache = partialsCacheMap(partials2Index, category, startPattern, endPattern);
		simpleAction3(partials2Cache, partials2, edgeIndex2, category, false);

		auto destP = partialsMap(destPIndex, category, startPattern, endPattern);
                destP = partials1Cache.cwiseProduct(partials2Cache);
            }
        }


        BEAGLE_CPU_ACTION_TEMPLATE
        void BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::calcPrePartialsPartials2(int destPIndex,
										     int partials1Index,
                                                                                     int edgeIndex1,
										     int partials2Index,
										     int edgeIndex2,
										     int startPattern,
										     int endPattern) {
            memset(gIntegrationTmp, 0, (kPatternCount * kStateCount * kCategoryCount)*sizeof(double));

            for (int category = 0; category < kCategoryCount; category++) {
		auto partialCache2 = partialsCacheMap(partials2Index, category, startPattern, endPattern);
		auto partials1     = partialsMap(partials1Index, category, startPattern, endPattern);
		auto destP         = partialsMap(destPIndex, category, startPattern, endPattern);

                gMappedIntegrationTmp[category] = partialCache2.cwiseProduct(partials1);
		simpleAction3(destP, gMappedIntegrationTmp[category], edgeIndex1, category, true);
            }
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        void
        BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::simpleAction2(MapType& destP, MapType& partials, int edgeIndex,
                                                                     int category, bool transpose) {
#ifdef BEAGLE_DEBUG_FLOW
            std::cerr<<"New impl 2\nRate category "<<category<<std::endl;
	    std::cerr<<"In partial: \n"<<partials<<std::endl;
#endif
	    const double tol = pow(2.0, -53.0);
	    const double t = 1.0;
	    const int nCol = (int)destP.cols();

	    const double edgeMultiplier = gEdgeMultipliers[edgeIndex * kCategoryCount + category];

	    auto [m,s] = getStatistics2(t, nCol, edgeMultiplier, gEigenMaps[edgeIndex]);


#ifdef BEAGLE_DEBUG_FLOW
	    std::cerr<<"simpleAction2: m = "<<m<<"  s = "<<s <<std::endl;
#endif

	    destP = partials;
	    SpMatrix A = gBs[gEigenMaps[edgeIndex]] * edgeMultiplier;
	    if (transpose) {
		A = A.transpose();
	    }

	    MatrixXd F(kStateCount, nCol);
	    F = destP;

	    const double eta = exp(t * gMuBs[gEigenMaps[edgeIndex]] * edgeMultiplier / (double) s);

	    for (int i = 0; i < s; i++) {
		double c1 = normPInf(destP);
		for (int j = 1; j < m + 1; j++) {
		    destP = A * destP;
		    destP *= t / ((double) s * j);
		    double c2 = normPInf(destP);
		    F += destP;
		    if (c1 + c2 <= tol * normPInf(F)) {
			break;
		    }
		    c1 = c2;
		}
		F *= eta;
		destP = F;
	    }


#ifdef BEAGLE_DEBUG_FLOW
	    std::cerr<<"Out partials: \n"<<destP<<std::endl;
#endif
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        void
        BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::simpleAction3(MapType& destP, MapType& partials, int edgeIndex,
                                                                     int category, bool transpose) {
#ifdef BEAGLE_DEBUG_FLOW
            std::cerr<<"New impl 2\nRate category "<<category<<std::endl;
	    std::cerr<<"In partial: \n"<<partials<<std::endl;
#endif
	    // This is on the column-wise maximum of the L1-norm of || Exp<m,s>(Q*t)*v - Exp(Q*t)*v ||.
	    const double tol = pow(2.0, -53.0);
	    int m = 2;
	    constexpr int M = 55;

	    const double edgeMultiplier = gEdgeMultipliers[edgeIndex * kCategoryCount + category];

	    SpMatrix A = gBs[gEigenMaps[edgeIndex]] * edgeMultiplier;
	    if (transpose) {
		A = A.transpose();
	    }

	    MatrixXd v = partials;
// BEGIN
	    std::vector<MatrixXd> V(M+2);
	    V[1] = A*v; // L1
	    for(int k=2;k<=m+1;k++) // L2
		V[k] = A*V[k-1] / k; // L3
	    // L4
	    double S = ceil(pow( normP1(V[m+1])/tol, 1.0/(m+1) )); // L5
	    if (not (S >= 1))
	    {
		// Handle the case where Qt - mu*I = 0
		// Handle the case where normP1( ) is NaN.
		S = 1;
	    }
	    else
	    {
		double P = m * S; // L6
		while (m < M) { // L8
		    m = m + 1; // L9
		    V[m+1] = A*V[m] / (m+1); // L10
		    double S1 = ceil(pow( normP1(V[m+1])/tol, 1.0/(m+1) )); //L11
		    assert( S1 >= 1 );
		    double P1 = m*S1; // L12
		    if (P1 <= P) // L13
		    {
			P = P1; // L14
			S = S1; // L15
		    }
		    else
		    {
			m = m-1; // L17
			break;
		    } //L19
		} // L20
	    }
	    assert( S >= 1 );
	    assert( S <= INT_MAX );

	    int s = int(S);

#ifdef BEAGLE_DEBUG_FLOW
	    std::cerr<<"simpleAction3: m = "<<m<<"  s = "<<s <<std::endl;
#endif
	    const double eta = exp(gMuBs[gEigenMaps[edgeIndex]] * edgeMultiplier / (double) s);

	    // This loop can't be rolled into the loop above because
	    // we don't know the value of 's' until we get here.
	    MatrixXd w = partials; // L21
	    for(int k=1;k<=m;k++) { // L22
		w += V[k]/pow(s,k); // L23
	    } //L24
	    w *= eta;
	    for(int i=2;i<=s;i++) { // L26
		v = w; // L27
		for(int k=1;k<=m;k++) { // L28
		    v = A*v;
		    v /= (double(s) * k);
		    w += v; // L30
		} // L31
		w *= eta;
	    } // L32
// END
	    destP = w;

#ifdef BEAGLE_DEBUG_FLOW
	    std::cerr<<"Out partials: \n"<<destP<<std::endl;
#endif
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        std::tuple<int,int>
	BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::getStatistics2(double t, int nCol,
								      double edgeMultiplier,
								      int eigenIndex) {
	    assert( t >= 0 );
	    assert( nCol >= 0);
	    assert( edgeMultiplier >= 0 );
	    assert( eigenIndex >= 0);

            if (t * gB1Norms[eigenIndex] == 0.0)
		return {0, 1};

	    int bestM = INT_MAX;
	    double bestS = INT_MAX;  // Not all the values of s can fit in a 32-bit int.

	    const double theta = thetaConstants[mMax];
	    const double pMax = floor((0.5 + 0.5 * sqrt(5.0 + 4.0 * mMax)));
	    // pMax is the largest positive integer such that p*(p-1) <= mMax + 1

	    const bool conditionFragment313 = gB1Norms[eigenIndex] * edgeMultiplier <= 2.0 * theta / ((double) nCol * mMax) * pMax * (pMax + 3);
	    // using l = 1 as in equation 3.13
	    if (conditionFragment313) {
		for (auto& [thisM, thetaM]: thetaConstants) {
		    const double thisS = ceil(gB1Norms[eigenIndex] * edgeMultiplier / thetaM);
		    if (bestM == INT_MAX || ((double) thisM) * thisS < bestM * bestS) {
			bestS = thisS;
			bestM = thisM;
		    }
		}
	    } else {
		for (int p = 2; p < pMax; p++) {
		    for (int thisM = p * (p - 1) - 1; thisM < mMax + 1; thisM++) {
			auto it = thetaConstants.find(thisM);
			if (it != thetaConstants.end()) {
			    // equation 3.7 in Al-Mohy and Higham
			    const double dValueP = getDValue2(p, eigenIndex);
			    const double dValuePPlusOne = getDValue2(p + 1, eigenIndex);
			    const double alpha = std::max(dValueP, dValuePPlusOne) * edgeMultiplier;
			    // part of equation 3.10
			    const double thisS = ceil(alpha / thetaConstants[thisM]);
			    if (bestM == INT_MAX || ((double) thisM) * thisS < bestM * bestS) {
				bestS = thisS;
				bestM = thisM;
			    }
			}
		    }
		}
	    }
	    bestS = std::max(std::min<double>(bestS, INT_MAX), 1.0);
	    assert( bestS >= 1 );
	    assert( bestS <= INT_MAX );

	    int m = bestM;
	    int s = (int) bestS;

	    assert(m >= 0);
	    assert(s >= 1);

	    return {m,s};
        }


        BEAGLE_CPU_ACTION_TEMPLATE
        double BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::getDValue2(int p, int eigenIndex)
        {
	    assert(p >= 0);

            // equation 3.7 in Al-Mohy and Higham

            const int start = ds[eigenIndex].size();

            for (int i = start; i <= p; i++)
            {
		powerMatrices[eigenIndex].push_back({});
		ds[eigenIndex].push_back(-1);

                if (i == 0)
                {
                    powerMatrices[eigenIndex][0] = SpMatrix();
                    ds[eigenIndex][0] = 1.0;
                }
                else if (i == 1)
                {
                    powerMatrices[eigenIndex][1] = gBs[eigenIndex];
                    ds[eigenIndex][1] = normP1(powerMatrices[eigenIndex][1]);
                }
                else
                {
                    assert(p > 1);
                    powerMatrices[eigenIndex][i] = powerMatrices[eigenIndex][i - 1] * powerMatrices[eigenIndex][1];
                    ds[eigenIndex][i] = pow(normP1(powerMatrices[eigenIndex][i]), 1.0 / ((double) i));
                }
            }

            return ds[eigenIndex][p];
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        const char* BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::getName() {
            return  getBeagleCPUActionName<double>();
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        long long BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::getFlags() {
            return  BEAGLE_FLAG_COMPUTATION_SYNCH |
                    BEAGLE_FLAG_COMPUTATION_ACTION |
                    BEAGLE_FLAG_PROCESSOR_CPU |
                    BEAGLE_FLAG_PRECISION_DOUBLE |
                    BEAGLE_FLAG_VECTOR_SSE |
                    BEAGLE_FLAG_FRAMEWORK_CPU;
        }

///////////////////////////////////////////////////////////////////////////////
// BeagleImplFactory public methods

        BEAGLE_CPU_FACTORY_TEMPLATE
        BeagleImpl* BeagleCPUActionImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::createImpl(int tipCount,
                                                                                       int partialsBufferCount,
                                                                                       int compactBufferCount,
                                                                                       int stateCount,
                                                                                       int patternCount,
                                                                                       int eigenBufferCount,
                                                                                       int matrixBufferCount,
                                                                                       int categoryCount,
                                                                                       int scaleBufferCount,
                                                                                       int resourceNumber,
                                                                                       int pluginResourceNumber,
                                                                                       long long preferenceFlags,
                                                                                       long long requirementFlags,
                                                                                       int* errorCode) {

            BeagleImpl* impl = new BeagleCPUActionImpl<REALTYPE, T_PAD_DEFAULT, P_PAD_DEFAULT>();

            try {
                *errorCode =
                        impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                             patternCount, eigenBufferCount, matrixBufferCount,
                                             categoryCount,scaleBufferCount, resourceNumber,
                                             pluginResourceNumber,
                                             preferenceFlags, requirementFlags);
                if (*errorCode == BEAGLE_SUCCESS) {
                    return impl;
                }
                delete impl;
                return NULL;
            }
            catch(...) {
                if (DEBUGGING_OUTPUT)
                    std::cerr << "exception in initialize\n";
                delete impl;
                throw;
            }

            delete impl;

            return NULL;
        }

        BEAGLE_CPU_FACTORY_TEMPLATE
        const char* BeagleCPUActionImplFactory<BEAGLE_CPU_FACTORY_GENERIC>::getName() {
            return getBeagleCPUActionName<BEAGLE_CPU_FACTORY_GENERIC>();
        }

        template <>
        long long BeagleCPUActionImplFactory<double>::getFlags() {
            return BEAGLE_FLAG_COMPUTATION_SYNCH | BEAGLE_FLAG_COMPUTATION_ACTION |
                   BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
                   BEAGLE_FLAG_THREADING_NONE | BEAGLE_FLAG_THREADING_CPP |
                   BEAGLE_FLAG_PROCESSOR_CPU |
                   BEAGLE_FLAG_VECTOR_SSE | BEAGLE_FLAG_VECTOR_AVX | BEAGLE_FLAG_VECTOR_NONE |
                   BEAGLE_FLAG_PRECISION_DOUBLE |
                   BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                   BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
                   BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
                   BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                   BEAGLE_FLAG_FRAMEWORK_CPU;
        }

        template <>
        long long BeagleCPUActionImplFactory<float>::getFlags() {
            return BEAGLE_FLAG_COMPUTATION_SYNCH | BEAGLE_FLAG_COMPUTATION_ACTION |
                   BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
                   BEAGLE_FLAG_THREADING_NONE | BEAGLE_FLAG_THREADING_CPP |
                   BEAGLE_FLAG_PROCESSOR_CPU |
                   BEAGLE_FLAG_VECTOR_SSE | BEAGLE_FLAG_VECTOR_AVX | BEAGLE_FLAG_VECTOR_NONE |
                   BEAGLE_FLAG_PRECISION_SINGLE |
                   BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                   BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
                   BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
                   BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                   BEAGLE_FLAG_FRAMEWORK_CPU;
        }

    }
}



































#endif //BEAGLE_BEAGLECPUACTIONIMPL_HPP
