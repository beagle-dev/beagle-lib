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
#include <random>

#include "libhmsbeagle/beagle.h"
#include "libhmsbeagle/CPU/BeagleCPUImpl.h"
#include "libhmsbeagle/CPU/BeagleCPUActionImpl.h"

using std::vector;
using std::tuple;
using Eigen::MatrixXi;

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

std::independent_bits_engine<std::mt19937_64,1,unsigned short> engine;

bool random_bool()
{
    return engine();
}

double random_plus_minus_1_func(double x)
{
    if (random_bool())
	return 1;
    else
	return -1;
}

// See OneNormEst in https://eprints.maths.manchester.ac.uk/2195/1/thesis-main.pdf
// See also https://github.com/gnu-octave/octave/blob/default/scripts/linear-algebra/normest1.m
double normest1(const SpMatrix& A, int p, int t=2, int itmax=5)
{
    assert(p >= 0);
    assert(t != 0); // negative means t = n
    assert(itmax >= 1);

    if (p == 0) return 1.0;

    // A is (n,n);
    assert(A.rows() == A.cols());
    int n = A.cols();

    // Handle t too large
    t = std::min(n,t);

    // Interpret negative t as t == n
    if (t < 0) t = n;

    // Defer to normP1 if p=1 and n is small or we want an exact answer.
    if (p == 1 and (n <= 4 or t == n))
	return normP1(A);

    // (0) Choose starting matrix X that is (n,t) with columns of unit 1-norm.
    MatrixXd X(n,t);
    // We choose the first column to be all 1s.
    X.col(0).setOnes();
    // The other columns have randomly chosen {-1,+1} entries.
    X = X.unaryExpr( &random_plus_minus_1_func );
    // Divide by n so that the norm of each column is 1.
    X /= n;

    // 3.
    std::vector<bool> ind_hist(n,0);
    std::vector<int> indices(n,0);
    int ind_best = -1;
    double est_old = 0;
    MatrixXd S = MatrixXd::Ones(n,t);
    MatrixXd S_old = MatrixXd::Ones(n,t);
    MatrixXi prodS(t,t);
    MatrixXd Y(n,t);
    MatrixXd Z(n,t);
    Eigen::VectorXd h(n);

    for(int k=1; k<=itmax; k++)
    {
	// std::cerr<<"iter "<<k<<"\n";
	Y = A*X; // Y is (n,n) * (n,t) = (n,t)
	for(int i=1;i<p;i++)
	    Y = A*Y;

	auto [est, j] = ArgNormP1(Y);

	if (est > est_old or k == 2)
	{
	    ind_best = indices[j];
	    // w = Y.col(ind_best);
	}
	assert(ind_best < n);

        // (1) of Algorithm 2.4
	if (est < est_old and k >= 2)
	{
	    // std::cerr<<"  The new estimate ("<<est<<") is smaller than the old estimate ("<<est_old<<")\n";
	    return est_old;
	}

	est_old = est;

	// S = sign(Y), 0.0 -> 1.0
	S = Y.unaryExpr([](const double& x) {return (x>=0) ? 1.0 : -1.0 ;});

	// prodS is (t,t)
	prodS = (S_old.transpose() * S).matrix().cast<int>() ;

	// (2) If each columns in S is parallel to SOME column of S_old
	if (prodS.cwiseAbs().colwise().maxCoeff().sum() == n * t and k >= 2)
	{
	    // std::cerr<<"  All columns of S parallel to S_old\n";
	    // converged = true
	    return est;
	}

        if (t > 1)
        {
            // If S(j) is parallel to S_old(i), replace S(j) with random column
            for(int j=0;j<S.cols();j++)
            {
                for(int i=0;i<S_old.cols();i++)
                    if (prodS(i,j) == n or prodS(i,j) == -n)
                    {
                        // std::cerr<<"  S.col("<<j<<") parallel to S_old.col("<<i<<")    prodS(i,j) = "<<prodS(i,j)<<"\n";
                        S.col(j) = S.col(j).unaryExpr( &random_plus_minus_1_func );
                        break;
                    }
            }

            // If S(j) is parallel to S(i) for i<j, replace S(j) with random column
            prodS = (S.transpose() * S).matrix().cast<int>() ;
            for(int i=0;i<S.cols();i++)
                for(int j=i+1;j<S.cols();j++)
                    if (prodS(i,j) == n or prodS(i,j) == -n)
                    {
                        // std::cerr<<"  S.col("<<j<<") parallel to S.col("<<i<<")    prodS(i,j) = "<<prodS(i,j)<<"\n";
                        S.col(j) = S.col(j).unaryExpr( &random_plus_minus_1_func );
                    }
        }

        // (3) of Algorithm 2.4
	Z = A.transpose() * S; // (t,n) * (n,t) -> (t,t)

	h = Z.cwiseAbs().rowwise().maxCoeff();

	// (4) of Algorithm 2.4
	if (k >= 2 and h.maxCoeff() == h[ind_best])
	{
	    // std::cerr<<"  The best column ("<<ind_best<<") is not new\n";
	    return est;
	}

	indices.resize(n);
	for(int i=0;i<n;i++)
	    indices[i] = i;

	// reorder idx so that the highest values of h[indices[i]] come first.
	std::sort(indices.begin(), indices.end(), [&](int i,int j) {return h[i] > h[j];});

	// (5) of Algorithm 2.4
	int n_found = 0;
	for(int i=0;i<t;i++)
	    if (ind_hist[indices[i]])
		n_found++;

	if (n_found == t)
	{
	    // std::cerr<<"  All columns were found in the column history.\n";
	    return est;
	}

	// find the first t indices that are not in ind_hist
	int l=0;
	for(int i=0;i<indices.size() and l < t;i++)
	{
	    if (not ind_hist[indices[i]])
	    {
		indices[l] = indices[i];
		l++;
	    }
	}
	indices.resize( std::min(l,t) );
	assert(not indices.empty());

	int tmax = std::min<int>(t, indices.size());

	X = MatrixXd::Zero(n, tmax);
	for(int j=0; j < tmax; j++)
	    X(indices[j], j) = 1; // X(:,j) = e(indices[j])

	for(int i: indices)
	    ind_hist[i] = true;

	S_old = S;
    }

    return est_old;
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
            for (int i = 0; i < eigenDecompositionCount; i++)
                gInstantaneousMatrices[i] = SpMatrix(kStateCount, kStateCount);
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

            gIntegrationTmp = new double[kStateCount * kPaddedPatternCount * kCategoryCount];

	    // TODO Eliminate this with an inline member function!
            gMappedIntegrationTmp = (MapType*) malloc(sizeof(MapType) * kCategoryCount);
            for (int category = 0; category < kCategoryCount; category++) {
                new (& gMappedIntegrationTmp[category]) MapType(gIntegrationTmp + category * kPaddedPatternCount * kStateCount, kStateCount, kPatternCount);
            }

            return BEAGLE_SUCCESS;
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::~BeagleCPUActionImpl() {
            delete[] gIntegrationTmp;
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
