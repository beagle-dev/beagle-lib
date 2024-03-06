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
            gInstantaneousMatrices = new SpMatrix[eigenDecompositionCount];
            gBs = new SpMatrix[eigenDecompositionCount];
            gMuBs = (double *) malloc(sizeof(double) * eigenDecompositionCount);
            gB1Norms = (double *) malloc(sizeof(double) * eigenDecompositionCount);
            gEigenMaps = (int *) malloc(sizeof(int) * kBufferCount);
            gEdgeMultipliers = (double *) malloc(sizeof(double) * kBufferCount * categoryCount);
//            gSimpleActions = (SimpleAction**) malloc(sizeof(SimpleAction *) * eigenDecompositionCount);
//            for (int eigen = 0; eigen < eigenDecompositionCount; eigen++) {
//                gSimpleActions[eigen] = (SimpleAction *) malloc(sizeof(SimpleAction));
//                SimpleAction* action = new SimpleAction();
//                action->createInstance(categoryCount, patternCount, stateCount, &gInstantaneousMatrices[eigen],
//                                       gEdgeMultipliers);
////                SimpleAction action(categoryCount, patternCount, stateCount);
//                gSimpleActions[eigen] = action;
//            }
            powerMatrices = new std::map<int, SpMatrix>[eigenDecompositionCount];
            ds = new std::map<int, double>[eigenDecompositionCount];
//            gScaledQs = new SpMatrix * [kBufferCount];
            gHighestPowers = (int *) malloc(sizeof(int) * eigenDecompositionCount);
            identity = SpMatrix(kStateCount, kStateCount);
            identity.setIdentity();
            gScaledQTransposeTmp = new SpMatrix[kCategoryCount];
            gIntegrationTmp = (double *) malloc(sizeof(double) * kStateCount * kPaddedPatternCount * kCategoryCount);

            gMappedIntegrationTmp = (MapType*) malloc(sizeof(MapType) * kCategoryCount);
            for (int category = 0; category < kCategoryCount; category++) {
                new (& gMappedIntegrationTmp[category]) MapType(gIntegrationTmp + category * kPaddedPatternCount * kStateCount, kStateCount, kPatternCount);
            }

            gRescaleTmp = (double *) malloc(sizeof(double) * kPatternCount);


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
            free(gMuBs);
            free(gB1Norms);
            free(gIntegrationTmp);
//            free(gScaledQs);
            free(gRescaleTmp);
            free(gEigenMaps);
            free(gEdgeMultipliers);
        }

        BEAGLE_CPU_FACTORY_TEMPLATE
        inline const char* getBeagleCPUActionName(){ return "CPU-Action-Unknown"; };

        template<>
        inline const char* getBeagleCPUActionName<double>(){ return "CPU-Action-Double"; };

        template<>
        inline const char* getBeagleCPUActionName<float>(){ return "CPU-Action-Single"; };

        BEAGLE_CPU_ACTION_TEMPLATE
        void BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::rescalePartials(int destPIndex,
									    double *scaleFactors,
									    double *cumulativeScaleFactors,
									    const int fillWithOnes) {
            memset(gRescaleTmp, 0, kPatternCount * sizeof(double));
            for (int category = 0; category < kCategoryCount; category++) {
                Eigen::VectorXd colMax = partialsMap(destPIndex,category).colwise().maxCoeff();
                for (int pattern = 0; pattern < kPatternCount; pattern++) {
                    if (gRescaleTmp[pattern] < colMax(pattern)) {
                        gRescaleTmp[pattern] = colMax(pattern);
                    }
                }
            }

            for (int pattern = 0; pattern < kPatternCount; pattern++) {
                gRescaleTmp[pattern] = gRescaleTmp[pattern] == 0 ? 1.0 : 1.0 / gRescaleTmp[pattern];
            }

            MapType gRescaleTmpMap(gRescaleTmp, 1, kPatternCount);

            for (int category = 0; category < kCategoryCount; category++) {
                partialsMap(destPIndex,category) *= gRescaleTmpMap.asDiagonal();
            }

            for (int pattern = 0; pattern < kPatternCount; pattern++) {
                if (kFlags & BEAGLE_FLAG_SCALERS_LOG) {
                    const double logInverseMax = log(gRescaleTmp[pattern]);
                    scaleFactors[pattern] = -logInverseMax;
                    if (cumulativeScaleFactors != NULL) {
                        cumulativeScaleFactors[pattern] -= logInverseMax;
                    }
                } else {
                    scaleFactors[pattern] = 1.0 / gRescaleTmp[pattern];
                    if (cumulativeScaleFactors != NULL) {
                        cumulativeScaleFactors[pattern] -= log(gRescaleTmp[pattern]);
                    }
                }
            }


        }


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
                    rescalePartials(destinationPartialIndex, scalingFactors, cumulativeScaleBuffer, 0);
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
                    rescalePartials(destinationPartialIndex, scalingFactors, cumulativeScaleBuffer, substitutionMatrixIndex);
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
            gHighestPowers[eigenIndex] = 0;

            double mu_B = 0.0;
            for (int i = 0; i < kStateCount; i++) {
                mu_B += gInstantaneousMatrices[eigenIndex].coeff(i, i);
            }
            mu_B /= (double) kStateCount;
            gMuBs[eigenIndex] = mu_B;
            gBs[eigenIndex] = gInstantaneousMatrices[eigenIndex] - mu_B * identity;
            gB1Norms[eigenIndex] = normP1(&gBs[eigenIndex]);

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
		simpleAction2(partials1Cache, partials1, edgeIndex1, category, false);

		auto partials2 = partialsMap(partials2Index, category, startPattern, endPattern);
		auto partials2Cache = partialsCacheMap(partials2Index, category, startPattern, endPattern);
		simpleAction2(partials2Cache, partials2, edgeIndex2, category, false);

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
		simpleAction2(destP, gMappedIntegrationTmp[category], edgeIndex1, category, true);
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
	    std::cerr<<" m = "<<m<<"  s = "<<s <<std::endl;
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
		    if (c1 + c2 <= tol * normPInf(&F)) {
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

	double factorial(int n)
	{
	    double f = 1;
	    for(int i=2;i<=n;i++)
		f *= double(i);
	    return f;
	}

        BEAGLE_CPU_ACTION_TEMPLATE
        void
        BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::simpleAction3(MapType& destP, MapType& partials, int edgeIndex,
                                                                     int category, bool transpose) {
#ifdef BEAGLE_DEBUG_FLOW
            std::cerr<<"New impl 2\nRate category "<<category<<std::endl;
	    std::cerr<<"In partial: \n"<<partials<<std::endl;
#endif
	    const double tol = pow(2.0, -53.0);
	    int m = 2;
	    constexpr int M = 55;

	    const double edgeMultiplier = gEdgeMultipliers[edgeIndex * kCategoryCount + category];

	    SpMatrix A = gInstantaneousMatrices[gEigenMaps[edgeIndex]] * edgeMultiplier;
	    if (transpose) {
		A = A.transpose();
	    }

	    MatrixXd v = partials;
// BEGIN
	    std::vector<MatrixXd> V(M);
	    V[1] = A*v; // L1
	    for(int k=2;k<=m+1;k++) // L2
		V[k] = A*V[k-1]; // L3
	    // L4
	    int s = ceil(pow(V[m+1].maxCoeff() / factorial(m+1) / tol, 1.0/(m+1))); // L5
	    int p = m * s; // L6
	    int f = 0; // L7
	    while (f == 0 and m < M) { // L8
		m = m + 1; // L9
		V[m+1] = A*V[m]; // L10
		int s1 = ceil(pow(V[m+1].maxCoeff()/factorial(m+1) / tol,1.0/(m+1))); //L11
		int p1 = m*s1; // L12
		if (p1 <= p) // L13
		{
		    p = p1; // L14
		    s = s1; // L15
		}
		else
		{
		    m = m-1; // L17
		    f = 1; // L18
		} //L19
	    } // L20
#ifdef BEAGLE_DEBUG_FLOW
	    std::cerr<<"simpleAction3: m = "<<m<<"  s = "<<s <<std::endl;
#endif
	    MatrixXd w = v; // L 21
	    for(int k=1;k<=m;k++) { // L22
		w += V[k]/pow(s,k)/factorial(k); // L23
	    } //L24
	    A /= s;  // L25
	    for(int i=2;i<=s;i++) { // L26
		v = w; // L27
		for(int k=1;k<=m;k++) { // L28
		    v = A*v; // L29
		    w += v/factorial(k); // L30
		} // L31
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
		if (gHighestPowers[eigenIndex] < 1) {
		    SpMatrix currentMatrix = gBs[eigenIndex];
		    powerMatrices[eigenIndex][1] = currentMatrix;
		    ds[eigenIndex][1] = normP1(&currentMatrix);
		    gHighestPowers[eigenIndex] = 1;
		}
		for (int p = 2; p < pMax; p++) {
		    for (int thisM = p * (p - 1) - 1; thisM < mMax + 1; thisM++) {
			auto it = thetaConstants.find(thisM);
			if (it != thetaConstants.end()) {
			    // equation 3.7 in Al-Mohy and Higham
			    const double dValueP = getDValue2(p, eigenIndex);
			    const double dValuePPlusOne = getDValue2(p + 1, eigenIndex);
			    const double alpha = (dValueP > dValuePPlusOne ? dValueP : dValuePPlusOne) * edgeMultiplier;
			    // part of equation 3.10
			    const double thisS = ceil(alpha / thetaConstants[thisM]);
			    if (bestM == INT_MAX || ((double) thisM) * thisS < bestM * bestS) {
				bestS = thisS;
				bestM = thisM;
			    }
			}
		    }
		}
		bestS = std::max(bestS, 1.0);
	    }

	    int m = bestM;
	    int s = (int) std::min<double>(bestS, INT_MAX);

	    assert(m >= 0);
	    assert(s >= 0);

	    return {m,s};
        }


        BEAGLE_CPU_ACTION_TEMPLATE
        void BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::simpleAction(MapType* destP,
                                                                         MapType* partials,
                                                                         SpMatrix* matrix) {
            for (int category = 0; category < kCategoryCount; category++) {
                SpMatrix thisMatrix = matrix[category];
#ifdef BEAGLE_DEBUG_FLOW
                std::cerr<<"OLD impl \nRate category "<<category<<std::endl;
                std::cerr<<"In partial: \n"<<partials[category]<<std::endl;
                std::cerr<<"Matrix: \n"<<thisMatrix<<std::endl;
#endif
                const double tol = pow(2.0, -53.0);
                const double t = 1.0;
                const int nCol = kPatternCount;
                SpMatrix identity(kStateCount, kStateCount);
                identity.setIdentity();
                double mu = 0.0;
//            for (int k=0; k<matrix->outerSize(); ++k) {
//                for (Eigen::SparseMatrix<double>::InnerIterator it(*matrix,k); it; ++it) {
//                    if (it.col() == it.row()) {
//                        mu += it.value();
//                    }
//                }
//            }
                for (int i = 0; i < kStateCount; i++) {
                    mu += thisMatrix.coeff(i, i);
                }
                mu /= (double) kStateCount;

                SpMatrix A(thisMatrix.rows(), thisMatrix.cols());

                A = thisMatrix - mu * identity;
                const double A1Norm = normP1(&A);

                auto [m,s] = getStatistics(A1Norm, &A, t, nCol);

#ifdef BEAGLE_DEBUG_FLOW
                std::cerr<<" m = "<<m<<"  s = "<<s <<std::endl;
                std::cerr<<"  A1Norm =" << A1Norm<< "  A ="<< std::endl<<A<<std::endl;
#endif

                destP[category] = partials[category];

                MatrixXd F(kStateCount, kPatternCount);
                F = destP[category];

                const double eta = exp(t * mu / (double) s);
                double c1, c2;
                for (int i = 0; i < s; i++) {
                    c1 = normPInf(destP[category]);
                    for (int j = 1; j < m + 1; j++) {
                        destP[category] = A * destP[category];
                        destP[category] *= t / ((double) s * j);
                        c2 = normPInf(destP[category]);
                        F += destP[category];
                        if (c1 + c2 <= tol * normPInf(&F)) {
                            break;
                        }
                        c1 = c2;
                    }
                    F *= eta;
                    destP[category] = F;
                }
#ifdef BEAGLE_DEBUG_FLOW
                std::cerr<<"Out partials: \n"<<destP[category]<<std::endl;
#endif
            }
        }


        BEAGLE_CPU_ACTION_TEMPLATE
        std::tuple<int,int> BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::getStatistics(double A1Norm, SpMatrix * matrix, double t, int nCol) {
	    assert( t >= 0 );
	    assert( A1Norm >= 0.0 );
	    assert( nCol >= 0 );

            if (t * A1Norm == 0.0)
                return {0,1};

	    int bestM = INT_MAX;
	    double bestS = INT_MAX;

	    const double theta = thetaConstants[mMax];
	    const double pMax = floor((0.5 + 0.5 * sqrt(5.0 + 4.0 * mMax)));
	    // pMax is the largest positive integer such that p*(p-1) <= mMax + 1

	    const bool conditionFragment313 = A1Norm <= 2.0 * theta / ((double) nCol * mMax) * pMax * (pMax + 3);
	    // using l = 1 as in equation 3.13

	    if (conditionFragment313) {
		for (auto& [thisM, thetaM]: thetaConstants) {
		    const double thisS = ceil(A1Norm/thetaM);
		    if (bestM == INT_MAX || ((double) thisM) * thisS < bestM * bestS) {
			bestS = thisS;
			bestM = thisM;
		    }
		}
	    } else {
		std::map<int, double> d;
		SpMatrix firstOrderMatrix = *matrix;
		std::map<int, SpMatrix> powerMatrices;
		powerMatrices[1] = firstOrderMatrix;
		d[1] = normP1(&firstOrderMatrix);
		for (int p = 2; p < pMax; p++) {
		    for (int thisM = p * (p - 1) - 1; thisM < mMax + 1; thisM++) {
			auto it = thetaConstants.find(thisM);
			if (it != thetaConstants.end()) {
			    // equation 3.7 in Al-Mohy and Higham
			    const double dValueP = getDValue(p, d, powerMatrices);
			    const double dValuePPlusOne = getDValue(p + 1, d, powerMatrices);
			    const double alpha = dValueP > dValuePPlusOne ? dValueP : dValuePPlusOne;
			    // part of equation 3.10
			    const double thisS = ceil(alpha / thetaConstants[thisM]);
			    if (bestM == INT_MAX || ((double) thisM) * thisS < bestM * bestS) {
				bestS = (int) thisS;
				bestM = thisM;
			    }
			}
		    }
		}
		bestS = std::max(bestS, 1.0);
	    }

	    int m = bestM;
	    int s = (int) std::min<double>(bestS, INT_MAX);

	    assert(m >= 0);
	    assert(s >= 0);

	    return {m,s};
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        double BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::normPInf(SpMatrix* matrix) {
            return ((*matrix).cwiseAbs() * Eigen::VectorXd::Ones(matrix->cols())).maxCoeff();
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        double BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::normPInf(MapType matrix) {
            return matrix.lpNorm<Eigen::Infinity>();
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        double BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::normPInf(MatrixXd* matrix) {
            return matrix->lpNorm<Eigen::Infinity>();
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        double BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::getDValue(int p, std::map<int, double> &d, std::map<int, SpMatrix> &powerMatrices) {
            // equation 3.7 in Al-Mohy and Higham
            std::map<int, double>::iterator it;
            it = d.find(p);
            if (it == d.end()) {
                const int highestPower = d.rbegin()->first;
                if (highestPower < p) {
                    for (int i = highestPower; i < p; i++) {
                        SpMatrix currentPowerMatrix = powerMatrices[i];
                        SpMatrix nextPowerMatrix = currentPowerMatrix * powerMatrices[1];
                        powerMatrices[i + 1] = nextPowerMatrix;
                    }
                }
                SpMatrix powerPMatrix = powerMatrices[p];
                d[p] = pow(normP1(&powerPMatrix), 1.0 / ((double) p));
            }
            return d[p];
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        double BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::getDValue2(int p, int eigenIndex) {
            // equation 3.7 in Al-Mohy and Higham
            std::map<int, double>::iterator it;
            it = ds[eigenIndex].find(p);
            if (it == ds[eigenIndex].end()) {
                const int cachedHighestPower = ds[eigenIndex].rbegin()->first;
                if (gHighestPowers[eigenIndex] < p) {
                    for (int i = gHighestPowers[eigenIndex]; i < (cachedHighestPower > p ? p : cachedHighestPower); i++) {
                        powerMatrices[eigenIndex][i + 1] = powerMatrices[eigenIndex][i] * powerMatrices[eigenIndex][1];
                        gHighestPowers[eigenIndex] = i + 1;
                        ds[eigenIndex][i + 1] = pow(normP1(&powerMatrices[eigenIndex][i + 1]), 1.0 / ((double) i + 1));
                    }

                    for (int i = gHighestPowers[eigenIndex]; i < p; i++) {
                        SpMatrix nextPowerMatrix = powerMatrices[eigenIndex][i] * powerMatrices[eigenIndex][1];
                        powerMatrices[eigenIndex][i + 1] = nextPowerMatrix;
                        ds[eigenIndex][i + 1] = pow(normP1(&powerMatrices[eigenIndex][i + 1]), 1.0 / ((double) i + 1));
                    }
                    gHighestPowers[eigenIndex] = p;
                }
            }
            return ds[eigenIndex][p];
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        double BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::normP1(SpMatrix * matrix) {
            return (Eigen::RowVectorXd::Ones(matrix -> rows()) * matrix -> cwiseAbs()).maxCoeff();
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
