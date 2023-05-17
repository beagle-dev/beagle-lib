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
                                                                    long preferenceFlags,
                                                                    long requirementFlags) {
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
            gMappedPartials = (MapType **) malloc(sizeof(MapType *) * kBufferCount);
            gMappedPartialCache = (MapType **) malloc(sizeof(MapType *) * kBufferCount);
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

            for (int i = 0; i < kBufferCount; i++) {
                gMappedPartials[i] = NULL;
                gMappedPartialCache[i] = NULL;
            }

            return BEAGLE_SUCCESS;
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        int BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::setPartials(int bufferIndex,
                                                                       const double* inPartials) {
            BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::setPartials(bufferIndex, inPartials);
            if (gMappedPartials[bufferIndex] == NULL) {
                gMappedPartials[bufferIndex] = (MapType*) malloc(sizeof(MapType) * kCategoryCount);
                for (int category = 0; category < kCategoryCount; category++) {
                    new (& gMappedPartials[bufferIndex][category]) MapType(gPartials[bufferIndex] + category * kPaddedPatternCount * kStateCount, kStateCount, kPatternCount);
                }
            }
            return BEAGLE_SUCCESS;
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::~BeagleCPUActionImpl() {
            free(gMappedPartials);
            free(gMappedPartialCache);
            free(gMuBs);
            free(gB1Norms);
            free(gIntegrationTmp);
//            free(gScaledQs);
            free(gRescaleTmp);
            free(gEigenMaps);
            free(gEdgeMultipliers);
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        int BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::setTipPartials(int tipIndex,
                                                              const double* inPartials) {
            BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::setTipPartials(tipIndex, inPartials);
            if (gMappedPartials[tipIndex] == NULL) {
                gMappedPartials[tipIndex] = (MapType*) malloc(sizeof(MapType) * kCategoryCount);
                for (int category = 0; category < kCategoryCount; category++) {
                    new (& gMappedPartials[tipIndex][category]) MapType(gPartials[tipIndex] + category * kPaddedPatternCount * kStateCount, kStateCount, kPatternCount);
#ifdef BEAGLE_DEBUG_FLOW
                    std::cout<<gMappedPartials[tipIndex][category]<<std::endl;
#endif
                }
            }

            return BEAGLE_SUCCESS;
        }


        BEAGLE_CPU_FACTORY_TEMPLATE
        inline const char* getBeagleCPUActionName(){ return "CPU-Action-Unknown"; };

        template<>
        inline const char* getBeagleCPUActionName<double>(){ return "CPU-Action-Double"; };

        template<>
        inline const char* getBeagleCPUActionName<float>(){ return "CPU-Action-Single"; };

        BEAGLE_CPU_ACTION_TEMPLATE
        void BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::rescalePartials(MapType *destP,
                             double *scaleFactors,
                             double *cumulativeScaleFactors,
                             const int fillWithOnes) {
            memset(gRescaleTmp, 0, kPatternCount * sizeof(double));
            for (int category = 0; category < kCategoryCount; category++) {
                Eigen::VectorXd colMax = destP[category].colwise().maxCoeff();
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
                destP[category] *= gRescaleTmpMap.asDiagonal();
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
        int BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::updatePartials(const int *operations,
                                                                          int operationCount,
                                                                          int cumulativeScalingIndex) {
            double* cumulativeScaleBuffer = NULL;
            if (cumulativeScalingIndex != BEAGLE_OP_NONE)
                cumulativeScaleBuffer = gScaleBuffers[cumulativeScalingIndex];

            for (int op = 0; op < operationCount; op++) {
                int numOps = BEAGLE_OP_COUNT;

                const int destinationPartialIndex = operations[op * numOps];
                const int writeScalingIndex = operations[op * numOps + 1];
                const int readScalingIndex = operations[op * numOps + 2];
                const int firstChildPartialIndex = operations[op * numOps + 3];
                const int firstChildSubstitutionMatrixIndex = operations[op * numOps + 4];
                const int secondChildPartialIndex = operations[op * numOps + 5];
                const int secondChildSubstitutionMatrixIndex = operations[op * numOps + 6];

                if (gMappedPartials[destinationPartialIndex] == NULL) {
                    gMappedPartials[destinationPartialIndex] = (MapType*) malloc(sizeof(MapType) * kCategoryCount);
                    for (int category = 0; category < kCategoryCount; category++) {
                        new (& gMappedPartials[destinationPartialIndex][category]) MapType(gPartials[destinationPartialIndex] + category * kPaddedPatternCount * kStateCount, kStateCount, kPatternCount);
                    }
                }

                if (gMappedPartialCache[firstChildPartialIndex] == NULL) {
                    gMappedPartialCache[firstChildPartialIndex] = (MapType*) malloc(sizeof(MapType) * kCategoryCount);
                    for (int category = 0; category < kCategoryCount; category++) {
                        new (& gMappedPartialCache[firstChildPartialIndex][category]) MapType(gPartials[firstChildPartialIndex + kPartialsCacheOffset] + category * kPaddedPatternCount * kStateCount, kStateCount, kPatternCount);
                    }
                }

                if (gMappedPartialCache[secondChildPartialIndex] == NULL) {
                    gMappedPartialCache[secondChildPartialIndex] = (MapType*) malloc(sizeof(MapType) * kCategoryCount);
                    for (int category = 0; category < kCategoryCount; category++) {
                        new (& gMappedPartialCache[secondChildPartialIndex][category]) MapType(gPartials[secondChildPartialIndex + kPartialsCacheOffset] + category * kPaddedPatternCount * kStateCount, kStateCount, kPatternCount);
                    }
                }

                MapType* destP = gMappedPartials[destinationPartialIndex];
                MapType* partials1 = gMappedPartials[firstChildPartialIndex];
//                SpMatrix* matrices1 = gScaledQs[firstChildSubstitutionMatrixIndex];
                MapType* partials2 = gMappedPartials[secondChildPartialIndex];
//                SpMatrix* matrices2 = gScaledQs[secondChildSubstitutionMatrixIndex];
                MapType* partialCache1 = gMappedPartialCache[firstChildPartialIndex];
                MapType* partialCache2 = gMappedPartialCache[secondChildPartialIndex];




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
                calcPartialsPartials2(destP, partials1, partials2, firstChildSubstitutionMatrixIndex,
                                      secondChildSubstitutionMatrixIndex, partialCache1, partialCache2);

                if (rescale == 1) {
                    rescalePartials(destP, scalingFactors, cumulativeScaleBuffer, 0);
                }
            }



            return BEAGLE_SUCCESS;
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        int BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::updatePrePartials(const int *operations,
                                                                          int operationCount,
                                                                          int cumulativeScalingIndex) {
            double* cumulativeScaleBuffer = NULL;
            if (cumulativeScalingIndex != BEAGLE_OP_NONE)
                cumulativeScaleBuffer = gScaleBuffers[cumulativeScalingIndex];

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

                if (gMappedPartials[destinationPartialIndex] == NULL) {
                    gMappedPartials[destinationPartialIndex] = (MapType*) malloc(sizeof(MapType) * kCategoryCount);
                    for (int category = 0; category < kCategoryCount; category++) {
                        new (& gMappedPartials[destinationPartialIndex][category]) MapType(gPartials[destinationPartialIndex] + category * kPaddedPatternCount * kStateCount, kStateCount, kPatternCount);
                    }
                }

                MapType* destP = gMappedPartials[destinationPartialIndex];
                MapType* partials1 = gMappedPartials[parentIndex];
//                SpMatrix* matrices1 = gScaledQs[substitutionMatrixIndex];
                MapType* partials2 = gMappedPartials[siblingIndex];
//                SpMatrix* matrices2 = gScaledQs[siblingSubstitutionMatrixIndex];
                MapType* partialCache2 = gMappedPartialCache[siblingIndex];

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
                calcPrePartialsPartials2(destP, partials1, partials2, substitutionMatrixIndex,
                                         siblingSubstitutionMatrixIndex, partialCache2);

                if (rescale == 1) {
                    rescalePartials(destP, scalingFactors, cumulativeScaleBuffer, substitutionMatrixIndex);
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
        void BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::calcPartialsPartials2(MapType *destP, MapType *partials1,
                                                                                  MapType *partials2, int edgeIndex1,
                                                                                  int edgeIndex2,
                                                                                  MapType *partialCache1,
                                                                                  MapType *partialCache2) {
            simpleAction2(partialCache1, partials1, edgeIndex1, false);
            simpleAction2(partialCache2, partials2, edgeIndex2, false);

            for (int i = 0; i < kCategoryCount; i++) {
                destP[i] = partialCache1[i].cwiseProduct(partialCache2[i]);
            }

        }


        BEAGLE_CPU_ACTION_TEMPLATE
        void BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::calcPrePartialsPartials2(MapType *destP, MapType *partials1,
                                                                                     MapType *partials2, int edgeIndex1,
                                                                                     int edgeIndex2,
                                                                                     MapType *partialCache2) {
            memset(gIntegrationTmp, 0, (kPatternCount * kStateCount * kCategoryCount)*sizeof(double));

            for (int i = 0; i < kCategoryCount; i++) {
                gMappedIntegrationTmp[i] = partialCache2[i].cwiseProduct(partials1[i]);
            }

            simpleAction2(destP, gMappedIntegrationTmp, edgeIndex1, true);
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        void
        BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::simpleAction2(MapType *destP, MapType *partials, int edgeIndex,
                                                                     bool transpose) {
            for (int category = 0; category < kCategoryCount; category++) {
#ifdef BEAGLE_DEBUG_FLOW
                std::cerr<<"New impl 2\nRate category "<<category<<std::endl;
                std::cerr<<"In partial: \n"<<partials[category]<<std::endl;
#endif
                const double tol = pow(2.0, -53.0);
                const double t = 1.0;
                const int nCol = kPatternCount;
                int m, s;

                const double edgeMultiplier = gEdgeMultipliers[edgeIndex * kCategoryCount + category];

                getStatistics2(t, nCol, m, s, edgeMultiplier, gEigenMaps[edgeIndex]);


#ifdef BEAGLE_DEBUG_FLOW
                std::cerr<<" m = "<<m<<"  s = "<<s <<std::endl;
#endif

                destP[category] = partials[category];
                SpMatrix A = gBs[gEigenMaps[edgeIndex]] * edgeMultiplier;
                if (transpose) {
                    A = A.transpose();
                }

                MatrixXd F(kStateCount, kPatternCount);
                F = destP[category];

                const double eta = exp(t * gMuBs[gEigenMaps[edgeIndex]] * edgeMultiplier / (double) s);
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
        void BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::getStatistics2(double t, int nCol, int &m, int &s,
                                                                           double edgeMultiplier,
                                                                           int eigenIndex) {
            if (t * gB1Norms[eigenIndex] == 0.0) {
                m = 0;
                s = 1;
            } else {
                int bestM = INT_MAX;
                int bestS = INT_MAX;

                const double theta = thetaConstants[mMax];
                const double pMax = floor((0.5 + 0.5 * sqrt(5.0 + 4.0 * mMax)));
                // pMax is the largest positive integer such that p*(p-1) <= mMax + 1

                const bool conditionFragment313 = gB1Norms[eigenIndex] * edgeMultiplier <= 2.0 * theta / ((double) nCol * mMax) * pMax * (pMax + 3);
                // using l = 1 as in equation 3.13
                std::map<int, double>::iterator it;
                if (conditionFragment313) {
                    for (it = thetaConstants.begin(); it != thetaConstants.end(); it++) {
                        const int thisM = it->first;
                        const double thisS = ceil(gB1Norms[eigenIndex] * edgeMultiplier / thetaConstants[thisM]);
                        if (bestM == INT_MAX || ((double) thisM) * thisS < bestM * bestS) {
                            bestS = (int) thisS;
                            bestM = thisM;
                        }
                    }
                    s = bestS;
                } else {
                    if (gHighestPowers[eigenIndex] < 1) {
                        SpMatrix currentMatrix = gBs[eigenIndex];
                        powerMatrices[eigenIndex][1] = currentMatrix;
                        ds[eigenIndex][1] = normP1(&currentMatrix);
                        gHighestPowers[eigenIndex] = 1;
                    }
                    for (int p = 2; p < pMax; p++) {
                        for (int thisM = p * (p - 1) - 1; thisM < mMax + 1; thisM++) {
                            it = thetaConstants.find(thisM);
                            if (it != thetaConstants.end()) {
                                // equation 3.7 in Al-Mohy and Higham
                                const double dValueP = getDValue2(p, eigenIndex);
                                const double dValuePPlusOne = getDValue2(p + 1, eigenIndex);
                                const double alpha = (dValueP > dValuePPlusOne ? dValueP : dValuePPlusOne) * edgeMultiplier;
                                // part of equation 3.10
                                const double thisS = ceil(alpha / thetaConstants[thisM]);
                                if (bestM == INT_MAX || ((double) thisM) * thisS < bestM * bestS) {
                                    bestS = (int) thisS;
                                    bestM = thisM;
                                }
                            }
                        }
                    }
                    s = bestS > 1 ? bestS : 1;
                }
                m = bestM;
            }
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

                int m, s;
                getStatistics(A1Norm, &A, t, nCol, m, s);

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
        void BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::getStatistics(double A1Norm, SpMatrix * matrix, double t, int nCol, int &m, int &s) {
            if (t * A1Norm == 0.0) {
                m = 0;
                s = 1;
            } else {
                int bestM = INT_MAX;
                int bestS = INT_MAX;

                const double theta = thetaConstants[mMax];
                const double pMax = floor((0.5 + 0.5 * sqrt(5.0 + 4.0 * mMax)));
                // pMax is the largest positive integer such that p*(p-1) <= mMax + 1

                const bool conditionFragment313 = A1Norm <= 2.0 * theta / ((double) nCol * mMax) * pMax * (pMax + 3);
                // using l = 1 as in equation 3.13

                std::map<int, double>::iterator it;
                if (conditionFragment313) {
                    for (it = thetaConstants.begin(); it != thetaConstants.end(); it++) {
                        const int thisM = it->first;
                        const double thisS = ceil(A1Norm/thetaConstants[thisM]);
                        if (bestM == INT_MAX || ((double) thisM) * thisS < bestM * bestS) {
                            bestS = (int) thisS;
                            bestM = thisM;
                        }
                    }
                    s = bestS;
                } else {
                    std::map<int, double> d;
                    SpMatrix firstOrderMatrix = *matrix;
                    std::map<int, SpMatrix> powerMatrices;
                    powerMatrices[1] = firstOrderMatrix;
                    d[1] = normP1(&firstOrderMatrix);
                    for (int p = 2; p < pMax; p++) {
                        for (int thisM = p * (p - 1) - 1; thisM < mMax + 1; thisM++) {
                            it = thetaConstants.find(thisM);
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
                    s = bestS > 1 ? bestS : 1;
                }
                m = bestM;
            }
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
        const long BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::getFlags() {
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
                                                                                       long preferenceFlags,
                                                                                       long requirementFlags,
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
        const long BeagleCPUActionImplFactory<double>::getFlags() {
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
        const long BeagleCPUActionImplFactory<float>::getFlags() {
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
