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
            int parentCode = BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::createInstance(tipCount, partialsBufferCount, compactBufferCount,
                                                                               stateCount, patternCount, eigenDecompositionCount,
                                                                               matrixCount, categoryCount, scaleBufferCount,
                                                                               resourceNumber, pluginResourceNumber,
                                                                               preferenceFlags, requirementFlags);
            gInstantaneousMatrices = new SpMatrix*[eigenDecompositionCount];
            gScaledQs = new SpMatrix * [kBufferCount];
            gMappedPartials = (MapType **) malloc(sizeof(MapType *) * kBufferCount);
            gIntegrationTmp = (double *) malloc(sizeof(double) * kStateCount * kPaddedPatternCount * kCategoryCount);
            gLeftPartialTmp = (double *) malloc(sizeof(double) * kStateCount * kPaddedPatternCount * kCategoryCount);
            gRightPartialTmp = (double *) malloc(sizeof(double) * kStateCount * kPaddedPatternCount * kCategoryCount);

            gMappedIntegrationTmp = (MapType*) malloc(sizeof(MapType) * kCategoryCount);
            for (int category = 0; category < kCategoryCount; category++) {
                new (& gMappedIntegrationTmp[category]) MapType(gIntegrationTmp + category * kPaddedPatternCount * kStateCount, kStateCount, kPatternCount);
            }
            gMappedLeftPartialTmp = (MapType*) malloc(sizeof(MapType) * kCategoryCount);
            for (int category = 0; category < kCategoryCount; category++) {
                new (& gMappedLeftPartialTmp[category]) MapType(gLeftPartialTmp + category * kPaddedPatternCount * kStateCount, kStateCount, kPatternCount);
           }
            gMappedRightPartialTmp = (MapType*) malloc(sizeof(MapType) * kCategoryCount);
            for (int category = 0; category < kCategoryCount; category++) {
                new (& gMappedRightPartialTmp[category]) MapType(gRightPartialTmp + category * kPaddedPatternCount * kStateCount, kStateCount, kPatternCount);
            }


            for (int i = 0; i < eigenDecompositionCount; i++) {
                gInstantaneousMatrices[i] = NULL;
            }

            for (int i = 0; i < kBufferCount; i++) {
                gScaledQs[i] = NULL;
            }

            for (int i = 0; i < kBufferCount; i++) {
                gMappedPartials[i] = NULL;
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
            free(gIntegrationTmp);
            free(gLeftPartialTmp);
            free(gRightPartialTmp);
            free(gInstantaneousMatrices);
            free(gScaledQs);

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
        int BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::setTransitionMatrix(int matrixIndex,
                                const double *inMatrix,
                                double paddedValue) {
            return BEAGLE_SUCCESS;
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

                MapType* destP = gMappedPartials[destinationPartialIndex];
                MapType* partials1 = gMappedPartials[firstChildPartialIndex];
                SpMatrix* matrices1 = gScaledQs[firstChildSubstitutionMatrixIndex];
                MapType* partials2 = gMappedPartials[secondChildPartialIndex];
                SpMatrix* matrices2 = gScaledQs[secondChildSubstitutionMatrixIndex];




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

                if (rescale == 0) {
                    std::cout<<partials1[0](0)<<std::endl;
                    calcPartialsPartials(destP, partials1, matrices1, partials2, matrices2);
                }
            }



            return BEAGLE_SUCCESS;
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        int BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::setEigenDecomposition(int eigenIndex,
                                                                                 const double *inEigenVectors,
                                                                                 const double *inInverseEigenVectors,
                                                                                 const double *inEigenValues) {
            if (gInstantaneousMatrices[eigenIndex] == NULL) {
                gInstantaneousMatrices[eigenIndex] = new SpMatrix[kCategoryCount];
                for (int i = 0; i < kCategoryCount; i++) {
//                    new (& gInstantaneousMatrices[eigenIndex][i]) SpMatrix (kStateCount, kStateCount);
                    SpMatrix matrix(kStateCount, kStateCount);
                    gInstantaneousMatrices[eigenIndex][i] = matrix;
                }
            }

            const int numNonZeros = (int) inInverseEigenVectors[0] / kCategoryCount;
            for (int category = 0; category < kCategoryCount; category++) {
                std::vector<Triplet> tripletList;
                for (int i = 0; i < numNonZeros; i++) {
                    tripletList.push_back(Triplet((int) inEigenVectors[2 * i + 2 * category * numNonZeros], (int) inEigenVectors[2 * i + 1 + 2 * category * numNonZeros], inEigenValues[i + category * numNonZeros]));
                }
                gInstantaneousMatrices[eigenIndex][category].setFromTriplets(tripletList.begin(), tripletList.end());
            }
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

                if (gScaledQs[nodeIndex] == NULL) {
                    gScaledQs[nodeIndex] = new SpMatrix[kCategoryCount];
//                gScaledQs[eigenIndex] = (SpMatrix **) malloc(sizeof(SpMatrix *) * kCategoryCount);
                    for (int i = 0; i < kCategoryCount; i++) {
                        SpMatrix matrix(kStateCount, kStateCount);
                        gScaledQs[nodeIndex][i] = matrix;
                    }
                }
                for (int i = 0; i < kCategoryCount; i++) {
                    const double categoryRate = gCategoryRates[eigenIndex][i];
                    gScaledQs[nodeIndex][i] = gInstantaneousMatrices[eigenIndex][i] * (edgeLengths[0] * categoryRate);
                }
            }
            return BEAGLE_SUCCESS;
        }


        BEAGLE_CPU_ACTION_TEMPLATE
        void BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::calcPartialsPartials(MapType* destP,
                                                                              MapType* partials1,
                                                                              SpMatrix* matrices1,
                                                                              MapType* partials2,
                                                                              SpMatrix* matrices2) {
            memset(gLeftPartialTmp, 0, (kPatternCount * kStateCount * kCategoryCount)*sizeof(double));
            std::cout<<"Initial Left partial: \n"<<gMappedLeftPartialTmp[0]<<std::endl;
            std::cout<<"In partial: \n"<<partials1[0]<<"\nMatrix: \n"<<matrices1[0]<<std::endl;
            memset(gRightPartialTmp, 0, (kPatternCount * kStateCount * kCategoryCount)*sizeof(double));
            simpleAction(gMappedLeftPartialTmp, partials1, matrices1);
            std::cout<<"Outpartial: \n"<<gMappedLeftPartialTmp[0]<<std::endl;
            simpleAction(gMappedRightPartialTmp, partials2, matrices2);

            for (int i = 0; i < kCategoryCount; i++) {
//                std::cout<< "left partial: " << gMappedLeftPartialTmp[i] << "\n right partial: " << gMappedRightPartialTmp[i]<< std::endl;
                destP[i] = gMappedLeftPartialTmp[i].cwiseProduct(gMappedRightPartialTmp[i]);
            }

        }

        BEAGLE_CPU_ACTION_TEMPLATE
        void BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::simpleAction(MapType* destP,
                                                                         MapType* partials,
                                                                         SpMatrix* matrix) {
            for (int category = 0; category < kCategoryCount; category++) {
                SpMatrix thisMatrix = matrix[category];
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

//                std::cout<<"matrix = "<<thisMatrix<<"\nmu = "<<mu<<std::endl;


                SpMatrix A(thisMatrix.rows(), thisMatrix.cols());

                A = thisMatrix - mu * identity;
                const double A1Norm = normP1(&A);

                int m, s;
                getStatistics(A1Norm, &A, t, nCol, m, s);


                destP[category] = partials[category];

                MatrixXd F(kStateCount, kPatternCount);
                F = destP[category];

//                std::cout<<"F = "<<F<<"\n B = "<<destP[category]<<std::endl;
                const double eta = exp(t * mu / (double) s);
                double c1, c2;
                for (int i = 0; i < s; i++) {
                    c1 = normPInf(destP[category]);
                    for (int j = 1; j < m + 1; j++) {
                        destP[category] = A * destP[category];
                        destP[category] *= t / ((double) s * j);
//                        std::cout<<"Pos 1, F = "<<F<<"\n B = "<<destP[category]<<std::endl;
                        c2 = normPInf(destP[category]);
                        F += destP[category];
//                        std::cout<<"Pos 2, F = "<<F<<"\n B = "<<destP[category]<<std::endl;
                        if (c1 + c2 <= tol * normPInf(&F)) {
                            break;
                        }
                        c1 = c2;
                    }
                    F *= eta;
                    destP[category] = F;
//                    std::cout<<"F = "<<F<<"\n B = "<<destP[category]<<std::endl;
                }
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
                    std::map<int, SpMatrix> powerMatrices = {
                            {1, firstOrderMatrix}
                    };
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
            if (it != d.end()) {
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
        double BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::normP1(SpMatrix * matrix) {
            double norm = 0;
            double* colSums = new double[kStateCount];
            for (int k=0; k < matrix->outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(*matrix, k); it; ++it) {
                    colSums[it.col()] += abs(it.value());
                    if (norm < colSums[it.col()])
                        norm = colSums[it.col()];
                }
            }
            return norm;
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
