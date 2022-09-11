/*
 *  BeagleCPUActionImpl.h
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

#ifndef BEAGLE_BEAGLECPUACTIONIMPL_H
#define BEAGLE_BEAGLECPUACTIONIMPL_H

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/CPU/BeagleCPUImpl.h"
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#define T_PAD_DEFAULT   1   // Pad transition matrix rows with an extra 1.0 for ambiguous characters
#define P_PAD_DEFAULT   0   // No partials padding necessary for non-SSE implementations


#define BEAGLE_CPU_ACTION_FLOAT	float, T_PAD, P_PAD
#define BEAGLE_CPU_ACTION_DOUBLE	double, T_PAD, P_PAD
#define BEAGLE_CPU_ACTION_TEMPLATE	template <int T_PAD, int P_PAD>

namespace beagle {
    namespace cpu {

        BEAGLE_CPU_TEMPLATE
        class BeagleCPUActionImpl : public BeagleCPUImpl<BEAGLE_CPU_GENERIC> {
        };

        BEAGLE_CPU_ACTION_TEMPLATE
        class BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE> : public BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE> {

        protected:
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kTipCount;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::integrationTmp;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kPatternCount;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kPaddedPatternCount;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kExtraPatterns;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kStateCount;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kCategoryCount;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::realtypeMin;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kMatrixSize;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kPartialsPaddedStateCount;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kBufferCount;
//            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::gStateFrequencies;
//            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::gCategoryWeights;
//            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::gScaleBuffers;
//            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::gTipStates;
//            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::gTransitionMatrices;
//            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::gPartials;
            Eigen::SparseMatrix<double>* gInstantaneousMatrices;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>** gPartials;
            Eigen::SparseMatrix<double>** gTipStates;

        public:
            virtual const char* getName();

            virtual const long getFlags();

            virtual int createInstance(int tipCount,
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
                                       long requirementFlags);

            virtual int setPartials(int bufferIndex,
                            const double* inPartials);
//        protected:
//            virtual int getPaddedPatternsModulus();

        private:
            virtual int setTransitionMatrix(int matrixIndex,
                                            const double *inMatrix,
                                            double paddedValue);

            virtual int setEigenDecomposition(int eigenIndex,
                                              const double *inEigenVectors,
                                              const double *inInverseEigenVectors,
                                              const double *inEigenValues);

            virtual int updateTransitionMatrices(int eigenIndex,
                                                 const int* probabilityIndices,
                                                 const int* firstDerivativeIndices,
                                                 const int* secondDerivativeIndices,
                                                 const double* edgeLengths,
                                                 int count);

            virtual void calcStatesStates(float *destP,
                                          const int *states1,
                                          const float *matrices1,
                                          const int *states2,
                                          const float *matrices2);

            virtual void calcStatesPartials(float* destP,
                                            const int* states1,
                                            const float* matrices1,
                                            const float* partials2,
                                            const float* matrices2);

            virtual void calcPartialsPartials(float* __restrict destP,
                                              const float* __restrict partials1,
                                              const float* __restrict matrices1,
                                              const float* __restrict partials2,
                                              const float* __restrict matrices2);


            virtual int calcEdgeLogLikelihoods(const int parentBufferIndex,
                                               const int childBufferIndex,
                                               const int probabilityIndex,
                                               const int categoryWeightsIndex,
                                               const int stateFrequenciesIndex,
                                               const int scalingFactorsIndex,
                                               double* outSumLogLikelihood);

        };


BEAGLE_CPU_FACTORY_TEMPLATE
class BeagleCPUActionImplFactory : public BeagleImplFactory {
public:
    virtual BeagleImpl* createImpl(int tipCount,
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
                                   int* errorCode);

    virtual const char* getName();
    virtual const long getFlags();
};



















    }
}

#include "libhmsbeagle/CPU/BeagleCPUActionImpl.hpp"

#endif //BEAGLE_BEAGLECPUACTIONIMPL_H
