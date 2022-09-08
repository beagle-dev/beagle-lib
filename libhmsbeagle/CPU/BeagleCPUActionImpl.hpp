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
        void BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::calcStatesStates(float* destP,
                                                                             const int* states1,
                                                                             const float* matrices1,
                                                                             const int* states2,
                                                                             const float* matrices2) {
        }


/*
 * Calculates partial likelihoods at a node when one child has states and one has partials.
   SSE version
 */
        BEAGLE_CPU_ACTION_TEMPLATE
        void BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::calcStatesPartials(float* destP,
                                                                               const int* states1,
                                                                               const float* matrices1,
                                                                               const float* partials2,
                                                                               const float* matrices2) {

        }


        BEAGLE_CPU_ACTION_TEMPLATE
        void BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::calcPartialsPartials(float* __restrict destP,
                                                                              const float* __restrict partials1,
                                                                              const float* __restrict matrices1,
                                                                              const float* __restrict partials2,
                                                                              const float* __restrict matrices2) {

        }

        BEAGLE_CPU_ACTION_TEMPLATE
        int BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::calcEdgeLogLikelihoods(const int parentBufferIndex,
                                                                                   const int childBufferIndex,
                                                                                   const int probabilityIndex,
                                                                                   const int categoryWeightsIndex,
                                                                                   const int stateFrequenciesIndex,
                                                                                   const int scalingFactorsIndex,
                                                                                   double* outSumLogLikelihood) {
            return BEAGLE_SUCCESS;

        }


        BEAGLE_CPU_ACTION_TEMPLATE
        const char* BeagleCPUActionImpl<BEAGLE_CPU_ACTION_FLOAT>::getName() {
            return  getBeagleCPUActionName<float>();
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        const char* BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::getName() {
            return  getBeagleCPUActionName<double>();
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        const long BeagleCPUActionImpl<BEAGLE_CPU_ACTION_FLOAT>::getFlags() {
            return  BEAGLE_FLAG_COMPUTATION_SYNCH |
                    BEAGLE_FLAG_PROCESSOR_CPU |
                    BEAGLE_FLAG_PRECISION_SINGLE |
                    BEAGLE_FLAG_VECTOR_SSE |
                    BEAGLE_FLAG_FRAMEWORK_CPU;
        }

        BEAGLE_CPU_ACTION_TEMPLATE
        const long BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE>::getFlags() {
            return  BEAGLE_FLAG_COMPUTATION_SYNCH |
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
            return BEAGLE_FLAG_COMPUTATION_SYNCH |
                   BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
                   BEAGLE_FLAG_THREADING_NONE | BEAGLE_FLAG_THREADING_CPP |
                   BEAGLE_FLAG_PROCESSOR_CPU |
                   BEAGLE_FLAG_VECTOR_SSE |
                   BEAGLE_FLAG_PRECISION_DOUBLE |
                   BEAGLE_FLAG_SCALERS_LOG | BEAGLE_FLAG_SCALERS_RAW |
                   BEAGLE_FLAG_EIGEN_COMPLEX | BEAGLE_FLAG_EIGEN_REAL |
                   BEAGLE_FLAG_INVEVEC_STANDARD | BEAGLE_FLAG_INVEVEC_TRANSPOSED |
                   BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL | BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO |
                   BEAGLE_FLAG_FRAMEWORK_CPU;
        }

        template <>
        const long BeagleCPUActionImplFactory<float>::getFlags() {
            return BEAGLE_FLAG_COMPUTATION_SYNCH |
                   BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALING_ALWAYS | BEAGLE_FLAG_SCALING_AUTO |
                   BEAGLE_FLAG_THREADING_NONE | BEAGLE_FLAG_THREADING_CPP |
                   BEAGLE_FLAG_PROCESSOR_CPU |
                   BEAGLE_FLAG_VECTOR_SSE |
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
