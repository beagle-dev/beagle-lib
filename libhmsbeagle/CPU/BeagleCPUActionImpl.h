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

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Map<MatrixXd> MapType;
typedef Eigen::SparseMatrix<double> SpMatrix;
typedef Eigen::Triplet<double> Triplet;



namespace beagle {
    namespace cpu {
        class SimpleAction;
        BEAGLE_CPU_TEMPLATE
        class BeagleCPUActionImpl : public BeagleCPUImpl<BEAGLE_CPU_GENERIC> {
        };

        BEAGLE_CPU_ACTION_TEMPLATE
        class BeagleCPUActionImpl<BEAGLE_CPU_ACTION_DOUBLE> : public BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE> {

        protected:
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kTipCount;
//            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::integrationTmp;
//            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::firstDerivTmp;
//            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::secondDerivTmp;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kPatternCount;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kPaddedPatternCount;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kExtraPatterns;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kStateCount;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kCategoryCount;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::realtypeMin;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kMatrixSize;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kPartialsPaddedStateCount;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kBufferCount;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kEigenDecompCount;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::gPartials;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::gCategoryRates;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::gScaleBuffers;
            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::kFlags;
//            SpMatrix** gScaledQs;
            int kPartialsCacheOffset;
            MapType** gMappedPartials;
            MapType** gMappedPartialCache;
//            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::gStateFrequencies;
//            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::gTipStates;
//            using BeagleCPUImpl<BEAGLE_CPU_ACTION_DOUBLE>::gTransitionMatrices;
            double* gIntegrationTmp;
//            double* gLeftPartialTmp;
//            double* gRightPartialTmp;
            SpMatrix* gInstantaneousMatrices;
            SpMatrix* gBs;
            double* gMuBs;
            double* gB1Norms;
            int* gEigenMaps;
            double* gEdgeMultipliers;
            std::map<int, SpMatrix>* powerMatrices;
            std::map<int, double>* ds;
            int* gHighestPowers;
            SpMatrix identity;
            SpMatrix* gScaledQTransposeTmp;
            MapType* gMappedIntegrationTmp;
//            MapType* gMappedLeftPartialTmp;
//            MapType* gMappedRightPartialTmp;
            double* gRescaleTmp;
            const int mMax = 55;
            std::map<int, double> thetaConstants = {
                    //The first 30 values are from table A.3 of  Computing Matrix Functions.
                    // For double precision, tol = 2^(-53)
                    // TODO: maybe calculate this
                    {1, 2.29E-16},
                    {2, 2.58E-8},
                    {3, 1.39E-5},
                    {4, 3.40E-4},
                    {5, 2.40E-3},
                    {6, 9.07E-3},
                    {7, 2.38E-2},
                    {8, 5.00E-2},
                    {9, 8.96E-2},
                    {10, 1.44E-1},
                    {11, 2.14E-1},
                    {12, 3.00E-1},
                    {13, 4.00E-1},
                    {14, 5.14E-1},
                    {15, 6.41E-1},
                    {16, 7.81E-1},
                    {17, 9.31E-1},
                    {18, 1.09},
                    {19, 1.26},
                    {20, 1.44},
                    {21, 1.62},
                    {22, 1.82},
                    {23, 2.01},
                    {24, 2.22},
                    {25, 2.43},
                    {26, 2.64},
                    {27, 2.86},
                    {28, 3.08},
                    {29, 3.31},
                    {30, 3.54},
                    //The rest are from table 3.1 of Computing the Action of the Matrix Exponential.
                    {35, 4.7},
                    {40, 6.0},
                    {45, 7.2},
                    {50, 8.5},
                    {55, 9.9},
            };

        public:
            virtual ~BeagleCPUActionImpl();

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

//            virtual int setCategoryRates(const double* inCategoryRates);

            virtual int setTipPartials(int tipIndex,
                                       const double* inPartials);

            virtual int updatePartials(const int *operations,
                                       int operationCount,
                                       int cumulativeScalingIndex);

            virtual int updatePrePartials(const int *operations,
                                       int operationCount,
                                       int cumulativeScalingIndex);
//        protected:
//            virtual int getPaddedPatternsModulus();

        private:
            virtual void rescalePartials(MapType *destP,
                                         double *scaleFactors,
                                         double *cumulativeScaleFactors,
                                         const int  fillWithOnes);


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

            void simpleAction2(MapType *destP, MapType *partials, int edgeIndex,
                               bool transpose);


            void simpleAction(MapType* destP,
                              MapType* partials,
                              SpMatrix* matrix);

            void calcPartialsPartials(MapType* destP,
                                      MapType* partials1,
                                      SpMatrix* matrices1,
                                      MapType* partials2,
                                      SpMatrix* matrices2);

            void calcPartialsPartials2(MapType *destP, MapType *partials1,
                                       MapType *partials2, int edgeIndex1,
                                       int edgeIndex2, MapType *partialCache1,
                                       MapType *partialCache2);

            void calcPrePartialsPartials(MapType *destP,
                                         MapType *partials1,
                                         SpMatrix *matrices1,
                                         MapType *partials2,
                                         SpMatrix *matrices2);

            void calcPrePartialsPartials2(MapType *destP, MapType *partials1,
                                          MapType *partials2, int edgeIndex1,
                                          int edgeIndex2, MapType *partialCache2);

	    // Return (m,s)
	    std::tuple<int,int> getStatistics(double A1Norm,
					      SpMatrix * matrix,
					      double t,
					      int nCol);

	    // Return (m,s)
	    std::tuple<int,int> getStatistics2(double t, int nCol, double edgeMultiplier,
					       int eigenIndex);

            double getDValue(int p,
                             std::map<int, double> &d,
                             std::map<int, SpMatrix> &powerMatrices);

            double getDValue2(int p, int eigenIndex);

            double normP1(SpMatrix * matrix);

            double normPInf(SpMatrix* matrix);
            double normPInf(MapType matrix);
            double normPInf(MatrixXd * matrix);


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
