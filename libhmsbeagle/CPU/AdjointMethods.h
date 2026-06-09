/*
 * AdjointMethods.h
 *
 *  Created on: May 29, 2026
 *      Author: msuchard
 */

#ifndef ADJOINTMETHODS_H_
#define ADJOINTMETHODS_H_

#include <cmath>
#include <vector>

#define BEAGLE_CPU_ADJOINT3_TEMPLATE template <typename REALTYPE>
#define BEAGLE_CPU_ADJOINT3_GENERIC  REALTYPE

#define OPT5 // make one / two indices

namespace beagle {
namespace cpu {

BEAGLE_CPU_ADJOINT3_TEMPLATE
class AdjointIntegralPlan {

    const REALTYPE* eval;
    const int       stateCount;
    const bool      allReal;

    const REALTYPE  *expat, *cosbt, *sinbt, *expatcosbt, *expatsinbt;
    REALTYPE time;

#ifdef OPT5
    // Pre-classified eigenvalue base indices (opt 5: branch-free inner loops)
    std::vector<int> realIdx;
    std::vector<int> complexIdx;
#endif

public:
    AdjointIntegralPlan(const REALTYPE* eval, const int stateCount, const bool onlyReal)
        : eval(eval), stateCount(stateCount),
          allReal(onlyReal || !containsImagineryValues(eval, stateCount)) {

#ifdef OPT5
        for (int i = 0; i < stateCount; ) {
            const REALTYPE imag = eval[stateCount + i];
            if (isReal(imag)) {
                realIdx.push_back(i);
                ++i;
            } else {
                complexIdx.push_back(i);
                i += 2;
            }
        }
#endif
    }

    inline REALTYPE branchLikelihoodInEigenBasis(
            const REALTYPE* lhs,
            const REALTYPE* rhs);

    inline int accumulateEigenBasisGradient(
            const int       matrixCount,
            const REALTYPE* lhs,
            const REALTYPE* rhs,
            const REALTYPE  scale,
            REALTYPE*       outRateGradient,
            const int       S);

    template <typename C>
    inline int accumulateEigenBasisGradient(
            const C& collector,
            REALTYPE* outRateGradient,
            const int stride);

    inline static bool isReal(const REALTYPE value) {
        return value == static_cast<REALTYPE>(0);
    }

    inline static bool closeToZero(const REALTYPE value) {
        return value < static_cast<REALTYPE>(1e-12);
    }

private:
    inline void computeOneByOneBlock(
            const int ls, const int rs,
            const REALTYPE lv, const REALTYPE rv,
            const REALTYPE scale,
            REALTYPE* __restrict__ eigenBasisGrad, const int S);

    inline void computeOneByTwoBlock(
            const int ls, const int rs,
            const REALTYPE lv1, const REALTYPE rv1, const REALTYPE rv2,
            const REALTYPE scale,
            REALTYPE* __restrict__ eigenBasisGrad, const int S);

    inline void computeTwoByOneBlock(
            const int ls, const int rs,
            const REALTYPE lv1, const REALTYPE lv2, const REALTYPE rv1,
            const REALTYPE scale,
            REALTYPE* __restrict__ eigenBasisGrad, const int S);

    inline void computeTwoByTwoBlock(
            const int ls, const int rs,
            const REALTYPE lv1, const REALTYPE lv2,
            const REALTYPE rv1, const REALTYPE rv2,
            const REALTYPE sc,
            REALTYPE* __restrict__ eigenBasisGrad, const int S);

    template <typename C>
    inline void computeOneByOneBlock(
            const C& collector,
            const int ls, const int rs,
            REALTYPE* __restrict__ eigenBasisGrad, const int S);

    template <typename C>
    inline void computeOneByTwoBlock(
            const C& collector,
            const int ls, const int rs,
            REALTYPE* __restrict__ eigenBasisGrad, const int S);

    template <typename C>
    inline void computeTwoByOneBlock(
            const C& collector,
            const int ls, const int rs,
            REALTYPE* __restrict__ eigenBasisGrad, const int S);

    template <typename C>
    inline void computeTwoByTwoBlock(
            const C& collector,
            const int ls, const int rs,
            REALTYPE* __restrict__ eigenBasisGrad, const int S);

    static bool containsImagineryValues(const REALTYPE* eval, const int stateCount) {
        for (int i = 0; i < stateCount; ++i) {
            if (!isReal(eval[stateCount + i])) {
                return true;
            }
        }
        return false;
    }
};

template <typename REALTYPE>
struct CollectorBase {

    REALTYPE* gradient;
    const int stateCount;
    const int stride;

    AdjointIntegralPlan<REALTYPE>* plan;

    REALTYPE time;
    const REALTYPE* expat;
    const REALTYPE* cosbt;
    const REALTYPE* sinbt;
    const REALTYPE* expatcosbt;
    const REALTYPE* expatsinbt;

    CollectorBase(REALTYPE* gradient, const int stateCount, const int stride)
            : gradient(gradient), stateCount(stateCount), stride(stride) { }

    inline void setTime(const REALTYPE t, const REALTYPE* inExpat,
            const REALTYPE* inCosbt, const REALTYPE* inSinbt,
            const REALTYPE* inExpatcosbt, const REALTYPE* inExpatsinbt) {
        time = t;
        expat = inExpat;
        cosbt = inCosbt;
        sinbt = inSinbt;
        expatcosbt = inExpatcosbt;
        expatsinbt = inExpatsinbt;
    }

    inline void setPlan(AdjointIntegralPlan<REALTYPE>* inPlan) {
        plan = inPlan;
    }
};

template <typename REALTYPE>
struct SimpleCollector : CollectorBase<REALTYPE> {
    using Base = CollectorBase<REALTYPE>;
    using Base::gradient;
    using Base::stateCount;
    using Base::stride;
    using Base::plan;

    const REALTYPE* lhs;
    const REALTYPE* rhs;
    REALTYPE scale;

    SimpleCollector(REALTYPE* gradient, const int stateCount, const int stride)
            : Base(gradient, stateCount, stride) { }

    inline void accumulateScaledOuterProducts(const REALTYPE* lhs, const REALTYPE* rhs, const REALTYPE scale) {
        this->lhs = lhs;
        this->rhs = rhs;
        this->scale = scale;

        plan->accumulateEigenBasisGradient(*this, gradient, stateCount);
    }

    inline void accumulateEigenBasisGradient() {
        // Handled in accumulateScaledOuterProducts
    }

    inline const REALTYPE get(int i, int j) const {
        return lhs[i] * rhs[j] * scale;
    }

    inline void flush() { }
};

template <typename REALTYPE>
struct MultipleCollector : CollectorBase<REALTYPE> {
    using Base = CollectorBase<REALTYPE>;
    using Base::gradient;
    using Base::stateCount;
    using Base::stride;
    using Base::plan;

    REALTYPE* buffer;

    MultipleCollector(REALTYPE* gradient, REALTYPE *buffer, const int stateCount, const int stride)
            : Base(gradient, stateCount, stride), buffer(buffer) { }

    inline void flush() {
        std::fill(buffer, buffer + stride * stateCount, REALTYPE(0));
    }

    inline void accumulateScaledOuterProducts(
            const REALTYPE*  __restrict__ lhs,
            const REALTYPE*  __restrict__ rhs,
            const REALTYPE scale) {
        for (int i = 0; i < stateCount; ++i) {
            const int row_i = i * stride;
            const REALTYPE scaledLeft = lhs[i] * scale;

            #pragma clang loop vectorize(enable)
            for (int j = 0; j < stateCount; ++j) {
                buffer[row_i + j] += scaledLeft * rhs[j];
            }
        }
    }

    inline const REALTYPE get(int i, int j) const {
        return buffer[i * stride + j];
    }

    inline void accumulateEigenBasisGradient() {
        plan->accumulateEigenBasisGradient(*this, gradient, stateCount);
    }
};

} // namespace cpu
} // namespace beagle

#include "libhmsbeagle/CPU/AdjointMethods.hpp"

#endif /* ADJOINTMETHODS_H_ */
