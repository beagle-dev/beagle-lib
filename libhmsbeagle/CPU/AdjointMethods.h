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
class AdjointMethods {

    const REALTYPE* eval;
    const int       stateCount;
    const bool      allReal;

    // Per-index precomputed values (indexed by base eigenvalue position)
    std::vector<REALTYPE> gexpat;        // exp(t * evalR[i])
    std::vector<REALTYPE> gcosbt;        // cos(t * evalI[i])   -- complex indices only
    std::vector<REALTYPE> gsinbt;        // sin(t * evalI[i])   -- complex indices only
    std::vector<REALTYPE> gexpatcosbt;   // expat[i] * cosbt[i] -- complex indices only
    std::vector<REALTYPE> gexpatsinbt;   // expat[i] * sinbt[i] -- complex indices only

    REALTYPE *expat, *cosbt, *sinbt, *expatcosbt, *expatsinbt;

    REALTYPE time;

#ifdef OPT5
    // Pre-classified eigenvalue base indices (opt 5: branch-free inner loops)
    std::vector<int> realIdx;
    std::vector<int> complexIdx;
#endif

public:
    AdjointMethods(const REALTYPE* eval, const int stateCount, const bool onlyReal)
        : eval(eval), stateCount(stateCount),
          gexpat(stateCount), allReal(onlyReal || !containsImagineryValues(eval, stateCount)),
          gcosbt(allReal ? 0 : stateCount), gsinbt(allReal ? 0 : stateCount),
          gexpatcosbt(allReal ? 0 : stateCount), gexpatsinbt(allReal ? 0 : stateCount),
          expat(gexpat.data()), cosbt(gcosbt.data()), sinbt(gsinbt.data()),
          expatcosbt(gexpatcosbt.data()), expatsinbt(gexpatsinbt.data()) {

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
            const int       matrixCount,
            const C& collector,
            const REALTYPE  scale,
            REALTYPE*       outRateGradient,
            const int       S);

    inline void setTime(const REALTYPE inTime,
            REALTYPE* inExpat,
            REALTYPE* inCosbt, REALTYPE* inSinbt,
            REALTYPE* inExpatcosbt, REALTYPE* inExpatsinbt) {
        time = inTime;
        expat = inExpat;
        cosbt = inCosbt;
        sinbt = inSinbt;
        expatcosbt = inExpatcosbt;
        expatsinbt = inExpatsinbt;
    }

    inline void setTime(const REALTYPE inTime) {
        time = inTime;
        for (int i = 0; i < stateCount; ++i) {
            expat[i] = std::exp(time * eval[i]);
        }
        if (!allReal) {
            for (int i = 0; i < stateCount; ++i) {
                const REALTYPE imag = eval[stateCount + i];
                if (!isReal(imag)) {
                    const REALTYPE c = std::cos(time * imag);
                    const REALTYPE s = std::sin(time * imag);
                    cosbt[i]      = c;
                    sinbt[i]      = s;
                    expatcosbt[i] = expat[i] * c;
                    expatsinbt[i] = expat[i] * s;
                    ++i;
                }
            }
        }
    }

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
            REALTYPE* eigenBasisGrad, const int S);

    inline void computeOneByTwoBlock(
            const int ls, const int rs,
            const REALTYPE lv1, const REALTYPE rv1, const REALTYPE rv2,
            const REALTYPE scale,
            REALTYPE* eigenBasisGrad, const int S);

    inline void computeTwoByOneBlock(
            const int ls, const int rs,
            const REALTYPE lv1, const REALTYPE lv2, const REALTYPE rv1,
            const REALTYPE scale,
            REALTYPE* eigenBasisGrad, const int S);

    inline void computeTwoByTwoBlock(
            const int ls, const int rs,
            const REALTYPE lv1, const REALTYPE lv2,
            const REALTYPE rv1, const REALTYPE rv2,
            const REALTYPE sc,
            REALTYPE* eigenBasisGrad, const int S);

    template <typename C>
    inline void computeOneByOneBlock(
            const C& collector,
            const int ls, const int rs,
            const REALTYPE scale,
            REALTYPE* eigenBasisGrad, const int S);

    template <typename C>
    inline void computeOneByTwoBlock(
            const C& collector,
            const int ls, const int rs,
            const REALTYPE scale,
            REALTYPE* eigenBasisGrad, const int S);

    template <typename C>
    inline void computeTwoByOneBlock(
            const C& collector,
            const int ls, const int rs,
            const REALTYPE scale,
            REALTYPE* eigenBasisGrad, const int S);

    template <typename C>
    inline void computeTwoByTwoBlock(
            const C& collector,
            const int ls, const int rs,
            const REALTYPE sc,
            REALTYPE* eigenBasisGrad, const int S);


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
struct SimpleCollector {

    REALTYPE* gradient;
    const int stateCount;
    const int stride;


    const REALTYPE* lhs;
    const REALTYPE* rhs;
    REALTYPE scale;

    AdjointMethods<REALTYPE>* plan;

    SimpleCollector(REALTYPE* gradient, const int stateCount, const int stride)
            : gradient(gradient), stateCount(stateCount), stride(stride) { }

    inline void accumulateScaledOuterProducts(const REALTYPE* lhs, const REALTYPE* rhs, const REALTYPE scale) {
        this->lhs = lhs;
        this->rhs = rhs;
        this->scale = scale;

        plan->accumulateEigenBasisGradient(1, *this, REALTYPE(1), gradient, stateCount);
    }

    inline void accumulateEigenBasisGradient() {
        // Handled in accumulateScaledOuterProducts
    }

    inline const REALTYPE get(int i, int j) const {
        return lhs[i] * rhs[j] * scale;
    }

    inline void setPlan(AdjointMethods<REALTYPE>* plan) {
        this->plan = plan;
    }

    inline AdjointMethods<REALTYPE>* getPlan() {
        return plan;
    }

    inline void flush() { }
};

template <typename REALTYPE>
struct MultipleCollector {

    REALTYPE* gradient;
    REALTYPE* buffer;
    const int stateCount;
    const int stride;

    AdjointMethods<REALTYPE>* plan;

    MultipleCollector(REALTYPE* gradient, REALTYPE *buffer, const int stateCount, const int stride)
            : gradient(gradient), buffer(buffer), stateCount(stateCount), stride(stride) { }

    inline void flush() {
        std::fill(buffer, buffer + stride * stateCount, REALTYPE(0));
    }

    inline void accumulateScaledOuterProducts(const REALTYPE* lhs, const REALTYPE* rhs, const REALTYPE scale) {
        for (int i = 0; i < stateCount; ++i) {
            const int row_i = i * stride;
            const REALTYPE scaledLeft = lhs[i] * scale;
            for (int j = 0; j < stateCount; ++j) { // TODO ensure this is vectorized or manually unroll
                buffer[row_i + j] += scaledLeft * rhs[j];
                // buffer[row_i + j] = scaledLeft * rhs[j];
            }
        }
    }

    inline const REALTYPE get(int i, int j) const {
        return buffer[i * stride + j];
    }

    inline void setPlan(AdjointMethods<REALTYPE>* plan) {
        this->plan = plan;
    }

    inline AdjointMethods<REALTYPE>* getPlan() {
        return plan;
    }

    inline void accumulateEigenBasisGradient() {
        plan->accumulateEigenBasisGradient(1, *this, REALTYPE(1), gradient, stateCount);
    }
};

} // namespace cpu
} // namespace beagle

#include "libhmsbeagle/CPU/AdjointMethods.hpp"

#endif /* ADJOINTMETHODS_H_ */
