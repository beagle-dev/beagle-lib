/*
 * AdjointMethods.hpp
 *
 *  Created on: May 29, 2026
 *      Author: msuchard
 */

#ifndef ADJOINTMETHODS_HPP_
#define ADJOINTMETHODS_HPP_

#include "AdjointMethods.h"

namespace beagle {
namespace cpu {

// branchLikelihoodInEigenBasis: uses expcosbt/expsinbt instead of per-call multiply
BEAGLE_CPU_ADJOINT3_TEMPLATE
REALTYPE AdjointMethods<BEAGLE_CPU_ADJOINT3_GENERIC>::branchLikelihoodInEigenBasis(
        const REALTYPE* lhs,
        const REALTYPE* rhs) {

    REALTYPE sum = static_cast<REALTYPE>(0);
    if (allReal) {
        for (int i = 0; i < stateCount; ++i) {
            sum += lhs[i] * expat[i] * rhs[i];
        }
    } else {
        for (int i = 0; i < stateCount; ++i) {
            const REALTYPE imag = eval[stateCount + i];
            if (isReal(imag)) {
                sum += lhs[i] * expat[i] * rhs[i];
            } else {
                const REALTYPE c = expatcosbt[i];  // opt 3: was expat[i] * cosbt[i]
                const REALTYPE s = expatsinbt[i];  // opt 3: was expat[i] * sinbt[i]

                const REALTYPE x = rhs[i];
                const REALTYPE y = rhs[i + 1];

                sum += lhs[i] * (c * x + s * y) + lhs[i + 1] * (c * y - s * x);

                ++i;
            }
        }
    }

    return sum;
}

// computeOneByOneBlock: unchanged from AdjointMethods2 — already uses precomputed expat
BEAGLE_CPU_ADJOINT3_TEMPLATE
void AdjointMethods<BEAGLE_CPU_ADJOINT3_GENERIC>::computeOneByOneBlock(
        const int ls, const int rs,
        const REALTYPE lv, const REALTYPE rv,
        const REALTYPE scale,
        REALTYPE* eigenBasisGrad,
        const int S) {

    const REALTYPE la    = eval[ls];
    const REALTYPE lb    = eval[rs];
    const REALTYPE coeff = (time * std::abs(la - lb) < 1e-12)
            ? time * expat[ls]
            : (expat[ls] - expat[rs]) / (la - lb);

    eigenBasisGrad[ls * S + rs] += lv * rv * scale * coeff;
}

// computeOneByTwoBlock: opt 1 — exp(t*sr) → expat[rs]/expat[ls]
BEAGLE_CPU_ADJOINT3_TEMPLATE
void AdjointMethods<BEAGLE_CPU_ADJOINT3_GENERIC>::computeOneByTwoBlock(
        const int ls, const int rs,
        const REALTYPE lv1, const REALTYPE rv1, const REALTYPE rv2,
        const REALTYPE sc,
        REALTYPE* eigenBasisGrad, const int S) {

    const REALTYPE la  = eval[ls];
    const REALTYPE rr  = eval[rs];
    const REALTYPE ri  = eval[stateCount + rs];
    const REALTYPE sr  = rr - la;
    const REALTYPE den = sr * sr + ri * ri;

    REALTYPE ic0, ic1;
    if (den < 1e-12) {
        ic0 = time;
        ic1 = 0.0;
    } else {
        const REALTYPE ex = expat[rs] / expat[ls];  // opt 1: was std::exp(t * sr)
        const REALTYPE cs = cosbt[rs];
        const REALTYPE sn = sinbt[rs];
        ic0 = (ex * (sr * cs + ri * sn) - sr) / den;
        ic1 = (ex * (sr * sn - ri * cs) + ri) / den;
    }

    const REALTYPE c0  = expat[ls] * ic0;
    const REALTYPE c1  = expat[ls] * ic1;
    const REALTYPE in0 = lv1 * rv1 * sc;
    const REALTYPE in1 = lv1 * rv2 * sc;

    eigenBasisGrad[ls * S + rs]     +=  c0 * in0 + c1 * in1;
    eigenBasisGrad[ls * S + rs + 1] += -c1 * in0 + c0 * in1;
}

// computeTwoByOneBlock: opt 1 — exp(t*sr) → expat[rs]/expat[ls]; opt 3 — expcosbt/expsinbt
BEAGLE_CPU_ADJOINT3_TEMPLATE
void AdjointMethods<BEAGLE_CPU_ADJOINT3_GENERIC>::computeTwoByOneBlock(
        const int ls, const int rs,
        const REALTYPE lv1, const REALTYPE lv2, const REALTYPE rv1,
        const REALTYPE sc,
        REALTYPE* eigenBasisGrad, const int S) {

    const REALTYPE lr  = eval[ls];
    const REALTYPE li  = eval[stateCount + ls];
    const REALTYPE rb  = eval[rs];
    const REALTYPE sr  = rb - lr;
    const REALTYPE den = sr * sr + li * li;

    const REALTYPE cI = cosbt[ls];
    const REALTYPE sI = sinbt[ls];

    REALTYPE ic0, ic1;
    if (den < 1e-12) {
        ic0 = time;
        ic1 = 0.0;
    } else {
        const REALTYPE ex = expat[rs] / expat[ls];  // opt 1: was std::exp(t * sr)
        ic0 = (ex * (sr * cI + li * sI) - sr) / den;
        ic1 = (ex * (sr * sI - li * cI) + li) / den;
    }

    // opt 3: was eR*cI, eR*(-sI), eR*sI, eR*cI with eR=expat[ls]
    const REALTYPE ec = expatcosbt[ls];  // eR *  cI
    const REALTYPE es = expatsinbt[ls];  // eR *  sI

    const REALTYPE p0 =  ec * ic0 + es * ic1;
    const REALTYPE p1 =  ec * ic1 - es * ic0;
    const REALTYPE p2 =  es * ic0 - ec * ic1;
    const REALTYPE p3 =  es * ic1 + ec * ic0;

    const REALTYPE in0 = lv1 * rv1 * sc;
    const REALTYPE in1 = lv2 * rv1 * sc;

    eigenBasisGrad[ls * S + rs]       += p0 * in0 + p1 * in1;
    eigenBasisGrad[(ls + 1) * S + rs] += p2 * in0 + p3 * in1;
}

// computeTwoByTwoBlock: opt 1 — exp(t*sr) → expat[rs]/expat[ls]
//            opt 2 — 4 cos/sin calls → angle-addition using cosbt/sinbt
//            opt 3 — expcosbt/expsinbt for er/ei
BEAGLE_CPU_ADJOINT3_TEMPLATE
void AdjointMethods<BEAGLE_CPU_ADJOINT3_GENERIC>::computeTwoByTwoBlock(
        const int ls, const int rs,
        const REALTYPE lv1, const REALTYPE lv2,
        const REALTYPE rv1, const REALTYPE rv2,
        const REALTYPE sc,
        REALTYPE* eigenBasisGrad, const int S) {

    const REALTYPE lr = eval[ls];
    const REALTYPE li = eval[stateCount + ls];
    const REALTYPE rr = eval[rs];
    const REALTYPE ri = eval[stateCount + rs];

    const REALTYPE sr  = rr - lr;
    const REALTYPE si1 = li + ri;
    const REALTYPE si2 = ri - li;

    const REALTYPE er = expatcosbt[ls];  // opt 3: was expat[ls] * cosbt[ls]
    const REALTYPE ei = expatsinbt[ls];  // opt 3: was expat[ls] * sinbt[ls]

    const REALTYPE sr2 = sr * sr;
    const REALTYPE d1  = sr2 + si1 * si1;
    const REALTYPE d2  = sr2 + si2 * si2;

    const REALTYPE ex = (d1 >= 1e-12 || d2 >= 1e-12) ? expat[rs] / expat[ls] : 0.0;  // opt 1

    // opt 2: precompute shared products for angle-addition formulas
    // cos(t*(li+ri)) = cos(t*li)*cos(t*ri) - sin(t*li)*sin(t*ri)
    // sin(t*(li+ri)) = sin(t*li)*cos(t*ri) + cos(t*li)*sin(t*ri)
    // cos(t*(ri-li)) = cos(t*li)*cos(t*ri) + sin(t*li)*sin(t*ri)
    // sin(t*(ri-li)) = cos(t*li)*sin(t*ri) - sin(t*li)*cos(t*ri)
    const REALTYPE clcr = cosbt[ls] * cosbt[rs];
    const REALTYPE slsr = sinbt[ls] * sinbt[rs];
    const REALTYPE clsr = cosbt[ls] * sinbt[rs];
    const REALTYPE slcr = sinbt[ls] * cosbt[rs];

    REALTYPE int1r, int1i;
    if (d1 < 1e-12) {
        int1r = time;
        int1i = 0.0;
    } else {
        const REALTYPE cs1       = clcr - slsr;        // opt 2: was std::cos(t * si1)
        const REALTYPE sn1       = slcr + clsr;        // opt 2: was std::sin(t * si1)
        const REALTYPE ex_cs1_m1 = ex * cs1 - 1.0;
        const REALTYPE ex_sn1    = ex * sn1;
        int1r = (sr * ex_cs1_m1 + si1 * ex_sn1) / d1;
        int1i = (sr * ex_sn1    - si1 * ex_cs1_m1) / d1;
    }

    REALTYPE int2r, int2i;
    if (d2 < 1e-12) {
        int2r = time;
        int2i = 0.0;
    } else {
        const REALTYPE cs2       = clcr + slsr;        // opt 2: was std::cos(t * si2)
        const REALTYPE sn2       = clsr - slcr;        // opt 2: was std::sin(t * si2)
        const REALTYPE ex_cs2_m1 = ex * cs2 - 1.0;
        const REALTYPE ex_sn2    = ex * sn2;
        int2r = (sr * ex_cs2_m1 + si2 * ex_sn2) / d2;
        int2i = (sr * ex_sn2    - si2 * ex_cs2_m1) / d2;
    }

    const REALTYPE pr  = er * int1r + ei * int1i;
    const REALTYPE pi_ = er * int1i - ei * int1r;
    const REALTYPE mr_ = er * int2r - ei * int2i;
    const REALTYPE mi_ = er * int2i + ei * int2r;

    const REALTYPE A = 0.5 * (mr_ + pr);
    const REALTYPE B = 0.5 * (mi_ + pi_);
    const REALTYPE C = 0.5 * (pi_ - mi_);
    const REALTYPE D = 0.5 * (mr_ - pr);

    const REALTYPE in00 = lv1 * rv1 * sc;
    const REALTYPE in01 = lv1 * rv2 * sc;
    const REALTYPE in10 = lv2 * rv1 * sc;
    const REALTYPE in11 = lv2 * rv2 * sc;

    eigenBasisGrad[ls * S + rs]           +=  A*in00 + B*in01 + C*in10 + D*in11;
    eigenBasisGrad[ls * S + rs + 1]       += -B*in00 + A*in01 - D*in10 + C*in11;
    eigenBasisGrad[(ls + 1) * S + rs]     += -C*in00 - D*in01 + A*in10 + B*in11;
    eigenBasisGrad[(ls + 1) * S + rs + 1] +=  D*in00 - C*in01 - B*in10 + A*in11;
}

#ifdef OPT5
// accumulateEigenBasisGradient: opt 5 — pre-classified indices eliminate inner branches
BEAGLE_CPU_ADJOINT3_TEMPLATE
int AdjointMethods<BEAGLE_CPU_ADJOINT3_GENERIC>::accumulateEigenBasisGradient(
        const int       M,
        const REALTYPE* lhs,
        const REALTYPE* rhs,
        const REALTYPE  scale,
        REALTYPE*       outRateGradient,
        const int       S) {

    REALTYPE* eigenBasisGrad = outRateGradient;

    for (int m = 0; m < M; m++) {

        if (allReal) {

            for (int ls = 0; ls < stateCount; ++ls) {
                for (int rs = 0; rs < stateCount; ++rs) {
                    computeOneByOneBlock(ls, rs, lhs[ls], rhs[rs], scale, eigenBasisGrad, S);
                }
            }

        } else {

            // OneOne: real × real
            for (int li : realIdx) {
                const REALTYPE lv1 = lhs[li];
                for (int ri : realIdx) {
                    computeOneByOneBlock(li, ri, lv1, rhs[ri], scale, eigenBasisGrad, S);
                }
            }

            // OneTwo: real × complex
            for (int li : realIdx) {
                const REALTYPE lv1 = lhs[li];
                for (int ri : complexIdx) {
                    computeOneByTwoBlock(li, ri, lv1, rhs[ri], rhs[ri + 1], scale, eigenBasisGrad, S);
                }
            }

            // TwoOne: complex × real
            for (int li : complexIdx) {
                const REALTYPE lv1 = lhs[li];
                const REALTYPE lv2 = lhs[li + 1];
                for (int ri : realIdx) {
                    computeTwoByOneBlock(li, ri, lv1, lv2, rhs[ri], scale, eigenBasisGrad, S);
                }
            }

            // TwoTwo: complex × complex
            for (int li : complexIdx) {
                const REALTYPE lv1 = lhs[li];
                const REALTYPE lv2 = lhs[li + 1];
                for (int ri : complexIdx) {
                    computeTwoByTwoBlock(li, ri, lv1, lv2, rhs[ri], rhs[ri + 1], scale, eigenBasisGrad, S);
                }
            }
        }
    }

    return 0;
}
#else
BEAGLE_CPU_ADJOINT3_TEMPLATE
int AdjointMethods<BEAGLE_CPU_ADJOINT3_GENERIC>::accumulateEigenBasisGradient(
        const int       M,
        const REALTYPE* lhs,
        const REALTYPE* rhs,
        const REALTYPE  scale,
        REALTYPE*       outRateGradient,
        const int       S) {

    REALTYPE* eigenBasisGrad = outRateGradient;

    for (int m = 0; m < M; m++) {

       if (allReal) {

            for (int ls = 0; ls < stateCount; ++ls) {
                for (int rs = 0; rs < stateCount; ++rs) {
                    computeOneByOneBlock(ls, rs, lhs[ls], rhs[rs], scale, eigenBasisGrad, S);
                }
            }

       } else {

            for (int ls = 0; ls < S; ++ls) {
                const REALTYPE lv1 = lhs[ls];

                if (eval[stateCount + ls] == static_cast<REALTYPE>(0.0)) {
                    for (int rs = 0; rs < S; ++rs) {
                        const REALTYPE rv1 = rhs[rs];

                        if (eval[stateCount + rs] == static_cast<REALTYPE>(0.0)) {

                            computeOneByOneBlock(ls, rs, lv1, rv1, scale, eigenBasisGrad, S);

                        } else {

                            const REALTYPE rv2 = rhs[rs + 1];
                            computeOneByTwoBlock(ls, rs, lv1, rv1, rv2, scale, eigenBasisGrad, S);

                            ++rs;
                        }
                    }
                } else {
                    const REALTYPE lv2 = lhs[ls + 1];

                    for (int rs = 0; rs < S; ++rs) {
                        const REALTYPE rv1 = rhs[rs];

                        if (eval[stateCount + rs] == static_cast<REALTYPE>(0.0)) {

                            computeTwoByOneBlock(ls, rs, lv1, lv2, rv1, scale, eigenBasisGrad, S);

                        } else {

                            const REALTYPE rv2 = rhs[rs + 1];
                            computeTwoByTwoBlock(ls, rs, lv1, lv2, rv1, rv2, scale, eigenBasisGrad, S);

                            ++rs;
                        }
                    }

                    ++ls;
                }
            }
       }
    }
    return 0;
}
#endif

} // namespace cpu
} // namespace beagle

#endif /* ADJOINTMETHODS_HPP_ */
