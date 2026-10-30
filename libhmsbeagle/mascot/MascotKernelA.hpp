/*
 * MascotKernelA.hpp -- MASCOT kernel A: runtime K, portable C++ (no intrinsics), lineage-SIMD layout.
 *
 * What it computes. The MASCOT structured-coalescent likelihood and its adjoint, operation for operation as BEAST's
 * GenericMascotLikelihoodDelegate (MascotForwardEvaluator / MascotAdjointEvaluator):
 *  - forward: walk the schedule; split each interval at epoch boundaries and integrate every segment with fixed-step
 *    RK4 (steps = max(1, ceil(len / maxStep))), projecting each lineage back onto the simplex after every step; the
 *    log-likelihood integrates -hazard; a coalescence adds log(lambda) and replaces the two children by the
 *    normalized parent. With problem.checkProbabilities, every active lineage is checked after each segment and
 *    after each event, as MascotForwardEvaluator.checkProbabilities.
 *  - forwardAndReverse: the same forward pass recording a tape (per step: the four RK4 stage inputs, the post-step
 *    state, the normalization sums and the per-stage deme sums / sums of squares), then the reverse pass: VJP of the
 *    normalization, the four stage VJPs in reverse, and the coalescent VJPs. Gradients with respect to M and Q are
 *    mapped to the theta layout (per epoch: K(K-1) off-diagonal migration rates in row-major order, then K log Ne),
 *    the clock gradient is accumulated per lineage, and each coalescent parent's ancestral score row is written.
 *
 * Inputs. M (row = source, diagonal = -row sum) and Q = 1 / Ne come from the caller (Java) and are copied as given;
 * no exp() is evaluated here.
 *
 * Layout. Between tree events the state is slot-major (y[slot * K + s]), because sample and coalescent events address
 * single lineages. Inside a segment (constant active count A) it is transposed to deme-major X[s * LD + l], with the
 * lineage index l contiguous and LD = A rounded up to even. Every elementwise RK4 / adjoint operation is then
 * vectorized over lineages in fixed lane blocks of 8 / 4 / 2 (trip count independent of K), and the K x K products are
 * register-tiled small GEMMs (a fixed tile of sinks x lanes in registers, the runtime K loop inside). Padded lanes
 * l >= A hold a copy of lineage 0 (finite and normalized) and are never read by a cross-lineage reduction. Nothing
 * on the stack is sized by K.
 *
 * Floating point. Every output element is produced with the reference's operation sequence: reductions over lineages
 * run in lineage order, reductions over demes in deme order, GEMM accumulators start from the reference's first term
 * and accumulate in source (or sink) order. In the strict build the theta gradient, clock gradient and ancestral
 * scores therefore equal Java's bit for bit (logL up to the platform log()).
 *
 * FP modes. MascotKernelA.cpp includes this header in two translation units, both compiled with -ffp-contract=off:
 * strict (namespace mascot_a_strict) and FMA (-DMASCOT_FP_FMA, namespace mascot_a_fma). The distinct namespaces keep
 * the linker from merging inline functions compiled for different FP modes. Multiply-adds written MASCOT_MUL_ADD
 * (MascotFma.hpp) are fused in the FMA build and rounded twice in the strict build; each is written in the
 * reference's evaluation order (x + a * b as MASCOT_MUL_ADD(a, b, x)), so the strict build is unchanged.
 *  - Fused: the K x K products (forward, adjoint, gram), the coalescent term r, the stage derivative k and the stage
 *    inputs cur + coef * k, the adjoint accumulations (r, b, clock term, out, the adjoint combination, the RK4 adjoint
 *    inputs) and the projection dot products.
 *  - Not fused: the lineage-ordered scalar reductions (per-deme sums of squares, cSums, gradQ; fusing a serial chain
 *    lengthens it), the O(K) per-step scalar terms, acc + 2k (2k is exact, so fusing changes nothing), and the final
 *    combination cur + h6 * (acc + k): fused, clang's SLP cost model leaves the 4-lane block of stage 4 scalar.
 *
 * Vectorizing std::fma. Clang's SLP vectorizer rates a 2-lane fma as saving only one instruction, so a tree of fmas
 * whose operands need broadcasts or lane gathers looks break-even and stays scalar. The kernel therefore feeds every
 * broadcast operand of a fused loop as a 2-lane load from a copy that stores each value twice: M2 and MT2 (M, M^T),
 * Q2, S2 (the stage's per-deme sums, coef and h6) and cys2 (the gram operand c * y). The adjoint and gram tiles peel
 * their first term as a * b + 0.0, the reference's first step (0 + a * b) in both builds, which also keeps them
 * vectorized. The strict build computes the same values from the same expressions.
 *
 * State. A KernelA object owns its workspace and tape; both grow on demand and are reused, so a warm object does not
 * allocate. There is no static or thread_local mutable state.
 *
 * Code generation notes (clang):
 *  - restrict is spelled __restrict__: macOS <sys/cdefs.h> defines __restrict to nothing in C++.
 *  - MASCOT_A_NOVEC keeps the loop vectorizer off an outer runtime-K loop whose body is a fixed-width lane loop, so
 *    the lane loop is unrolled and SLP-vectorized instead of the K loop being vectorized with in-order reductions.
 *  - The free functions are static inline (internal linkage), which lets the inliner fold the single-call-site
 *    lane-block kernels completely; plain inline measured 3-5% slower at K = 5 and 8.
 */
#ifndef MASCOT_KERNEL_A_HPP
#define MASCOT_KERNEL_A_HPP

#include "MascotFma.hpp"
#include "MascotProblem.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define MASCOT_A_RESTRICT __restrict__
#else
#define MASCOT_A_RESTRICT __restrict
#endif

#if defined(__clang__)
#define MASCOT_A_NOVEC _Pragma("clang loop vectorize(disable) interleave(disable)")
#else
#define MASCOT_A_NOVEC
#endif

#ifdef MASCOT_FP_FMA
namespace mascot_a_fma {
#else
namespace mascot_a_strict {
#endif

constexpr double TOL = 1.0e-14;   /* MascotRuntime.TIME_TOLERANCE */
constexpr double EPS = 1.0e-300;  /* MascotAdjointEvaluator.EPS   */

static inline int roundEven(int a) { return (a + 1) & ~1; }

template <class V>
static inline double* growTape(V& v, size_t& used, size_t add) {
    if (v.size() < used + add) v.resize(std::max(v.size() * 2, used + add));
    double* p = v.data() + used;
    used += add;
    return p;
}

// ------------------------------------------------------------------------------------------------------------------
// Register-tiled kernels. SB / SR / SN / LB are tile widths (sinks, sources, lanes), not the deme count: the K loops
// inside are runtime loops and the K dimension is covered by a cascade of tiles (4/2/1 or 8/4/2).
// ------------------------------------------------------------------------------------------------------------------

// Forward migration product for one lane block: OUT[sink][j] = sum_source CP[source][j] * M[source][sink], first term
// CP[0][j] * M[0][sink], then sources 1..K-1 in order (as the reference). M2 holds every entry of M twice, so the
// broadcast operand is a 2-lane load (M2[2 * (source * K + sink) + (j & 1)] = M[source][sink]).
template <int SB, int LB>
static inline void fwdTile(int K, const double* MASCOT_A_RESTRICT CP, const double* MASCOT_A_RESTRICT M2, int sb,
                           double* MASCOT_A_RESTRICT OUT) {
    double acc[SB][LB];
    for (int i = 0; i < SB; i++)
        for (int j = 0; j < LB; j++) acc[i][j] = CP[j] * M2[2 * (sb + i) + (j & 1)];
    MASCOT_A_NOVEC for (int src = 1; src < K; src++) {
        const double* MASCOT_A_RESTRICT cp = CP + src * LB;
        const double* MASCOT_A_RESTRICT mr = M2 + ((size_t) src * K + sb) * 2;
        for (int i = 0; i < SB; i++)
            for (int j = 0; j < LB; j++) acc[i][j] = MASCOT_MUL_ADD(cp[j], mr[2 * i + (j & 1)], acc[i][j]);
    }
    for (int i = 0; i < SB; i++)
        for (int j = 0; j < LB; j++) OUT[(sb + i) * LB + j] = acc[i][j];
}

template <int LB>
static inline void fwdGemm(int K, const double* CP, const double* M2, double* OUT) {
    constexpr int SB = LB >= 8 ? 4 : 8;
    int sb = 0;
    for (; sb + SB <= K; sb += SB) fwdTile<SB, LB>(K, CP, M2, sb, OUT);
    if (SB > 4 && sb + 4 <= K) { fwdTile<4, LB>(K, CP, M2, sb, OUT); sb += 4; }
    switch (K - sb) {
        case 3: fwdTile<3, LB>(K, CP, M2, sb, OUT); break;
        case 2: fwdTile<2, LB>(K, CP, M2, sb, OUT); break;
        case 1: fwdTile<1, LB>(K, CP, M2, sb, OUT); break;
        default: break;
    }
}

// Adjoint migration product for one lane block: V[source][j] = 0 + sum_sink AR[sink][j] * M[source][sink], sinks in
// order (as the reference's v = 0; v += adj * M). MT2 is M^T with every entry stored twice, as M2 in fwdTile. The
// first sink is peeled as a * m + 0.0: the reference's first step, and (unlike an all-fma tree) clang's SLP
// vectorizer keeps the tile vectorized.
template <int SB, int LB>
static inline void vjpTile(int K, int LD, const double* MASCOT_A_RESTRICT AR, const double* MASCOT_A_RESTRICT MT2,
                           int sb, double* MASCOT_A_RESTRICT V) {
    double acc[SB][LB];
    for (int i = 0; i < SB; i++)
        for (int j = 0; j < LB; j++) acc[i][j] = AR[j] * MT2[2 * (sb + i) + (j & 1)] + 0.0;
    MASCOT_A_NOVEC for (int snk = 1; snk < K; snk++) {
        const double* MASCOT_A_RESTRICT ar = AR + (size_t) snk * LD;
        const double* MASCOT_A_RESTRICT mt = MT2 + ((size_t) snk * K + sb) * 2;
        for (int i = 0; i < SB; i++)
            for (int j = 0; j < LB; j++) acc[i][j] = MASCOT_MUL_ADD(ar[j], mt[2 * i + (j & 1)], acc[i][j]);
    }
    for (int i = 0; i < SB; i++)
        for (int j = 0; j < LB; j++) V[(sb + i) * LB + j] = acc[i][j];
}

template <int LB>
static inline void vjpGemm(int K, int LD, const double* AR, const double* MT2, double* V) {
    constexpr int SB = LB >= 8 ? 4 : 8;
    int sb = 0;
    for (; sb + SB <= K; sb += SB) vjpTile<SB, LB>(K, LD, AR, MT2, sb, V);
    if (SB > 4 && sb + 4 <= K) { vjpTile<4, LB>(K, LD, AR, MT2, sb, V); sb += 4; }
    switch (K - sb) {
        case 3: vjpTile<3, LB>(K, LD, AR, MT2, sb, V); break;
        case 2: vjpTile<2, LB>(K, LD, AR, MT2, sb, V); break;
        case 1: vjpTile<1, LB>(K, LD, AR, MT2, sb, V); break;
        default: break;
    }
}

// Gram tile: G[src][snk] = 0 + sum_{l < A} CYS[src][l] * ART[l][snk], lineages in order (as gram += cys * adj).
// Vectorized over sinks (ART is the lineage-major transpose of the adjoint, row stride KP = K rounded up to even).
// CYS2 holds every entry of CYS twice (CYS2[2 * (src * LD + l) + (j & 1)]); the first lineage is peeled as in vjpTile.
template <int SR, int SN>
static inline void gramTile(int A, int LD, int KP, const double* MASCOT_A_RESTRICT CYS2,
                            const double* MASCOT_A_RESTRICT ART, int sr, int sn, double* MASCOT_A_RESTRICT G) {
    double acc[SR][SN];
    for (int i = 0; i < SR; i++)
        for (int j = 0; j < SN; j++) acc[i][j] = CYS2[(size_t) (sr + i) * LD * 2 + (j & 1)] * ART[sn + j] + 0.0;
    MASCOT_A_NOVEC for (int l = 1; l < A; l++) {
        const double* MASCOT_A_RESTRICT art = ART + (size_t) l * KP + sn;
        for (int i = 0; i < SR; i++)
            for (int j = 0; j < SN; j++)
                acc[i][j] = MASCOT_MUL_ADD(CYS2[((size_t) (sr + i) * LD + l) * 2 + (j & 1)], art[j], acc[i][j]);
    }
    for (int i = 0; i < SR; i++)
        for (int j = 0; j < SN; j++) G[(size_t) (sr + i) * KP + sn + j] = acc[i][j];
}

template <int SR>
static inline void gramRow(int A, int LD, int KP, const double* CYS2, const double* ART, int sr, double* G) {
    int sn = 0;
    for (; sn + 8 <= KP; sn += 8) gramTile<SR, 8>(A, LD, KP, CYS2, ART, sr, sn, G);
    if (sn + 4 <= KP) { gramTile<SR, 4>(A, LD, KP, CYS2, ART, sr, sn, G); sn += 4; }
    if (sn < KP) gramTile<SR, 2>(A, LD, KP, CYS2, ART, sr, sn, G);
}

static inline void gramAll(int K, int A, int LD, int KP, const double* CYS2, const double* ART, double* G) {
    int sr = 0;
    for (; sr + 4 <= K; sr += 4) gramRow<4>(A, LD, KP, CYS2, ART, sr, G);
    if (sr + 2 <= K) { gramRow<2>(A, LD, KP, CYS2, ART, sr, G); sr += 2; }
    if (sr < K) gramRow<1>(A, LD, KP, CYS2, ART, sr, G);
}

// ------------------------------------------------------------------------------------------------------------------
// Lane-block kernels. All pointers are offset to the block's first lane; restrict-qualified parameters let the SLP
// vectorizer turn each fixed-width lane loop into vector code.
// ------------------------------------------------------------------------------------------------------------------

// ns += y_0 + ... in lane order (only the nreal real lanes); nq likewise with y^2.
template <int LB>
static inline void addLanes(double* MASCOT_A_RESTRICT ns, double* MASCOT_A_RESTRICT nq, const double* yv,
                            const double* ymem, int nreal) {
    double a = *ns, b = *nq;
    if (nreal == LB) {
        for (int j = 0; j < LB; j++) { a += yv[j]; b += yv[j] * yv[j]; }
    } else {
        for (int j = 0; j < nreal; j++) { a += ymem[j]; b += ymem[j] * ymem[j]; }
    }
    *ns = a;
    *nq = b;
}

// One lane block of one RK4 stage: rhs at P, then the stage update.
// STAGE 1-3: acc (+)= k, Y = cur + coef * k.  STAGE 4: Y = normalize(cur + h6 * (acc + k)), sc = 1'(...).
// Also accumulates the next stage input's per-deme sums / sums of squares over the real lanes, in lane order.
// The scalars broadcast over lanes come in pairs, so that each is a 2-lane load: Q2 = Q with every entry twice;
// S2 = the stage input's per-deme sums with every entry twice, then coef, coef, h6, h6.
template <int STAGE, int LB>
static inline void fwdBlock(const int K, const size_t LD, const int nreal, const double* MASCOT_A_RESTRICT P,
                            const double* MASCOT_A_RESTRICT cur, double* MASCOT_A_RESTRICT acc,
                            double* MASCOT_A_RESTRICT Y, double* MASCOT_A_RESTRICT sc,
                            const double* MASCOT_A_RESTRICT clk, const double* MASCOT_A_RESTRICT M2,
                            const double* MASCOT_A_RESTRICT Q2, const double* MASCOT_A_RESTRICT S2,
                            double* MASCOT_A_RESTRICT nsums, double* MASCOT_A_RESTRICT nsq,
                            double* MASCOT_A_RESTRICT CP, double* MASCOT_A_RESTRICT OUT,
                            double* MASCOT_A_RESTRICT HV) {
    const double* MASCOT_A_RESTRICT coef2 = S2 + 2 * K;
    const double* MASCOT_A_RESTRICT h62 = S2 + 2 * K + 2;
    // one pass over the stage input: c * p for the migration product, hv and r for the coalescent term
    double cl[LB], r[LB];
    for (int j = 0; j < LB; j++) { cl[j] = clk[j]; r[j] = 0.0; }
    MASCOT_A_NOVEC for (int s = 0; s < K; s++) {
        const double* MASCOT_A_RESTRICT p = P + s * LD;
        double* MASCOT_A_RESTRICT cp = CP + s * LB;
        double* MASCOT_A_RESTRICT hvs = HV + s * LB;
        for (int j = 0; j < LB; j++) {
            cp[j] = cl[j] * p[j];
            const double hv = (S2[2 * s + (j & 1)] - p[j]) * Q2[2 * s + (j & 1)];
            hvs[j] = hv;
            r[j] = MASCOT_MUL_ADD(p[j], hv, r[j]);
        }
    }
    fwdGemm<LB>(K, CP, M2, OUT);
    if constexpr (STAGE < 4) {
        MASCOT_A_NOVEC for (int s = 0; s < K; s++) {
            const size_t o = s * LD;
            const double* MASCOT_A_RESTRICT out = OUT + s * LB;
            const double* MASCOT_A_RESTRICT hvs = HV + s * LB;
            double yv[LB];
            for (int j = 0; j < LB; j++) {
                // reference: k = matvec + (-p) * (hv - r)
                const double k = MASCOT_MUL_ADD(-P[o + j], hvs[j] - r[j], out[j]);
                if constexpr (STAGE == 1) acc[o + j] = k;
                else acc[o + j] = acc[o + j] + 2.0 * k;  // 2k is exact, so fusing would not change the result
                yv[j] = MASCOT_MUL_ADD(coef2[j & 1], k, cur[o + j]);
                Y[o + j] = yv[j];
            }
            addLanes<LB>(nsums + s, nsq + s, yv, Y + o, nreal);
        }
    } else {
        double sum[LB];
        for (int j = 0; j < LB; j++) sum[j] = 0.0;
        MASCOT_A_NOVEC for (int s = 0; s < K; s++) {
            const size_t o = s * LD;
            const double* MASCOT_A_RESTRICT out = OUT + s * LB;
            const double* MASCOT_A_RESTRICT hvs = HV + s * LB;
            for (int j = 0; j < LB; j++) {
                const double k = MASCOT_MUL_ADD(-P[o + j], hvs[j] - r[j], out[j]);
                // left unfused: fused, clang's SLP cost model leaves the 4-lane block of this stage scalar
                const double a = cur[o + j] + h62[j & 1] * (acc[o + j] + k);
                sum[j] += a;
                Y[o + j] = a;
            }
        }
        MASCOT_A_NOVEC for (int s = 0; s < K; s++) {
            const size_t o = s * LD;
            double yv[LB];
            for (int j = 0; j < LB; j++) {
                yv[j] = Y[o + j] / sum[j];
                Y[o + j] = yv[j];
            }
            addLanes<LB>(nsums + s, nsq + s, yv, Y + o, nreal);
        }
        if (sc)
            for (int j = 0; j < LB; j++) sc[j] = sum[j];
    }
}

// VJP pass A for one lane block: migration product, clock gradient, gram operand (cys2: c * y, every entry twice),
// hv / r / b, partial out, cv, and the lineage-ordered scalar reductions cSums / gradQ (real lanes only).
template <int LB>
static inline void vjpBlockA(const int K, const size_t LD, const int KP, const int nreal,
                             const double* MASCOT_A_RESTRICT y, const double* MASCOT_A_RESTRICT ar,
                             const double* MASCOT_A_RESTRICT clk, const int* MASCOT_A_RESTRICT ids,
                             double* MASCOT_A_RESTRICT clockGradient, const double* MASCOT_A_RESTRICT MT2,
                             const double* MASCOT_A_RESTRICT Q, const double* MASCOT_A_RESTRICT sums,
                             double* MASCOT_A_RESTRICT V, double* MASCOT_A_RESTRICT out, double* MASCOT_A_RESTRICT hvb,
                             double* MASCOT_A_RESTRICT cvb, double* MASCOT_A_RESTRICT cys2,
                             double* MASCOT_A_RESTRICT art, double* MASCOT_A_RESTRICT cSums,
                             double* MASCOT_A_RESTRICT gradQ) {
    // pass 1 (independent of the migration product): gram operand c * y, hv, r, b
    double cl[LB], r[LB], bb[LB];
    for (int j = 0; j < LB; j++) { cl[j] = clk[j]; r[j] = 0.0; bb[j] = 0.0; }
    MASCOT_A_NOVEC for (int s = 0; s < K; s++) {
        const size_t o = s * LD;
        const double sm = sums[s], q = Q[s];
        for (int j = 0; j < LB; j++) {
            const double ys = y[o + j];
            cys2[2 * (o + j)] = cys2[2 * (o + j) + 1] = cl[j] * ys;
            const double hv = (sm - ys) * q;
            hvb[o + j] = hv;
            r[j] = MASCOT_MUL_ADD(ys, hv, r[j]);
            bb[j] = MASCOT_MUL_ADD(ar[o + j], ys, bb[j]);
        }
    }
    vjpGemm<LB>(K, (int) LD, ar, MT2, V);
    // lineage-major copy of the adjoint, the vectorized operand of the gram product (2 x 2 blocks: zip1 / zip2)
    for (int j = 0; j < LB; j += 2) {
        double* MASCOT_A_RESTRICT r0 = art + (size_t) j * KP;
        double* MASCOT_A_RESTRICT r1 = r0 + KP;
        int s = 0;
        MASCOT_A_NOVEC for (; s + 1 < K; s += 2) {
            const double a0 = ar[s * LD + j], a1 = ar[s * LD + j + 1];
            const double b0 = ar[(s + 1) * LD + j], b1 = ar[(s + 1) * LD + j + 1];
            r0[s] = a0; r0[s + 1] = b0;
            r1[s] = a1; r1[s + 1] = b1;
        }
        if (s < K) { r0[s] = ar[s * LD + j]; r1[s] = ar[s * LD + j + 1]; }
    }
    // pass 2: out = c * v + (up * (r - hv) + b * hv) (the reference rounds c * v, then adds), clock term, cv
    double cc[LB];
    for (int j = 0; j < LB; j++) cc[j] = 0.0;
    MASCOT_A_NOVEC for (int s = 0; s < K; s++) {
        const size_t o = s * LD;
        const double* MASCOT_A_RESTRICT v = V + s * LB;
        double cv[LB];
        for (int j = 0; j < LB; j++) {
            const double ys = y[o + j];
            const double up = ar[o + j];
            const double hv = hvb[o + j];
            cc[j] = MASCOT_MUL_ADD(ys, v[j], cc[j]);
            out[o + j] = MASCOT_MUL_ADD(cl[j], v[j], MASCOT_MUL_ADD(up, r[j] - hv, bb[j] * hv));
            cv[j] = ys * (bb[j] - up);
            cvb[o + j] = cv[j];
        }
        double cs = cSums[s], gq = gradQ[s];
        const double sm = sums[s];
        if (nreal == LB) {
            for (int j = 0; j < LB; j++) { cs += cv[j]; gq += cv[j] * (sm - y[o + j]); }
        } else {
            for (int j = 0; j < nreal; j++) { cs += cvb[o + j]; gq += cvb[o + j] * (sm - y[o + j]); }
        }
        cSums[s] = cs;
        gradQ[s] = gq;
    }
    if (clockGradient)
        for (int j = 0; j < nreal; j++) clockGradient[ids[j]] += cc[j];
}

// VJP pass B for one lane block: finish out and combine into the RK4 adjoints.
// MODE 4: R = cx * X + co * out, Y0 = X + out      (aK3 = h3 x + h vjp4)
// MODE 3/2: R = cx * X + co * out, Y0 = Y0 + out   (aK2 = h3 x + hh vjp3; aK1 = h6 x + hh vjp2)
// MODE 1: X = Y0 + out
template <int MODE, int LB>
static inline void vjpBlockB(const int K, const size_t LD, const double* MASCOT_A_RESTRICT Q,
                             const double* MASCOT_A_RESTRICT cSums, const double ellAdj,
                             const double* MASCOT_A_RESTRICT out, const double* MASCOT_A_RESTRICT hvb,
                             const double* MASCOT_A_RESTRICT cvb, double* MASCOT_A_RESTRICT X,
                             double* MASCOT_A_RESTRICT Y0, double* MASCOT_A_RESTRICT R, const double cx,
                             const double co) {
    MASCOT_A_NOVEC for (int s = 0; s < K; s++) {
        const size_t o = s * LD;
        const double q = Q[s], cs = cSums[s];
        for (int j = 0; j < LB; j++) {
            // out + (q * (cs - c) - ellAdj * hv)
            const double ov = out[o + j] + MASCOT_MUL_ADD(-ellAdj, hvb[o + j], q * (cs - cvb[o + j]));
            if constexpr (MODE == 4) {
                R[o + j] = MASCOT_MUL_ADD(co, ov, cx * X[o + j]);
                Y0[o + j] = X[o + j] + ov;
            } else if constexpr (MODE == 3 || MODE == 2) {
                R[o + j] = MASCOT_MUL_ADD(co, ov, cx * X[o + j]);
                Y0[o + j] = Y0[o + j] + ov;
            } else {
                X[o + j] = Y0[o + j] + ov;
            }
        }
    }
}

// Projection VJP for one lane block: dot = 0 + sum_s a * post (deme order); a = (a - dot) / scale; also the first
// RK4 adjoint input aK4 = h6 * a.
template <int LB>
static inline void projBlock(const int K, const size_t LD, double* MASCOT_A_RESTRICT X,
                             const double* MASCOT_A_RESTRICT post, const double* MASCOT_A_RESTRICT scl,
                             double* MASCOT_A_RESTRICT R, const double h6) {
    double dot[LB];
    for (int j = 0; j < LB; j++) dot[j] = 0.0;
    MASCOT_A_NOVEC for (int s = 0; s < K; s++)
        for (int j = 0; j < LB; j++) dot[j] = MASCOT_MUL_ADD(X[s * LD + j], post[s * LD + j], dot[j]);
    MASCOT_A_NOVEC for (int s = 0; s < K; s++)
        for (int j = 0; j < LB; j++) {
            const double x = (X[s * LD + j] - dot[j]) / scl[j];
            X[s * LD + j] = x;
            R[s * LD + j] = h6 * x;
        }
}

// ------------------------------------------------------------------------------------------------------------------
// Workspace (all sizes runtime; nothing is bounded by a compile-time K). Grows on demand, never shrinks.
// ------------------------------------------------------------------------------------------------------------------
struct Workspace {
    int K = 0, LDmax = 0, KP = 0;
    // per epoch: Q (K); M2, MT2 (K x K) and Q2 (K) hold every entry twice, as 2-lane broadcast operands
    std::vector<double> Q, M2, MT2, Q2;
    std::vector<double> y, clock;                       // slot-major live state
    std::vector<int> activeIds, slotOf;
    std::vector<double> v;                              // coalescent scratch (K)
    // segment SoA buffers (K x LDmax) and lane vectors (LDmax)
    std::vector<double> cur, nxt, ysA, ysB, acc, clk;
    std::vector<double> sums, sq, nsums, nsq;           // K
    std::vector<double> S2;                             // stage broadcasts: sums twice (2K), coef, coef, h6, h6
    std::vector<double> blk;                            // lane-block scratch: CP, OUT, HV (3 x K x 8)
    // tape
    struct IntervalRec { int steps, active, epoch, LD; double h; size_t y0Off, yOff, scaleOff, sumsOff, idOff; };
    struct CoalRec { int epoch, first, second, parentSlot, movedFrom, movedTo, parentId; size_t pOff; };
    std::vector<IntervalRec> intervals;
    std::vector<CoalRec> coals;
    std::vector<int> ops;                               // -1 sample, >= 0 interval index, <= -2 coalescent -(index+2)
    std::vector<double> y0, y2, y3, y4, scale, sumsTape, clockTape, coalBuf;
    std::vector<int> idTape;
    size_t y0Used = 0, yUsed = 0, scaleUsed = 0, sumsUsed = 0, idUsed = 0, coalUsed = 0;
    // reverse
    std::vector<double> a, b;                           // slot-major adjoint (ping-pong across coalescences)
    std::vector<double> X, aY0, ar1, ar2, outv, hvb, cvb, cys2, art, gram, Vb, gradQ, cSums;  // cys2: 2 x K x LDmax

    void ensure(int k, int nodeCount, int epochCount) {
        K = k;
        KP = roundEven(k);
        LDmax = roundEven(nodeCount) + 2;
        size_t soa = (size_t) k * LDmax;
        Q.resize((size_t) epochCount * k); Q2.resize(2 * Q.size());
        M2.resize((size_t) epochCount * k * k * 2); MT2.resize(M2.size());
        if (y.size() < (size_t) nodeCount * k) y.resize((size_t) nodeCount * k);
        if (clock.size() < (size_t) nodeCount) {
            clock.resize(nodeCount); activeIds.resize(nodeCount); slotOf.resize(nodeCount);
        }
        if (v.size() < (size_t) k) v.resize(k);
        if (cur.size() < soa) { cur.resize(soa); nxt.resize(soa); ysA.resize(soa); ysB.resize(soa); acc.resize(soa); }
        if (clk.size() < (size_t) LDmax) clk.resize(LDmax);
        if (sums.size() < (size_t) k) {
            sums.resize(k); sq.resize(k); nsums.resize(k); nsq.resize(k); gradQ.resize(k); cSums.resize(k);
        }
        if (S2.size() < (size_t) 2 * k + 4) S2.resize((size_t) 2 * k + 4);
        if (blk.size() < (size_t) 3 * k * 8) blk.resize((size_t) 3 * k * 8);
    }
    void ensureRev(int k, int nodeCount) {
        size_t soa = (size_t) K * LDmax;
        if (a.size() < (size_t) nodeCount * k) { a.resize((size_t) nodeCount * k); b.resize((size_t) nodeCount * k); }
        if (X.size() < soa) {
            X.resize(soa); aY0.resize(soa); ar1.resize(soa); ar2.resize(soa); outv.resize(soa); hvb.resize(soa);
            cvb.resize(soa); cys2.resize(2 * soa);
        }
        if (art.size() < (size_t) LDmax * KP) art.resize((size_t) LDmax * KP);
        if (gram.size() < (size_t) K * KP) gram.resize((size_t) K * KP);
        if (Vb.size() < (size_t) K * 8) Vb.resize((size_t) K * 8);
    }
};

class KernelA final : public mascot::Kernel {
public:
    int forward(const mascot::Problem& problem, double* logL) override {
        begin(problem);
        const int status = run(false);
        if (status == mascot::OK) *logL = ell;
        return status;
    }

    int forwardAndReverse(const mascot::Problem& problem, double* logL, double* thetaGradient, double* clockGradient,
                          double* ancestralScores) override {
        begin(problem);
        std::fill(thetaGradient, thetaGradient + (size_t) problem.epochCount * (K * (K - 1) + K), 0.0);
        if (clockGradient) std::fill(clockGradient, clockGradient + problem.lineageCount, 0.0);
        if (ancestralScores)
            std::fill(ancestralScores, ancestralScores + (size_t) problem.lineageCount * K,
                      std::numeric_limits<double>::quiet_NaN());
        const int status = run(true);
        if (status != mascot::OK) return status;
        reverse(thetaGradient, clockGradient, ancestralScores);
        *logL = ell;
        return mascot::OK;
    }

    char name() const override { return 'A'; }
    int simdWidth() const override { return 0; }
    bool fusedMultiplyAdd() const override {
#ifdef MASCOT_FP_FMA
        return true;
#else
        return false;
#endif
    }

private:
    const mascot::Problem* in = nullptr;  // the current call's problem; not used after the call returns
    Workspace w;
    int K = 0;
    int active = 0;
    double ell = 0.0;
    int epochCursor = 0;
    bool taped = false;

    void begin(const mascot::Problem& problem) {
        in = &problem;
        K = problem.K;
        w.ensure(K, problem.lineageCount, problem.epochCount);
        copyRates();
    }

    // ---------------- schedule helpers (as MascotForwardEvaluator) ----------------
    void copyRates() {
        const size_t kk = (size_t) K * K;
        std::memcpy(w.Q.data(), in->inversePopulation, sizeof(double) * in->epochCount * K);
        for (size_t i = 0; i < w.Q.size(); i++) w.Q2[2 * i] = w.Q2[2 * i + 1] = w.Q[i];
        for (int e = 0; e < in->epochCount; e++) {
            const double* Me = in->migration + e * kk;
            double* M2e = w.M2.data() + e * kk * 2;
            double* MT2e = w.MT2.data() + e * kk * 2;
            for (int source = 0; source < K; source++)
                for (int sink = 0; sink < K; sink++) {
                    const double m = Me[(size_t) source * K + sink];
                    M2e[2 * ((size_t) source * K + sink)] = M2e[2 * ((size_t) source * K + sink) + 1] = m;
                    MT2e[2 * ((size_t) sink * K + source)] = MT2e[2 * ((size_t) sink * K + source) + 1] = m;
                }
        }
    }

    int epochAt(double t) {
        const double* b = in->boundaries;
        int nb = in->epochCount + 1;
        while (epochCursor + 1 < nb && t >= b[epochCursor + 1] - TOL) epochCursor++;
        return std::min(epochCursor, in->epochCount - 1);
    }

    double nextBoundaryAfter(double start, double stop) {
        const double* b = in->boundaries;
        int nb = in->epochCount + 1;
        for (int i = epochCursor + 1; i < nb; i++) {
            if (b[i] > start + TOL) return b[i] <= stop + TOL ? std::min(b[i], stop) : stop;
        }
        return stop;
    }

    // MascotForwardEvaluator.checkProbabilities on the slot-major state.
    bool probabilitiesValid() const {
        const double* y = w.y.data();
        for (int l = 0; l < active; l++) {
            const double* p = y + (size_t) l * K;
            double sum = 0.0;
            for (int s = 0; s < K; s++) {
                if (p[s] < -1.0e-8 || !std::isfinite(p[s])) return false;
                sum += p[s];
            }
            if (std::fabs(sum - 1.0) > 1.0e-6) return false;
        }
        return true;
    }

    int run(bool tape) {
        taped = tape;
        if (taped) {
            w.intervals.clear(); w.coals.clear(); w.ops.clear();
            w.y0Used = w.yUsed = w.scaleUsed = w.sumsUsed = w.idUsed = w.coalUsed = 0;
        }
        active = 0;
        ell = 0.0;
        epochCursor = 0;
        const bool check = in->checkProbabilities;
        sample(in->initialSampleNode);
        double t = in->initialTime;
        for (int i = 0; i < in->intervalCount; i++) {
            double end = t + in->intervalLengths[i];
            while (end > t + TOL) {
                double segEnd = nextBoundaryAfter(t, end);
                int epoch = epochAt(t + TOL);
                integrate(t, segEnd, epoch);
                t = segEnd;
                if (check && !probabilitiesValid()) return mascot::ERROR_PROBABILITIES;
            }
            t = end;
            if (in->eventTypes[i] == mascot::SAMPLE) {
                sample(in->sampleNodes[i]);
            } else if (!coalesce(in->child1[i], in->child2[i], in->parent[i], epochAt(t))) {
                return mascot::ERROR_COALESCENT_RATE;
            }
            if (check && !probabilitiesValid()) return mascot::ERROR_PROBABILITIES;
        }
        return mascot::OK;
    }

    void sample(int node) {
        int slot = active++;
        std::memcpy(w.y.data() + (size_t) slot * K, in->tipPartials + (size_t) node * K, sizeof(double) * K);
        w.activeIds[slot] = node;
        w.slotOf[node] = slot;
        w.clock[slot] = in->branchRates ? in->branchRates[node] : 1.0;
        if (taped) w.ops.push_back(-1);
    }

    bool coalesce(int child1, int child2, int parent, int epoch) {
        double* y = w.y.data();
        int first = w.slotOf[child1], second = w.slotOf[child2];
        const double* Q = w.Q.data() + (size_t) epoch * K;
        double* v = w.v.data();
        const double* y1 = y + (size_t) first * K;
        const double* y2 = y + (size_t) second * K;
        double lambda = 0.0;
        for (int s = 0; s < K; s++) {
            v[s] = y1[s] * y2[s] * Q[s];
            lambda += v[s];
        }
        if (!(lambda > 0.0) || !std::isfinite(lambda)) return false;
        size_t pOff = 0;
        if (taped) {
            double* rec = growTape(w.coalBuf, w.coalUsed, 3 * (size_t) K);  // p1, p2, parent
            pOff = rec - w.coalBuf.data();
            std::memcpy(rec, y + (size_t) first * K, sizeof(double) * K);
            std::memcpy(rec + K, y + (size_t) second * K, sizeof(double) * K);
        }
        int last = active - 1;
        int parentSlot = first == last ? second : first;
        int removed = parentSlot == first ? second : first;
        int movedFrom = -1, movedTo = -1;
        if (removed != last) {
            movedFrom = last;
            movedTo = removed;
            int moved = w.activeIds[last];
            w.activeIds[removed] = moved;
            std::memcpy(y + (size_t) removed * K, y + (size_t) last * K, sizeof(double) * K);
            w.clock[removed] = w.clock[last];
            w.slotOf[moved] = removed;
        }
        double* yp = y + (size_t) parentSlot * K;
        for (int s = 0; s < K; s++) yp[s] = v[s] / lambda;
        if (taped) std::memcpy(w.coalBuf.data() + pOff + 2 * K, yp, sizeof(double) * K);
        w.activeIds[parentSlot] = parent;
        w.slotOf[parent] = parentSlot;
        w.clock[parentSlot] = in->branchRates ? in->branchRates[parent] : 1.0;
        active--;
        ell += std::log(lambda);
        if (taped) {
            w.coals.push_back({epoch, first, second, parentSlot, movedFrom, movedTo, parent, pOff});
            w.ops.push_back(-(int) (w.coals.size() - 1) - 2);
        }
        return true;
    }

    // ---------------- forward integration ----------------
    struct Stage {
        int K, LD, A;
        const double* M2;
        const double* Q2;
        const double* clk;
        const double* sums;   // of the stage input
        double* nsums;        // of the next stage input (accumulated here)
        double* nsq;
        double* S2;
        double* CP;
        double* OUT;
        double* HV;
    };

    template <int STAGE>
    static inline void fwdStage(const Stage& c, const double* P, const double* cur, double* acc, double* Y,
                                double* sc, double coef, double h6) {
        for (int s = 0; s < c.K; s++) { c.nsums[s] = 0.0; c.nsq[s] = 0.0; }
        for (int s = 0; s < c.K; s++) c.S2[2 * s] = c.S2[2 * s + 1] = c.sums[s];
        c.S2[2 * c.K] = c.S2[2 * c.K + 1] = coef;
        c.S2[2 * c.K + 2] = c.S2[2 * c.K + 3] = h6;
        const size_t LD = c.LD;
        int lb = 0;
#define MASCOT_A_FWD_BLOCK(LBV)                                                                                     \
    fwdBlock<STAGE, LBV>(c.K, LD, std::min(LBV, c.A - lb), P + lb, cur + lb, acc + lb, Y + lb,                     \
                         sc ? sc + lb : nullptr, c.clk + lb, c.M2, c.Q2, c.S2, c.nsums, c.nsq, c.CP, c.OUT, c.HV)
        for (; lb + 8 <= c.LD; lb += 8) MASCOT_A_FWD_BLOCK(8);
        if (lb + 4 <= c.LD) { MASCOT_A_FWD_BLOCK(4); lb += 4; }
        if (lb < c.LD) MASCOT_A_FWD_BLOCK(2);
#undef MASCOT_A_FWD_BLOCK
    }

    double hazardOf(const double* sums, const double* sq, const double* Q) const {
        double hazard = 0.0;
        for (int s = 0; s < K; s++) hazard += 0.5 * Q[s] * (sums[s] * sums[s] - sq[s]);
        return -hazard;
    }

    void integrate(double start, double end, int epoch) {
        const int steps = std::max(1, (int) std::ceil((end - start) / in->maxStep));
        const double h = (end - start) / steps;
        const double hh = 0.5 * h;
        const double h6 = h / 6.0;
        const int A = active;
        const int LD = roundEven(A);
        const size_t n = (size_t) K * LD;
        const double* M2 = w.M2.data() + (size_t) epoch * K * K * 2;
        const double* Q = w.Q.data() + (size_t) epoch * K;
        const double* Q2 = w.Q2.data() + (size_t) epoch * K * 2;

        double *y0 = nullptr, *y2 = nullptr, *y3 = nullptr, *y4 = nullptr, *scl = nullptr, *st = nullptr;
        if (taped) {
            size_t y0Off = w.y0Used, yOff = w.yUsed, scaleOff = w.scaleUsed, sumsOff = w.sumsUsed, idOff = w.idUsed;
            growTape(w.y0, w.y0Used, (size_t) (steps + 1) * n);
            growTape(w.y2, w.yUsed, (size_t) steps * n); w.yUsed = yOff;
            growTape(w.y3, w.yUsed, (size_t) steps * n); w.yUsed = yOff;
            growTape(w.y4, w.yUsed, (size_t) steps * n);
            growTape(w.scale, w.scaleUsed, (size_t) steps * LD);
            growTape(w.sumsTape, w.sumsUsed, (size_t) steps * 8 * K);
            if (w.idTape.size() < idOff + A) {
                w.idTape.resize(std::max(w.idTape.size() * 2, idOff + A));
                w.clockTape.resize(w.idTape.size());
            }
            w.idUsed += A;
            for (int l = 0; l < A; l++) { w.idTape[idOff + l] = w.activeIds[l]; w.clockTape[idOff + l] = w.clock[l]; }
            w.intervals.push_back({steps, A, epoch, LD, h, y0Off, yOff, scaleOff, sumsOff, idOff});
            w.ops.push_back((int) w.intervals.size() - 1);
            y0 = w.y0.data() + y0Off;
            y2 = w.y2.data() + yOff;
            y3 = w.y3.data() + yOff;
            y4 = w.y4.data() + yOff;
            scl = w.scale.data() + scaleOff;
            st = w.sumsTape.data() + sumsOff;
        }

        // slot-major -> deme-major; padded lanes copy lineage 0
        double* cur = taped ? y0 : w.cur.data();
        const double* y = w.y.data();
        for (int s = 0; s < K; s++) {
            double* row = cur + (size_t) s * LD;
            for (int l = 0; l < A; l++) row[l] = y[(size_t) l * K + s];
            for (int l = A; l < LD; l++) row[l] = y[s];
        }
        double* clk = w.clk.data();
        for (int l = 0; l < A; l++) clk[l] = w.clock[l];
        for (int l = A; l < LD; l++) clk[l] = w.clock[0];
        double* sums = w.sums.data();
        double* sq = w.sq.data();
        for (int s = 0; s < K; s++) {
            const double* row = cur + (size_t) s * LD;
            double sm = 0.0, q = 0.0;
            for (int l = 0; l < A; l++) { sm += row[l]; q += row[l] * row[l]; }
            sums[s] = sm;
            sq[s] = q;
        }

        double* blk = w.blk.data();
        Stage c{K, LD, A, M2, Q2, clk, sums, w.nsums.data(), w.nsq.data(), w.S2.data(), blk, blk + (size_t) K * 8,
                blk + (size_t) 16 * K};
        double* acc = w.acc.data();
        double* nxt = w.nxt.data();
        for (int step = 0; step < steps; step++) {
            double *pA, *pB, *pC, *pN;
            double* sc = nullptr;
            if (taped) {
                cur = y0 + (size_t) step * n;
                pA = y2 + (size_t) step * n;
                pB = y3 + (size_t) step * n;
                pC = y4 + (size_t) step * n;
                pN = y0 + (size_t) (step + 1) * n;
                sc = scl + (size_t) step * LD;
            } else {
                pA = w.ysA.data();
                pB = w.ysB.data();
                pC = w.ysA.data();
                pN = nxt;
            }
            double* tp = taped ? st + (size_t) step * 8 * K : nullptr;
            auto stage = [&](auto tag, const double* P, double* Y, double coef) {
                constexpr int S = decltype(tag)::value;
                if (tp) {
                    std::memcpy(tp + (S - 1) * 2 * K, sums, sizeof(double) * K);
                    std::memcpy(tp + (S - 1) * 2 * K + K, sq, sizeof(double) * K);
                }
                double e = hazardOf(sums, sq, Q);
                fwdStage<S>(c, P, cur, acc, Y, sc, coef, h6);
                std::swap(c.nsums, sums);  // the next stage input's sums become current
                std::swap(c.nsq, sq);
                c.sums = sums;
                return e;
            };
            double e1 = stage(std::integral_constant<int, 1>(), cur, pA, hh);
            double e2 = stage(std::integral_constant<int, 2>(), pA, pB, hh);
            double e3 = stage(std::integral_constant<int, 3>(), pB, pC, h);
            double e4 = stage(std::integral_constant<int, 4>(), pC, pN, 0.0);
            ell = ell + h6 * (e1 + 2.0 * e2 + 2.0 * e3 + e4);
            if (!taped) std::swap(cur, nxt);
        }
        // final state (untaped: cur / nxt ping-pong storage; taped: last tape slot) -> slot-major
        const double* fin = taped ? y0 + (size_t) steps * n : cur;
        double* yy = w.y.data();
        for (int s = 0; s < K; s++) {
            const double* row = fin + (size_t) s * LD;
            for (int l = 0; l < A; l++) yy[(size_t) l * K + s] = row[l];
        }
    }

    // ---------------- reverse ----------------
    void reverse(double* gradient, double* clockGradient, double* ancestral) {
        w.ensureRev(K, in->lineageCount);
        int act = active;
        double* a = w.a.data();
        double* b = w.b.data();
        std::fill(a, a + (size_t) act * K, 0.0);
        const double aell = 1.0;  // adjoint of logL; every operation passes it through unchanged
        for (int op = (int) w.ops.size() - 1; op >= 0; op--) {
            int code = w.ops[op];
            if (code == -1) {
                act--;
            } else if (code >= 0) {
                reverseInterval(w.intervals[code], a, aell, gradient, clockGradient);
            } else {
                reverseCoalescent(w.coals[-code - 2], a, b, act, aell, gradient, ancestral);
                std::swap(a, b);
                act++;
            }
        }
    }

    struct Vjp {
        int K, LD, A, KP;
        const double* MT2;
        const double* Q;
        const double* clk;
        const int* ids;
        double* clockGradient;
        double* gTheta;
        double* gEta;
        // scratch
        double* outv;
        double* hvb;
        double* cvb;
        double* cys2;
        double* art;
        double* gram;
        double* Vb;
        double* gradQ;
        double* cSums;
    };

    template <int MODE>
    void rhsVjp(const Vjp& c, const double* y, const double* ar, const double* sums, const double* sq, double ellAdj,
                double* X, double* Y0, double* R, double cx, double co) {
        for (int s = 0; s < K; s++) {
            double pairSum = 0.5 * (sums[s] * sums[s] - sq[s]);
            c.gradQ[s] = -ellAdj * pairSum;
            c.cSums[s] = 0.0;
        }
        const size_t LD = c.LD;
        int lb = 0;
#define MASCOT_A_VJP_A(LBV)                                                                                         \
    vjpBlockA<LBV>(K, LD, c.KP, std::min(LBV, c.A - lb), y + lb, ar + lb, c.clk + lb, c.ids + lb, c.clockGradient,  \
                   c.MT2, c.Q, sums, c.Vb, c.outv + lb, c.hvb + lb, c.cvb + lb, c.cys2 + 2 * lb,                    \
                   c.art + (size_t) lb * c.KP, c.cSums, c.gradQ)
        for (; lb + 8 <= c.LD; lb += 8) MASCOT_A_VJP_A(8);
        if (lb + 4 <= c.LD) { MASCOT_A_VJP_A(4); lb += 4; }
        if (lb < c.LD) MASCOT_A_VJP_A(2);
#undef MASCOT_A_VJP_A
        // migration-rate gradient: gram over lineages (in order), then gTheta += gram - diagonal
        gramAll(K, c.A, c.LD, c.KP, c.cys2, c.art, c.gram);
        int rateIndex = 0;
        for (int source = 0; source < K; source++) {
            const double* g = c.gram + (size_t) source * c.KP;
            const double diagonal = g[source];
            for (int sink = 0; sink < K; sink++) {
                if (source == sink) continue;
                c.gTheta[rateIndex++] += g[sink] - diagonal;
            }
        }
        for (int s = 0; s < K; s++) c.gEta[s] += -c.Q[s] * c.gradQ[s];
        lb = 0;
#define MASCOT_A_VJP_B(LBV)                                                                                         \
    vjpBlockB<MODE, LBV>(K, LD, c.Q, c.cSums, ellAdj, c.outv + lb, c.hvb + lb, c.cvb + lb, X + lb, Y0 + lb,         \
                         R ? R + lb : nullptr, cx, co)
        for (; lb + 8 <= c.LD; lb += 8) MASCOT_A_VJP_B(8);
        if (lb + 4 <= c.LD) { MASCOT_A_VJP_B(4); lb += 4; }
        if (lb < c.LD) MASCOT_A_VJP_B(2);
#undef MASCOT_A_VJP_B
    }

    void reverseInterval(const Workspace::IntervalRec& r, double* aSlot, double aell, double* gradient,
                         double* clockGradient) {
        const int A = r.active;
        const int LD = r.LD;
        const size_t n = (size_t) K * LD;
        const double h = r.h;
        const double* y0 = w.y0.data() + r.y0Off;
        const double* y2 = w.y2.data() + r.yOff;
        const double* y3 = w.y3.data() + r.yOff;
        const double* y4 = w.y4.data() + r.yOff;
        const double* sc = w.scale.data() + r.scaleOff;
        const double* st = w.sumsTape.data() + r.sumsOff;
        const int ppe = K * (K - 1) + K;
        double* clk = w.clk.data();
        const double* ct = w.clockTape.data() + r.idOff;
        for (int l = 0; l < A; l++) clk[l] = ct[l];
        for (int l = A; l < LD; l++) clk[l] = ct[0];
        Vjp c{K, LD, A, w.KP,
              w.MT2.data() + (size_t) r.epoch * K * K * 2, w.Q.data() + (size_t) r.epoch * K, clk,
              w.idTape.data() + r.idOff, clockGradient,
              gradient + (size_t) r.epoch * ppe, gradient + (size_t) r.epoch * ppe + K * (K - 1),
              w.outv.data(), w.hvb.data(), w.cvb.data(), w.cys2.data(), w.art.data(), w.gram.data(), w.Vb.data(),
              w.gradQ.data(), w.cSums.data()};
        double* X = w.X.data();
        double* Y0 = w.aY0.data();
        double* R1 = w.ar1.data();
        double* R2 = w.ar2.data();
        // slot-major adjoint -> deme-major; padded lanes start at zero
        for (int s = 0; s < K; s++) {
            double* row = X + (size_t) s * LD;
            for (int l = 0; l < A; l++) row[l] = aSlot[(size_t) l * K + s];
            for (int l = A; l < LD; l++) row[l] = 0.0;
        }
        const double h6 = h / 6.0, h3 = h / 3.0, hh = 0.5 * h;
        for (int step = r.steps - 1; step >= 0; step--) {
            // VJP of the per-lineage projection y / (1'y)
            const double* post = y0 + (size_t) (step + 1) * n;
            const double* scl = sc + (size_t) step * LD;
            {
                int lb = 0;
                for (; lb + 8 <= LD; lb += 8) projBlock<8>(K, LD, X + lb, post + lb, scl + lb, R1 + lb, h6);
                if (lb + 4 <= LD) { projBlock<4>(K, LD, X + lb, post + lb, scl + lb, R1 + lb, h6); lb += 4; }
                if (lb < LD) projBlock<2>(K, LD, X + lb, post + lb, scl + lb, R1 + lb, h6);
            }
            // logL-slot adjoints of the four stages (Java adds h * 0.0 or hh * 0.0 to these, which is exact)
            const double* ts = st + (size_t) step * 8 * K;  // per stage: sums, sq
            const double e1 = h6 * aell, e2 = h3 * aell, e3 = h3 * aell, e4 = h6 * aell;
            rhsVjp<4>(c, y4 + (size_t) step * n, R1, ts + 6 * K, ts + 7 * K, e4, X, Y0, R2, h3, h);   // R2 = aK3
            rhsVjp<3>(c, y3 + (size_t) step * n, R2, ts + 4 * K, ts + 5 * K, e3, X, Y0, R1, h3, hh);  // R1 = aK2
            rhsVjp<2>(c, y2 + (size_t) step * n, R1, ts + 2 * K, ts + 3 * K, e2, X, Y0, R2, h6, hh);  // R2 = aK1
            rhsVjp<1>(c, y0 + (size_t) step * n, R2, ts, ts + K, e1, X, Y0, nullptr, 0.0, 0.0);      // X = a
        }
        for (int s = 0; s < K; s++) {
            const double* row = X + (size_t) s * LD;
            for (int l = 0; l < A; l++) aSlot[(size_t) l * K + s] = row[l];
        }
    }

    // MascotAdjointEvaluator.reverseCoalescentInto, including the ancestral state scores.
    void reverseCoalescent(const Workspace::CoalRec& r, const double* after, double* before, int afterCount,
                           double aell, double* gradient, double* ancestral) {
        const int beforeCount = afterCount + 1;
        const double* p1 = w.coalBuf.data() + r.pOff;
        const double* p2 = p1 + K;
        const double* pp = p2 + K;
        for (int bi = 0; bi < beforeCount; bi++) {
            if (bi == r.first || bi == r.second) continue;
            int ai = bi == r.movedFrom ? r.movedTo : bi;
            std::memcpy(before + (size_t) bi * K, after + (size_t) ai * K, sizeof(double) * K);
        }
        const double* ap = after + (size_t) r.parentSlot * K;
        double dot = 0.0;
        for (int s = 0; s < K; s++) dot += ap[s] * pp[s];
        const int ppe = K * (K - 1) + K;
        double* gEta = gradient + (size_t) r.epoch * ppe + K * (K - 1);
        double* score = ancestral ? ancestral + (size_t) r.parentId * K : nullptr;
        for (int s = 0; s < K; s++) {
            double q1 = std::max(p1[s], EPS);
            double q2 = std::max(p2[s], EPS);
            double centered = ap[s] - dot;
            double nodeStateScore = pp[s] * (centered + aell);
            before[(size_t) r.first * K + s] = nodeStateScore / q1;
            before[(size_t) r.second * K + s] = nodeStateScore / q2;
            gEta[s] -= nodeStateScore;
            if (score) score[s] = nodeStateScore;
        }
    }
};

}  // namespace mascot_a_strict / mascot_a_fma

#undef MASCOT_A_RESTRICT
#undef MASCOT_A_NOVEC

#endif
