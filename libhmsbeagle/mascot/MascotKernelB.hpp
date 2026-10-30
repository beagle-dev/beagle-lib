/*
 * MascotKernelB.hpp -- MASCOT kernel B: one compiled kernel for any runtime K >= 2, vectorized across states.
 *
 * Included only by MascotKernelB.cpp, which is compiled twice (strict, and with MASCOT_FP_FMA); see there.
 *
 * What it computes. The same as BEAST's GenericMascotLikelihoodDelegate (MascotForwardEvaluator and
 * MascotAdjointEvaluator), from a mascot::Problem whose migration matrices M (row = source, diagonal = -row sum) and
 * inverse population sizes Q were computed by Java:
 *  - Forward. Each active lineage l carries state probabilities p_l (K values). Between tree events they follow
 *        dp_l/dt = c_l p_l M - p_l o (h_l - r_l),   h_l = (S - p_l) o Q,   r_l = p_l . h_l,   S = sum_l p_l,
 *        dlogL/dt = -sum_s Q_s (S_s^2 - sum_l p_ls^2) / 2,
 *    where c_l is the lineage's branch rate (1 without branch rates). Each interval is split at epoch boundaries and
 *    each segment is integrated with classical RK4 in equal steps; after every step each lineage is renormalized to
 *    sum 1. The step count is BEAST's MascotStepRule, with its operation order: FIXED, max(1, ceil(length / maxStep));
 *    ADAPTIVE, from the stiffness clock * maxOut + maxLoad of the state at the segment's start (largest active branch
 *    rate, largest -M_kk, largest Q_k sum_l p_lk), bounded by maxStep, minStep and maxStepsPerSegment. The rule uses
 *    plain arithmetic in both FP modes, so strict and FMA builds take Java's step counts; they are recorded for
 *    lastStepCounts. A sample event appends the tip's row. A coalescence of lineages a and b replaces them by the
 *    parent row p_a o p_b o Q / lambda with lambda = sum_s p_as p_bs Q_s, and adds log(lambda) to logL.
 *  - forwardAndReverse. The forward pass also records a tape: the stage inputs y0, y2, y3, y4 of every RK4 step, the
 *    normalization scales, the lineage sums (S and sum of squares) of every stage input, the active lineage ids and
 *    clock rates of every segment, and the operands of every coalescence. The reverse pass replays it backwards (the
 *    projection VJP, the four RHS VJPs of each step, the coalescent VJP) and accumulates
 *      - dlogL/dM, mapped to the off-diagonal rates: g[src][sink] - g[src][src] (theta, K(K-1) per epoch, row-major);
 *      - dlogL/dQ, mapped to log Ne through dQ/dlogNe = -Q (theta, the K entries after the rates of each epoch);
 *      - dlogL/dc per lineage (clockGradient);
 *      - the ancestral state scores of every coalescent parent, p_parent o (centered adjoint + logL adjoint), written
 *        into ancestralScores (NaN in every other row).
 *  - Status. A coalescent rate lambda <= 0 or non-finite returns ERROR_COALESCENT_RATE. With
 *    Problem::checkProbabilities, every active lineage is checked after each integrated segment and after each event
 *    (any p < -1e-8 or non-finite, or |sum_s p - 1| > 1e-6) and a violation returns ERROR_PROBABILITIES.
 *
 * Floating point. Every operation is the one Java performs, with the same operands in the same order:
 *  - elementwise expressions are evaluated as in the Java source;
 *  - every reduction over states (r, b, dot, clock contribution, normalizer, hazard, lambda) is a sequential sum
 *    s = 0..K-1 starting from +0.0: products are formed in vectors but summed lane by lane;
 *  - sums over lineages (S, sum of squares, gram, cSums, gradQ) keep lineage order 0..A-1.
 * Both builds compile with -ffp-contract=off. Multiply-adds on the hot paths (the K x K products, the RK4 stage
 * combinations, the VJP epilogues, the lineage sums of squares, the logL and gradient updates) are written with FMAV /
 * FMAS (MascotFma.hpp): the FMA build fuses exactly those, and the strict build evaluates them as a rounded product
 * then a rounded sum, the Java order. The lane-by-lane reductions over states are never fused. So the strict build
 * gives Java's gradients and ancestral scores bit for bit (logL differs only through the platform's log()), and the
 * FMA build's results do not depend on the compiler or its optimization level.
 *
 * Layout. Every lineage row has stride Kp = roundUp(K, W), W doubles per vector (a compile-time constant, never a
 * function of K). Padded lanes hold +-0 throughout the forward pass: their columns of M and M^T and their entries of
 * Q are 0, and sample and parent rows are padded with +0. Adding a +-0 lane to a reduction that starts at +0.0 never
 * changes it, so padded lanes ride along in vector form. In the reverse pass the padded lanes of the adjoint are NOT
 * self-limiting (they grow every step, and inf * 0 would poison the real lanes), so they are masked to exactly 0 after
 * every projection VJP.
 *
 * Blocking. The K x K products (forward p^T M, adjoint M a through M^T, and the gram rank-1 updates) are register
 * blocked over NB chunks of W outputs (NB = NBMAX, plus one remainder block of 1..NBMAX-1 chunks: a fixed set of edge
 * kernels independent of K) and over 2 lineages, so each M, M^T or gram chunk loaded serves two lineages. The gram
 * update is fused into the M^T loop. The per-lineage O(K) work is fused into the block epilogues (the stage update,
 * the next stage's lineage sums, and the out / cv / cSums of the VJP), so no K-vector is stored and re-read within a
 * lineage. Hot loops work on local copies of K, Kp, nc and the buffer pointers: vector stores carry no type-based
 * alias information, so member fields would be reloaded after every store.
 *
 * State. Every buffer belongs to the kernel object (no static or thread_local mutable state). Buffers grow as needed
 * and are reused, so a warm object does not allocate.
 */
#ifndef MASCOT_KERNEL_B_HPP
#define MASCOT_KERNEL_B_HPP

#include "MascotFma.hpp"
#include "MascotProblem.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

/* One namespace per FP mode: inline functions compiled with different FP flags must never be merged by the linker. */
#ifdef MASCOT_FP_FMA
#define MASCOT_B_NS mascot_b_fma
#else
#define MASCOT_B_NS mascot_b_strict
#endif

/* Doubles per vector: 2 on every instruction set (NEON; 128-bit SSE2 on x86) until wider vectors are timed. */
#ifndef MASCOT_B_SIMD_WIDTH
#define MASCOT_B_SIMD_WIDTH 2
#endif

#define MASCOT_B_INLINE inline __attribute__((always_inline))

namespace MASCOT_B_NS {

constexpr int W = MASCOT_B_SIMD_WIDTH;
static_assert(W == 2 || W == 4 || W == 8, "MASCOT_B_SIMD_WIDTH must be 2, 4 or 8");
constexpr int NBMAX = W == 2 ? 8 : 4;  // chunks per forward block: 2 lineages x 8 accumulators fit 32 NEON registers
constexpr int NBA = 4;                 // chunks per block of the fused M^T matvec + gram update
constexpr double TOL = 1.0e-14;        // MascotRuntime.TIME_TOLERANCE
constexpr double EPS = 1.0e-300;       // MascotAdjointEvaluator.EPS
constexpr double LARGEST_STEP_COUNT = 1073741824.0;  // MascotStepRule.LARGEST_STEP_COUNT, 2^30

// Java's Math.max / Math.min on doubles: NaN if either argument is NaN, and +0 > -0.
static inline double javaMax(double a, double b) {
    if (a != a) return a;
    if (a == 0.0 && b == 0.0 && std::signbit(a)) return b;
    return a >= b ? a : b;
}
static inline double javaMin(double a, double b) {
    if (a != a) return a;
    if (a == 0.0 && b == 0.0 && std::signbit(b)) return b;
    return a <= b ? a : b;
}

// ---------------------------------------------------------------------- vector abstraction
// clang ext_vector_type; a GCC vector_size or MSVC port changes only this block.
typedef double vd __attribute__((ext_vector_type(MASCOT_B_SIMD_WIDTH)));
typedef int64_t vi __attribute__((ext_vector_type(MASCOT_B_SIMD_WIDTH)));

static MASCOT_B_INLINE vd LD(const double* p) { return *(const vd*) p; }
static MASCOT_B_INLINE void ST(double* p, vd v) { *(vd*) p = v; }
static MASCOT_B_INLINE vd SPL(double x) { return (vd) x; }
static MASCOT_B_INLINE void addLanes(double& acc, vd v) {
    for (int j = 0; j < W; j++) acc += v[j];
}
static MASCOT_B_INLINE vd maskv(vd x, vi m) { return (vd) ((vi) x & m); }
// a * b + c: one fused operation in the FMA build, a rounded product then a rounded sum in the strict build.
static MASCOT_B_INLINE vd FMAV(vd a, vd b, vd c) { return MASCOT_VMUL_ADD(a, b, c); }
static MASCOT_B_INLINE double FMAS(double a, double b, double c) { return MASCOT_MUL_ADD(a, b, c); }
static inline vi laneMask(int realLanes) {
    vi m;
    for (int j = 0; j < W; j++) m[j] = j < realLanes ? -1 : 0;
    return m;
}

// Dispatch a block template over the chunk range [0, NC): full NBM-chunk blocks, then one remainder block of
// 1..NBM-1 chunks (cases >= NBM are unreachable; they are clamped so the templates stay valid for any NBM <= 8).
#define MASCOT_B_CL(n, NBM) ((n) < (NBM) ? (n) : 1)
#define MASCOT_B_BLOCKS(NBM, NC, CALL)                                 \
    do {                                                               \
        int cb_ = 0;                                                   \
        for (; cb_ + (NBM) <= (NC); cb_ += (NBM)) CALL((NBM), cb_);    \
        switch ((NC) - cb_) {                                          \
            case 1: CALL(MASCOT_B_CL(1, NBM), cb_); break;             \
            case 2: CALL(MASCOT_B_CL(2, NBM), cb_); break;             \
            case 3: CALL(MASCOT_B_CL(3, NBM), cb_); break;             \
            case 4: CALL(MASCOT_B_CL(4, NBM), cb_); break;             \
            case 5: CALL(MASCOT_B_CL(5, NBM), cb_); break;             \
            case 6: CALL(MASCOT_B_CL(6, NBM), cb_); break;             \
            case 7: CALL(MASCOT_B_CL(7, NBM), cb_); break;             \
            default: break;                                            \
        }                                                              \
    } while (0)

// Lineage blocks over [0, A): pairs, then one single lineage. CALL(NL, l).
#define MASCOT_B_LINEAGES(A, CALL)                         \
    do {                                                   \
        int l_ = 0;                                        \
        for (; l_ + 2 <= (A); l_ += 2) CALL(2, l_);        \
        if (l_ < (A)) CALL(1, l_);                         \
    } while (0)

// 64-byte aligned, growable double buffer.
struct ABuf {
    double* p = nullptr;
    size_t cap = 0;
    ABuf() = default;
    ABuf(const ABuf&) = delete;
    ABuf& operator=(const ABuf&) = delete;
    ~ABuf() { std::free(p); }
    double* data() { return p; }
    const double* data() const { return p; }
    // Grows to at least m doubles, keeping the first `keep`.
    void reserve(size_t m, size_t keep = 0) {
        if (m <= cap) return;
        size_t nc = std::max(m, cap * 2);
        void* q = nullptr;
        if (posix_memalign(&q, 64, ((nc * sizeof(double) + 63) / 64) * 64) != 0) throw std::bad_alloc();
        if (keep) std::memcpy(q, p, keep * sizeof(double));
        std::free(p);
        p = (double*) q;
        cap = nc;
    }
};

static inline size_t take(ABuf& b, size_t& used, size_t add) {
    b.reserve(used + add, used);
    size_t off = used;
    used += add;
    return off;
}

struct Dims { int K, Kp, nc; };

// ====================================================================== forward stage
// Per-lineage-block pointers for one RK4 stage.
template <int NL>
struct FwdRows {
    const double* p[NL];   // stage input
    const double* cur[NL]; // step start
    double* ys[NL];        // stage output (STAGE < 4)
    double* acc[NL];       // k1 + 2 k2 + 2 k3
    double* nxt[NL];       // step end (STAGE 4)
    const double* cp[NL];  // clock * p
    const double* hv[NL];  // (sums - p) * Q
    double r[NL];
    double sm[NL];         // normalizer (STAGE 4)
};

// out_i[chunks cb..cb+NB) = sum_{source < K} cp_i[source] * M[source][.] (first term not added to 0), then the
// stage update for those chunks: k = out + (-p)(hv - r); STAGE 1: acc = k; 2,3: acc = acc + 2k;
// ys = cur + coef k (and its lineage sums for the next stage); STAGE 4: nxt = cur + h6 (acc + k), sm += lanes.
template <int NL, int NB, int STAGE>
static MASCOT_B_INLINE void fwdBlock(const Dims d, int cb, const double* Me, FwdRows<NL>& R, double coef, double h6,
                                     double* sun, double* sqn) {
    const int K = d.K, Kp = d.Kp;
    vd acc[NL][NB];
    const double* m = Me + cb * W;
    for (int j = 0; j < NB; j++) {
        vd mv = LD(m + j * W);
        for (int i = 0; i < NL; i++) acc[i][j] = R.cp[i][0] * mv;
    }
    for (int s = 1; s < K; s++) {
        m += Kp;
        double c[NL];
        for (int i = 0; i < NL; i++) c[i] = R.cp[i][s];
        for (int j = 0; j < NB; j++) {
            vd mv = LD(m + j * W);
            for (int i = 0; i < NL; i++) acc[i][j] = FMAV(SPL(c[i]), mv, acc[i][j]);
        }
    }
    for (int j = 0; j < NB; j++) {
        const int o = (cb + j) * W;
        vd sn, qn;
        if (STAGE < 4) { sn = LD(sun + o); qn = LD(sqn + o); }
        for (int i = 0; i < NL; i++) {
            vd pv = LD(R.p[i] + o);
            vd k = FMAV(-pv, LD(R.hv[i] + o) - R.r[i], acc[i][j]);
            if (STAGE < 4) {
                if (STAGE == 1) ST(R.acc[i] + o, k);
                else ST(R.acc[i] + o, FMAV(SPL(2.0), k, LD(R.acc[i] + o)));
                vd yv = FMAV(SPL(coef), k, LD(R.cur[i] + o));
                ST(R.ys[i] + o, yv);
                sn = sn + yv;
                qn = FMAV(yv, yv, qn);
            } else {
                vd a = FMAV(SPL(h6), LD(R.acc[i] + o) + k, LD(R.cur[i] + o));
                ST(R.nxt[i] + o, a);
                addLanes(R.sm[i], a);
            }
        }
        if (STAGE < 4) { ST(sun + o, sn); ST(sqn + o, qn); }
    }
}

template <int NL, int STAGE>
static MASCOT_B_INLINE void fwdLineages(const Dims d, int l, const double* clk, const double* p, const double* cur,
                                        double* ys, double* accb, double* nxt, double* scaleOut, double coef, double h6,
                                        const double* Me, const double* Qe, const double* su, double* sun, double* sqn,
                                        double* cpb, double* hvb) {
    const int Kp = d.Kp, nc = d.nc;
    FwdRows<NL> R;
    double c[NL];
    for (int i = 0; i < NL; i++) {
        size_t o = (size_t) (l + i) * Kp;
        R.p[i] = p + o;
        R.cur[i] = cur + o;
        R.ys[i] = STAGE < 4 ? ys + o : nullptr;    // ys is null in stage 4, nxt in stages 1-3
        R.acc[i] = accb + o;
        R.nxt[i] = STAGE == 4 ? nxt + o : nullptr;
        R.cp[i] = cpb + i * Kp;
        R.hv[i] = hvb + i * Kp;
        R.r[i] = 0.0;
        R.sm[i] = 0.0;
        c[i] = clk[l + i];
    }
    // cp = c p, hv = (sums - p) Q, r = sum_s p hv (sequential)
    for (int j = 0; j < nc; j++) {
        const int o = j * W;
        vd sj = LD(su + o), qj = LD(Qe + o);
        for (int i = 0; i < NL; i++) {
            vd pv = LD(R.p[i] + o);
            ST(cpb + i * Kp + o, c[i] * pv);
            vd hv = (sj - pv) * qj;
            ST(hvb + i * Kp + o, hv);
            addLanes(R.r[i], pv * hv);
        }
    }
#define MASCOT_B_FWD_CALL(NB, CB) fwdBlock<NL, NB, STAGE>(d, CB, Me, R, coef, h6, sun, sqn)
    MASCOT_B_BLOCKS(NBMAX, nc, MASCOT_B_FWD_CALL);
#undef MASCOT_B_FWD_CALL
    if (STAGE == 4) {
        // normalize; the result is the next step's stage input, so accumulate its lineage sums here
        for (int i = 0; i < NL; i++)
            if (scaleOut) scaleOut[l + i] = R.sm[i];
        for (int j = 0; j < nc; j++) {
            const int o = j * W;
            vd sn = LD(sun + o), qn = LD(sqn + o);
            for (int i = 0; i < NL; i++) {
                vd v = LD(R.nxt[i] + o) / R.sm[i];
                ST(R.nxt[i] + o, v);
                sn = sn + v;
                qn = FMAV(v, v, qn);
            }
            ST(sun + o, sn);
            ST(sqn + o, qn);
        }
    }
}

// One RK4 stage over all A lineages. (su, sq) = lineage sums of p (input); (sun, sqn) receive the lineage sums of
// the stage output (ys for STAGE < 4, normalized nxt for STAGE 4). Returns d(logL)/dt at p.
template <int STAGE>
static MASCOT_B_INLINE double fwdStage(const Dims d, int A, const double* clk, const double* p, const double* cur,
                                       double* ys, double* accb, double* nxt, double* scaleOut, double coef, double h6,
                                       const double* Me, const double* Qe, const double* su, const double* sq,
                                       double* sun, double* sqn, double* cpb, double* hvb) {
    double hazard = 0.0;
    for (int s = 0; s < d.K; s++) hazard = FMAS(0.5 * Qe[s], FMAS(su[s], su[s], -sq[s]), hazard);
    for (int j = 0; j < d.nc; j++) { ST(sun + j * W, SPL(0.0)); ST(sqn + j * W, SPL(0.0)); }
#define MASCOT_B_FWD_L(NL, L) \
    fwdLineages<NL, STAGE>(d, L, clk, p, cur, ys, accb, nxt, scaleOut, coef, h6, Me, Qe, su, sun, sqn, cpb, hvb)
    MASCOT_B_LINEAGES(A, MASCOT_B_FWD_L);
#undef MASCOT_B_FWD_L
    return -hazard;
}

static void lineageSums(const Dims d, int A, const double* p, double* su, double* sq) {
    for (int j = 0; j < d.nc; j++) {
        vd s = SPL(0.0), q = SPL(0.0);
        const double* pj = p + j * W;
        for (int l = 0; l < A; l++) {
            vd v = LD(pj + (size_t) l * d.Kp);
            s = s + v;
            q = FMAV(v, v, q);
        }
        ST(su + j * W, s);
        ST(sq + j * W, q);
    }
}

// ====================================================================== adjoint (VJP of the RHS)
template <int NL>
struct VjpRows {
    const double* y[NL];
    const double* ar[NL];
    double* out[NL];
    double* hv[NL];
    double* cv[NL];
    double c[NL], r[NL], b[NL], cc[NL];
};

// One pass over the rows s < K does both the M^T matvec v_i[source chunks cb..cb+NB) = 0 + sum_{sink < K}
// ar_i[sink] M^T[sink][.] and the gram rank-NL update gram[s][sink chunks cb..cb+NB) += cys_i[s] ar_i[.] (i in order).
// Epilogue: cc_i += y_i . v_i (in order), out_i = c_i v_i + (ar_i (r_i - hv_i) + b_i hv_i), cv_i = y_i (b_i - ar_i),
// cSums += cv_i (lineage order).
template <int NL, int NB>
static MASCOT_B_INLINE void vjpGramBlock(const Dims d, int cb, const double* MTe, VjpRows<NL>& R,
                                         const double* const* cys, double* gram, double* cSums) {
    const int K = d.K, Kp = d.Kp;
    vd acc[NL][NB], a[NL][NB];
    for (int j = 0; j < NB; j++)
        for (int i = 0; i < NL; i++) { acc[i][j] = SPL(0.0); a[i][j] = LD(R.ar[i] + (cb + j) * W); }
    const double* m = MTe + cb * W;
    double* g = gram + cb * W;
    for (int s = 0; s < K; s++, m += Kp, g += Kp) {
        double as[NL], cs[NL];
        for (int i = 0; i < NL; i++) { as[i] = R.ar[i][s]; cs[i] = cys[i][s]; }
        for (int j = 0; j < NB; j++) {
            vd mv = LD(m + j * W);
            vd gv = LD(g + j * W);
            for (int i = 0; i < NL; i++) {
                acc[i][j] = FMAV(SPL(as[i]), mv, acc[i][j]);
                gv = FMAV(SPL(cs[i]), a[i][j], gv);
            }
            ST(g + j * W, gv);
        }
    }
    for (int j = 0; j < NB; j++) {
        const int o = (cb + j) * W;
        vd cs = LD(cSums + o);
        for (int i = 0; i < NL; i++) {
            vd yv = LD(R.y[i] + o), up = LD(R.ar[i] + o), hv = LD(R.hv[i] + o);
            addLanes(R.cc[i], yv * acc[i][j]);
            ST(R.out[i] + o, FMAV(SPL(R.c[i]), acc[i][j], FMAV(up, R.r[i] - hv, R.b[i] * hv)));
            vd cv = yv * (R.b[i] - up);
            ST(R.cv[i] + o, cv);
            cs = cs + cv;
        }
        ST(cSums + o, cs);
    }
}

struct VjpBufs {
    double* out;    // A x Kp
    double* hv;     // A x Kp
    double* cv;     // A x Kp
    double* cys;    // 2 x Kp
    double* cSums;  // Kp
    double* gradQ;  // Kp
    double* gram;   // K x Kp, all zero between calls
};

// Pass A for lineages [l, l + NL): hv, r, b, then the fused M^T matvec + gram update and the clock gradient.
template <int NL>
static MASCOT_B_INLINE void vjpLineagesA(const Dims d, int l, const double* y, const double* ar, const double* su,
                                         const double* Qe, const double* MTe, const int* ids, const double* clk,
                                         double* clockGradient, const VjpBufs& B) {
    const int Kp = d.Kp, nc = d.nc;
    VjpRows<NL> R;
    const double* cys[NL];
    for (int i = 0; i < NL; i++) {
        size_t o = (size_t) (l + i) * Kp;
        R.y[i] = y + o;
        R.ar[i] = ar + o;
        R.out[i] = B.out + o;
        R.hv[i] = B.hv + o;
        R.cv[i] = B.cv + o;
        R.c[i] = clk[l + i];
        R.r[i] = 0.0;
        R.b[i] = 0.0;
        R.cc[i] = 0.0;
        cys[i] = B.cys + i * Kp;
    }
    // cys = c y; hv = (sums - y) Q; r = sum y hv; b = sum ar y (sequential)
    for (int j = 0; j < nc; j++) {
        const int o = j * W;
        vd sj = LD(su + o), qj = LD(Qe + o);
        for (int i = 0; i < NL; i++) {
            vd yv = LD(R.y[i] + o);
            ST(B.cys + i * Kp + o, R.c[i] * yv);
            vd hv = (sj - yv) * qj;
            ST(R.hv[i] + o, hv);
            addLanes(R.r[i], yv * hv);
            addLanes(R.b[i], LD(R.ar[i] + o) * yv);
        }
    }
#define MASCOT_B_VG_CALL(NB, CB) vjpGramBlock<NL, NB>(d, CB, MTe, R, cys, B.gram, B.cSums)
    MASCOT_B_BLOCKS(NBA, nc, MASCOT_B_VG_CALL);
#undef MASCOT_B_VG_CALL
    if (clockGradient)
        for (int i = 0; i < NL; i++) clockGradient[ids[l + i]] += R.cc[i];
}

// Pass D for lineages [l, l + NL), fused with the RK4 adjoint combination of this stage.
template <int NL, int STAGE>
static MASCOT_B_INLINE void vjpLineagesD(const Dims d, int l, const double* y, const double* su, const double* Qe,
                                         double ellAdj, double* x, double* ay0, double* ak, const VjpBufs& B, double h,
                                         double hh, double h3, double h6) {
    const int Kp = d.Kp, nc = d.nc;
    const double negEll = -ellAdj;  // (-e) * hv is exactly -(e * hv): q (cs - cv) - e hv = FMA(q, cs - cv, (-e) hv)
    for (int j = 0; j < nc; j++) {
        const int oj = j * W;
        vd sj = LD(su + oj), qj = LD(Qe + oj), csj = LD(B.cSums + oj);
        vd g = LD(B.gradQ + oj);
        for (int i = 0; i < NL; i++) {
            size_t o = (size_t) (l + i) * Kp + oj;
            vd cv = LD(B.cv + o), hv = LD(B.hv + o), yv = LD(y + o);
            vd v = LD(B.out + o) + FMAV(qj, csj - cv, negEll * hv);
            g = FMAV(cv, sj - yv, g);
            vd xv = LD(x + o);
            if (STAGE == 4) { ST(ay0 + o, xv + v); ST(ak + o, FMAV(SPL(h3), xv, h * v)); }
            else if (STAGE == 3) { ST(ay0 + o, LD(ay0 + o) + v); ST(ak + o, FMAV(SPL(h3), xv, hh * v)); }
            else if (STAGE == 2) { ST(ay0 + o, LD(ay0 + o) + v); ST(ak + o, FMAV(SPL(h6), xv, hh * v)); }
            else { ST(x + o, LD(ay0 + o) + v); }
        }
        ST(B.gradQ + oj, g);
    }
}

// VJP of the RHS at y (stage STAGE) with upstream ak (in: adjoint of k_STAGE; out: adjoint of k_{STAGE-1}, or for
// STAGE 1 the step-start adjoint written to x).
template <int STAGE>
static MASCOT_B_INLINE void rhsVjp(const Dims d, int A, const double* y, const double* su, const double* sq,
                                   const double* MTe, const double* Qe, double ellAdj, double* x, double* ay0,
                                   double* ak, const VjpBufs& B, double* gTheta, const int* ids, const double* clk,
                                   double* clockGradient, double h, double hh, double h3, double h6) {
    const int K = d.K, Kp = d.Kp, nc = d.nc;
    for (int j = 0; j < nc; j++) ST(B.cSums + j * W, SPL(0.0));
#define MASCOT_B_VA_L(NL, L) vjpLineagesA<NL>(d, L, y, ak, su, Qe, MTe, ids, clk, clockGradient, B)
    MASCOT_B_LINEAGES(A, MASCOT_B_VA_L);
#undef MASCOT_B_VA_L
    // gram -> rate gradient; gram re-zeroed for the next call
    for (int src = 0; src < K; src++) {
        double* __restrict gr = B.gram + (size_t) src * Kp;
        double* __restrict gt = gTheta + (size_t) src * (K - 1);
        const double dg = gr[src];
        for (int sink = 0; sink < src; sink++) gt[sink] += gr[sink] - dg;
        for (int sink = src + 1; sink < K; sink++) gt[sink - 1] += gr[sink] - dg;
        for (int j = 0; j < nc; j++) ST(gr + j * W, SPL(0.0));
    }
    for (int j = 0; j < nc; j++) {
        vd s = LD(su + j * W), q = LD(sq + j * W);
        vd pairSum = 0.5 * FMAV(s, s, -q);
        ST(B.gradQ + j * W, -ellAdj * pairSum);
    }
#define MASCOT_B_VD_L(NL, L) vjpLineagesD<NL, STAGE>(d, L, y, su, Qe, ellAdj, x, ay0, ak, B, h, hh, h3, h6)
    MASCOT_B_LINEAGES(A, MASCOT_B_VD_L);
#undef MASCOT_B_VD_L
    double* gEta = gTheta + K * (K - 1);
    for (int s = 0; s < K; s++) gEta[s] = FMAS(-Qe[s], B.gradQ[s], gEta[s]);
}

// VJP of the per-lineage projection y / (1'y) at step end `post`; padded lanes masked to 0. Also seeds ak = h6 x.
static void projectionVjp(const Dims d, int A, double* a, const double* post, const double* scale, double* ak,
                          double h6, vi lastMask) {
    const int Kp = d.Kp, nc = d.nc;
    for (int l = 0; l < A; l++) {
        double* al = a + (size_t) l * Kp;
        const double* pl = post + (size_t) l * Kp;
        double* kl = ak + (size_t) l * Kp;
        double dot = 0.0;
        for (int j = 0; j < nc; j++) addLanes(dot, LD(al + j * W) * LD(pl + j * W));
        const double scl = scale[l];
        for (int j = 0; j < nc - 1; j++) {
            vd xv = (LD(al + j * W) - dot) / scl;
            ST(al + j * W, xv);
            ST(kl + j * W, h6 * xv);
        }
        const int j = nc - 1;
        vd xv = maskv((LD(al + j * W) - dot) / scl, lastMask);
        ST(al + j * W, xv);
        ST(kl + j * W, h6 * xv);
    }
}

// ====================================================================== driver
class KernelB final : public mascot::Kernel {
public:
    int forward(const mascot::Problem& problem, double* logL) override {
        setup(problem);
        tape = false;
        const int status = sweep();
        if (status == mascot::OK) *logL = ell;
        return status;
    }

    int forwardAndReverse(const mascot::Problem& problem, double* logL, double* thetaGradient, double* clockGradient,
                          double* ancestralScores) override {
        setup(problem);
        const int K = d.K;
        std::fill(thetaGradient, thetaGradient + (size_t) E * (K * (K - 1) + K), 0.0);
        if (clockGradient) std::fill(clockGradient, clockGradient + N, 0.0);
        if (ancestralScores)
            std::fill(ancestralScores, ancestralScores + (size_t) N * K, std::numeric_limits<double>::quiet_NaN());
        tape = true;
        intervals.clear();
        coals.clear();
        ops.clear();
        y0Used = yUsed = scaleUsed = sumUsed = coalUsed = idUsed = 0;
        const int status = sweep();
        if (status != mascot::OK) return status;
        reverse(thetaGradient, clockGradient, ancestralScores);
        *logL = ell;
        return mascot::OK;
    }

    int lastStepCounts(int* steps, int capacity, int* count) const override {
        *count = segments;
        for (int i = 0; i < std::min(segments, capacity); i++) steps[i] = stepLog[i];
        return capacity < segments ? mascot::ERROR_ARGUMENT : mascot::OK;
    }

    char name() const override { return 'B'; }
    int simdWidth() const override { return W; }
    bool fusedMultiplyAdd() const override {
#ifdef MASCOT_FP_FMA
        return true;
#else
        return false;
#endif
    }

private:
    const mascot::Problem* inp = nullptr;
    Dims d{0, 0, 0};
    int E = 0, N = 0;
    bool tape = false, check = false;
    vi lastMask;
    ABuf M, MT, Q;
    ABuf y, yOut, ys, accb, cpb, hvb, sumsb, rowb;
    std::vector<double> clock;
    std::vector<int> activeIds, slotOf;
    int active = 0;
    double ell = 0.0;
    int epochCursor = 0;
    std::vector<int> stepLog;  // step count of each segment of the last evaluation
    int segments = 0;

    struct IntervalRec { int steps, active, epoch; double h; size_t y0Off, yOff, scaleOff, sumOff, idOff; };
    struct CoalRec { int epoch, first, second, parentSlot, movedFrom, movedTo, parentId; size_t pOff; };
    std::vector<IntervalRec> intervals;
    std::vector<CoalRec> coals;
    std::vector<int> ops;  // >= 0: interval, -1: sample, <= -2: coalescent -(index + 2)
    ABuf tY0, tY2, tY3, tY4, tScale, tSums, tCoal;
    size_t y0Used = 0, yUsed = 0, scaleUsed = 0, sumUsed = 0, coalUsed = 0, idUsed = 0;
    std::vector<int> idTape;
    std::vector<double> clockTape;

    ABuf ra, rb, aY0, aK, outb, cvb, hvr, cSums, gradQ, gram, cysb;

    void setup(const mascot::Problem& in) {
        inp = &in;
        const int K = in.K;
        const int Kp = (K + W - 1) / W * W;
        d = Dims{K, Kp, Kp / W};
        E = in.epochCount;
        N = in.lineageCount;
        check = in.checkProbabilities;
        lastMask = laneMask(K - (d.nc - 1) * W);
        size_t nk = (size_t) N * Kp;
        y.reserve(nk); yOut.reserve(nk); ys.reserve(nk); accb.reserve(nk);
        cpb.reserve(2 * (size_t) Kp); hvb.reserve(2 * (size_t) Kp);
        sumsb.reserve(4 * (size_t) Kp); rowb.reserve(Kp);
        M.reserve((size_t) E * K * Kp); MT.reserve((size_t) E * K * Kp); Q.reserve((size_t) E * Kp);
        if (clock.size() < (size_t) N) { clock.resize(N); activeIds.resize(N); slotOf.resize(N); }
        copyRates();
    }

    // M (row = source), M^T and Q of every epoch into the padded layout; padded entries are 0.
    void copyRates() {
        const mascot::Problem& in = *inp;
        const int K = d.K, Kp = d.Kp;
        for (int e = 0; e < E; e++) {
            const double* src = in.migration + (size_t) e * K * K;
            double* Me = M.data() + (size_t) e * K * Kp;
            double* MTe = MT.data() + (size_t) e * K * Kp;
            std::memset(Me, 0, sizeof(double) * K * Kp);
            std::memset(MTe, 0, sizeof(double) * K * Kp);
            for (int source = 0; source < K; source++)
                for (int sink = 0; sink < K; sink++) {
                    const double m = src[source * K + sink];
                    Me[source * Kp + sink] = m;
                    MTe[sink * Kp + source] = m;
                }
            const double* q = in.inversePopulation + (size_t) e * K;
            double* Qe = Q.data() + (size_t) e * Kp;
            for (int s = 0; s < Kp; s++) Qe[s] = s < K ? q[s] : 0.0;
        }
    }

    int epochAt(double t) {
        const double* b = inp->boundaries;
        int nb = E + 1;
        while (epochCursor + 1 < nb && t >= b[epochCursor + 1] - TOL) epochCursor++;
        return std::min(epochCursor, E - 1);
    }

    double nextBoundaryAfter(double start, double stop) {
        const double* b = inp->boundaries;
        int nb = E + 1;
        for (int i = epochCursor + 1; i < nb; i++) {
            if (b[i] > start + TOL) return b[i] <= stop + TOL ? std::min(b[i], stop) : stop;
        }
        return stop;
    }

    // MascotForwardEvaluator.checkProbabilities on the active lineages (real states only).
    bool probabilitiesValid() const {
        const int K = d.K, Kp = d.Kp;
        const double* yy = y.data();
        for (int l = 0; l < active; l++) {
            const double* row = yy + (size_t) l * Kp;
            double sum = 0.0;
            for (int s = 0; s < K; s++) {
                const double p = row[s];
                if (p < -1.0e-8 || !std::isfinite(p)) return false;
                sum += p;
            }
            if (std::fabs(sum - 1.0) > 1.0e-6) return false;
        }
        return true;
    }

    // MascotStepRule.stiffness of the current state (the segment's start) with the segment's epoch rates:
    // lambda = clock * maxOut + maxLoad, real states only, a plain multiply and add in both FP modes.
    double stiffness(int epoch) {
        const int K = d.K, Kp = d.Kp, nc = d.nc, A = active;
        double c = 1.0;
        if (inp->branchRates) {
            c = 0.0;
            for (int l = 0; l < A; l++) c = javaMax(c, clock[l]);
        }
        const double* Me = M.data() + (size_t) epoch * K * Kp;
        double maxOut = 0.0;
        for (int k = 0; k < K; k++) maxOut = javaMax(maxOut, -Me[(size_t) k * Kp + k]);
        // load_k = sum over lineages in slot order from 0.0, lane by lane
        const double* yy = y.data();
        double* load = rowb.data();
        for (int j = 0; j < nc; j++) {
            vd s = SPL(0.0);
            for (int l = 0; l < A; l++) s = s + LD(yy + (size_t) l * Kp + j * W);
            ST(load + j * W, s);
        }
        const double* Qe = Q.data() + (size_t) epoch * Kp;
        double maxLoad = 0.0;
        for (int k = 0; k < K; k++) maxLoad = javaMax(maxLoad, Qe[k] * load[k]);
        return c * maxOut + maxLoad;
    }

    // MascotStepRule.steps for a segment of length len (end - start) in the given epoch; recorded for lastStepCounts.
    int segmentSteps(double len, int epoch) {
        const mascot::Problem& in = *inp;
        int n;
        if (in.stepRuleKind != mascot::STEPS_ADAPTIVE) {
            n = std::max(1, (int) std::ceil(len / in.maxStep));
        } else {
            const double target = len * stiffness(epoch) / in.tau;
            n = target > 0.0 ? (int) std::ceil(javaMin(target, LARGEST_STEP_COUNT)) : 1;
            n = std::max(1, n);
            if (in.maxStep > 0.0) n = std::max(n, (int) std::ceil(javaMin(len / in.maxStep, LARGEST_STEP_COUNT)));
            if (in.minStep > 0.0)
                n = std::min(n, std::max(1, (int) std::floor(javaMin(len / in.minStep, LARGEST_STEP_COUNT))));
            if (in.maxStepsPerSegment > 0) n = std::min(n, in.maxStepsPerSegment);
        }
        if (segments == (int) stepLog.size()) stepLog.resize(std::max((size_t) 64, 2 * stepLog.size()));
        stepLog[segments++] = n;
        return n;
    }

    int sweep() {
        const mascot::Problem& in = *inp;
        active = 0;
        ell = 0.0;
        epochCursor = 0;
        segments = 0;
        sample(in.initialSampleNode);
        double t = in.initialTime;
        for (int i = 0; i < in.intervalCount; i++) {
            double end = t + in.intervalLengths[i];
            while (end > t + TOL) {
                double segEnd = nextBoundaryAfter(t, end);
                int epoch = epochAt(t + TOL);
                if (tape) integrateTape(t, segEnd, epoch); else integrate(t, segEnd, epoch);
                t = segEnd;
                if (check && !probabilitiesValid()) return mascot::ERROR_PROBABILITIES;
            }
            t = end;
            if (in.eventTypes[i] == mascot::SAMPLE) {
                sample(in.sampleNodes[i]);
            } else if (!coalesce(in.child1[i], in.child2[i], in.parent[i], epochAt(t))) {
                return mascot::ERROR_COALESCENT_RATE;
            }
            if (check && !probabilitiesValid()) return mascot::ERROR_PROBABILITIES;
        }
        return mascot::OK;
    }

    void sample(int node) {
        const mascot::Problem& in = *inp;
        const int K = d.K, Kp = d.Kp;
        int slot = active++;
        double* row = y.data() + (size_t) slot * Kp;
        std::memcpy(row, in.tipPartials + (size_t) node * K, sizeof(double) * K);
        for (int s = K; s < Kp; s++) row[s] = 0.0;
        activeIds[slot] = node;
        slotOf[node] = slot;
        clock[slot] = in.branchRates ? in.branchRates[node] : 1.0;
        if (tape) ops.push_back(-1);
    }

    bool coalesce(int child1, int child2, int parent, int epoch) {
        const mascot::Problem& in = *inp;
        const int K = d.K, Kp = d.Kp;
        double* yy = y.data();
        int first = slotOf[child1], second = slotOf[child2];
        const double* Qe = Q.data() + (size_t) epoch * Kp;
        double* v = rowb.data();
        double lambda = 0.0;
        for (int s = 0; s < K; s++) {
            v[s] = yy[(size_t) first * Kp + s] * yy[(size_t) second * Kp + s] * Qe[s];
            lambda += v[s];
        }
        if (!(lambda > 0.0) || !std::isfinite(lambda)) return false;
        size_t pOff = 0;
        if (tape) {
            pOff = take(tCoal, coalUsed, 3 * (size_t) K);  // p1, p2, parent
            double* rec = tCoal.data() + pOff;
            for (int s = 0; s < K; s++) {
                rec[s] = yy[(size_t) first * Kp + s];
                rec[K + s] = yy[(size_t) second * Kp + s];
            }
        }
        int last = active - 1;
        int parentSlot = first == last ? second : first;
        int removed = parentSlot == first ? second : first;
        int movedFrom = -1, movedTo = -1;
        if (removed != last) {
            movedFrom = last;
            movedTo = removed;
            int moved = activeIds[last];
            activeIds[removed] = moved;
            std::memcpy(yy + (size_t) removed * Kp, yy + (size_t) last * Kp, sizeof(double) * Kp);
            clock[removed] = clock[last];
            slotOf[moved] = removed;
        }
        double* prow = yy + (size_t) parentSlot * Kp;
        for (int s = 0; s < K; s++) prow[s] = v[s] / lambda;
        for (int s = K; s < Kp; s++) prow[s] = 0.0;
        if (tape) std::memcpy(tCoal.data() + pOff + 2 * K, prow, sizeof(double) * K);
        activeIds[parentSlot] = parent;
        slotOf[parent] = parentSlot;
        clock[parentSlot] = in.branchRates ? in.branchRates[parent] : 1.0;
        active--;
        ell += std::log(lambda);
        if (tape) {
            coals.push_back({epoch, first, second, parentSlot, movedFrom, movedTo, parent, pOff});
            ops.push_back(-(int) (coals.size() - 1) - 2);
        }
        return true;
    }

    void integrate(double start, double end, int epoch) {
        const Dims dd = d;
        const int Kp = dd.Kp;
        const int steps = segmentSteps(end - start, epoch);
        double h = (end - start) / steps;
        const int A = active;
        size_t n = (size_t) A * Kp;
        const double* Me = M.data() + (size_t) epoch * dd.K * Kp;
        const double* Qe = Q.data() + (size_t) epoch * Kp;
        const double* clk = clock.data();
        double* cur = y.data();
        double* nxt = yOut.data();
        double* yst = ys.data();
        double* acc = accb.data();
        double* cp = cpb.data();
        double* hv = hvb.data();
        double* sA = sumsb.data();
        double* qA = sA + Kp;
        double* sB = sA + 2 * Kp;
        double* qB = sA + 3 * Kp;
        double hh = 0.5 * h;
        double h6 = h / 6.0;
        lineageSums(dd, A, cur, sA, qA);
        for (int step = 0; step < steps; step++) {
            double e1 = fwdStage<1>(dd, A, clk, cur, cur, yst, acc, nullptr, nullptr, hh, h6, Me, Qe, sA, qA, sB, qB,
                                    cp, hv);
            double e2 = fwdStage<2>(dd, A, clk, yst, cur, yst, acc, nullptr, nullptr, hh, h6, Me, Qe, sB, qB, sA, qA,
                                    cp, hv);
            double e3 = fwdStage<3>(dd, A, clk, yst, cur, yst, acc, nullptr, nullptr, h, h6, Me, Qe, sA, qA, sB, qB,
                                    cp, hv);
            double e4 = fwdStage<4>(dd, A, clk, yst, cur, nullptr, acc, nxt, nullptr, 0.0, h6, Me, Qe, sB, qB, sA, qA,
                                    cp, hv);
            ell = FMAS(h6, e1 + 2.0 * e2 + 2.0 * e3 + e4, ell);
            std::swap(cur, nxt);
        }
        if (cur != y.data()) std::memcpy(y.data(), cur, sizeof(double) * n);
    }

    void integrateTape(double start, double end, int epoch) {
        const Dims dd = d;
        const int Kp = dd.Kp;
        const int steps = segmentSteps(end - start, epoch);
        double h = (end - start) / steps;
        const int A = active;
        size_t n = (size_t) A * Kp;
        size_t y0Off = take(tY0, y0Used, (size_t) (steps + 1) * n);
        size_t yOff = yUsed;
        tY2.reserve(yUsed + (size_t) steps * n, yUsed);
        tY3.reserve(yUsed + (size_t) steps * n, yUsed);
        tY4.reserve(yUsed + (size_t) steps * n, yUsed);
        yUsed += (size_t) steps * n;
        size_t scaleOff = take(tScale, scaleUsed, (size_t) steps * A);
        // 8 Kp per step: (sums, sq) of the four stage inputs; + 2 Kp for the sums of the final state (unused)
        size_t sumOff = take(tSums, sumUsed, (size_t) steps * 8 * Kp + 2 * Kp);
        size_t idOff = idUsed;
        if (idTape.size() < idOff + A) {
            idTape.resize(std::max(idTape.size() * 2, idOff + A));
            clockTape.resize(idTape.size());
        }
        idUsed += A;
        for (int l = 0; l < A; l++) {
            idTape[idOff + l] = activeIds[l];
            clockTape[idOff + l] = clock[l];
        }
        intervals.push_back({steps, A, epoch, h, y0Off, yOff, scaleOff, sumOff, idOff});
        ops.push_back((int) intervals.size() - 1);

        const double* Me = M.data() + (size_t) epoch * dd.K * Kp;
        const double* Qe = Q.data() + (size_t) epoch * Kp;
        const double* clk = clock.data();
        double* y0 = tY0.data() + y0Off;
        double* y2 = tY2.data() + yOff;
        double* y3 = tY3.data() + yOff;
        double* y4 = tY4.data() + yOff;
        double* sc = tScale.data() + scaleOff;
        double* ss = tSums.data() + sumOff;
        double* acc = accb.data();
        double* cp = cpb.data();
        double* hv = hvb.data();
        std::memcpy(y0, y.data(), sizeof(double) * n);
        double hh = 0.5 * h;
        double h6 = h / 6.0;
        lineageSums(dd, A, y0, ss, ss + Kp);
        for (int step = 0; step < steps; step++) {
            const double* cur = y0 + (size_t) step * n;
            double* nx = y0 + (size_t) (step + 1) * n;
            double* s2 = y2 + (size_t) step * n;
            double* s3 = y3 + (size_t) step * n;
            double* s4 = y4 + (size_t) step * n;
            double* st = ss + (size_t) step * 8 * Kp;
            double e1 = fwdStage<1>(dd, A, clk, cur, cur, s2, acc, nullptr, nullptr, hh, h6, Me, Qe, st, st + Kp,
                                    st + 2 * Kp, st + 3 * Kp, cp, hv);
            double e2 = fwdStage<2>(dd, A, clk, s2, cur, s3, acc, nullptr, nullptr, hh, h6, Me, Qe, st + 2 * Kp,
                                    st + 3 * Kp, st + 4 * Kp, st + 5 * Kp, cp, hv);
            double e3 = fwdStage<3>(dd, A, clk, s3, cur, s4, acc, nullptr, nullptr, h, h6, Me, Qe, st + 4 * Kp,
                                    st + 5 * Kp, st + 6 * Kp, st + 7 * Kp, cp, hv);
            double e4 = fwdStage<4>(dd, A, clk, s4, cur, nullptr, acc, nx, sc + (size_t) step * A, 0.0, h6, Me, Qe,
                                    st + 6 * Kp, st + 7 * Kp, st + 8 * Kp, st + 9 * Kp, cp, hv);
            ell = FMAS(h6, e1 + 2.0 * e2 + 2.0 * e3 + e4, ell);
        }
        std::memcpy(y.data(), y0 + (size_t) steps * n, sizeof(double) * n);
    }

    // Outputs were zeroed (gradients) and NaN-filled (ancestral scores) by forwardAndReverse.
    void reverse(double* gradient, double* clockGradient, double* ancestral) {
        const int K = d.K, Kp = d.Kp;
        size_t nk = (size_t) N * Kp;
        ra.reserve(nk); rb.reserve(nk); aY0.reserve(nk); aK.reserve(nk); outb.reserve(nk); cvb.reserve(nk);
        hvr.reserve(nk);
        cSums.reserve(Kp); gradQ.reserve(Kp); cysb.reserve(2 * (size_t) Kp);
        gram.reserve((size_t) K * Kp);
        std::memset(gram.data(), 0, sizeof(double) * K * Kp);
        int act = active;
        double* a = ra.data();
        double* b = rb.data();
        std::fill(a, a + (size_t) act * Kp, 0.0);
        const double aell = 1.0;
        for (int op = (int) ops.size() - 1; op >= 0; op--) {
            int code = ops[op];
            if (code == -1) {
                act--;
            } else if (code >= 0) {
                reverseInterval(intervals[code], a, aell, gradient, clockGradient);
            } else {
                reverseCoalescent(coals[-code - 2], a, b, act, aell, gradient, ancestral);
                std::swap(a, b);
                act++;
            }
        }
    }

    void reverseInterval(const IntervalRec& r, double* a, double aell, double* gradient, double* clockGradient) {
        const Dims dd = d;
        const int K = dd.K, Kp = dd.Kp;
        const int A = r.active;
        const size_t n = (size_t) A * Kp;
        const double h = r.h;
        const double* y0 = tY0.data() + r.y0Off;
        const double* y2 = tY2.data() + r.yOff;
        const double* y3 = tY3.data() + r.yOff;
        const double* y4 = tY4.data() + r.yOff;
        const double* sc = tScale.data() + r.scaleOff;
        const double* ss = tSums.data() + r.sumOff;
        const int* ids = idTape.data() + r.idOff;
        const double* clk = clockTape.data() + r.idOff;
        const double* MTe = MT.data() + (size_t) r.epoch * K * Kp;
        const double* Qe = Q.data() + (size_t) r.epoch * Kp;
        double* gTheta = gradient + (size_t) r.epoch * (K * (K - 1) + K);
        double* ay0 = aY0.data();
        double* ak = aK.data();
        const VjpBufs B{outb.data(), hvr.data(), cvb.data(), cysb.data(), cSums.data(), gradQ.data(), gram.data()};
        const vi mask = lastMask;
        const double h6 = h / 6.0, h3 = h / 3.0, hh = 0.5 * h;
        const double e1 = h6 * aell, e2 = h3 * aell, e3 = h3 * aell, e4 = h6 * aell;
        for (int step = r.steps - 1; step >= 0; step--) {
            projectionVjp(dd, A, a, y0 + (size_t) (step + 1) * n, sc + (size_t) step * A, ak, h6, mask);
            const size_t off = (size_t) step * n;
            const double* st = ss + (size_t) step * 8 * Kp;
            rhsVjp<4>(dd, A, y4 + off, st + 6 * Kp, st + 7 * Kp, MTe, Qe, e4, a, ay0, ak, B, gTheta, ids, clk,
                      clockGradient, h, hh, h3, h6);
            rhsVjp<3>(dd, A, y3 + off, st + 4 * Kp, st + 5 * Kp, MTe, Qe, e3, a, ay0, ak, B, gTheta, ids, clk,
                      clockGradient, h, hh, h3, h6);
            rhsVjp<2>(dd, A, y2 + off, st + 2 * Kp, st + 3 * Kp, MTe, Qe, e2, a, ay0, ak, B, gTheta, ids, clk,
                      clockGradient, h, hh, h3, h6);
            rhsVjp<1>(dd, A, y0 + off, st, st + Kp, MTe, Qe, e1, a, ay0, ak, B, gTheta, ids, clk, clockGradient, h,
                      hh, h3, h6);
        }
    }

    // MascotAdjointEvaluator.reverseCoalescentInto: `after` has the parent row, `before` receives both child rows.
    // The parent's ancestral scores are written to its row of `ancestral` (if not null), real states only.
    void reverseCoalescent(const CoalRec& r, const double* after, double* before, int afterCount, double aell,
                           double* gradient, double* ancestral) {
        const int K = d.K, Kp = d.Kp;
        const int beforeCount = afterCount + 1;
        const double* p1 = tCoal.data() + r.pOff;
        const double* p2 = p1 + K;
        const double* pp = p2 + K;
        for (int bi = 0; bi < beforeCount; bi++) {
            if (bi == r.first || bi == r.second) continue;
            int ai = bi == r.movedFrom ? r.movedTo : bi;
            std::memcpy(before + (size_t) bi * Kp, after + (size_t) ai * Kp, sizeof(double) * Kp);
        }
        const double* ap = after + (size_t) r.parentSlot * Kp;
        double dot = 0.0;
        for (int s = 0; s < K; s++) dot = FMAS(ap[s], pp[s], dot);
        double* gEta = gradient + (size_t) r.epoch * (K * (K - 1) + K) + K * (K - 1);
        double* b1 = before + (size_t) r.first * Kp;
        double* b2 = before + (size_t) r.second * Kp;
        double* score = ancestral ? ancestral + (size_t) r.parentId * K : nullptr;
        for (int s = 0; s < K; s++) {
            double q1 = std::max(p1[s], EPS);
            double q2 = std::max(p2[s], EPS);
            double centered = ap[s] - dot;
            double nodeStateScore = pp[s] * (centered + aell);
            b1[s] = nodeStateScore / q1;
            b2[s] = nodeStateScore / q2;
            gEta[s] -= nodeStateScore;
            if (score) score[s] = nodeStateScore;
        }
        for (int s = K; s < Kp; s++) { b1[s] = 0.0; b2[s] = 0.0; }
    }
};

}  // namespace MASCOT_B_NS

#undef MASCOT_B_LINEAGES
#undef MASCOT_B_BLOCKS
#undef MASCOT_B_CL
#undef MASCOT_B_INLINE

#endif
