/*
 * bastav5test: the 9-integer BASTA _v5 functions (BastaOperation_v5) and the population-size buffers on CPU
 * instances, in double and single precision, with 1 and 4 BASTA threads, for 3, 4 and 5 demes. The problem is a
 * structured coalescent of 40 serially sampled tips; the population sizes and integrals are in BEAST's layout
 * (R = K + n*K, interval j at offset K + j*K, K demes, n intervals).
 * T1 constant sizes through _v5 against the 8-integer functions with beagleSetStateFrequencies(N): every partial
 *    and coalescent probability bit for bit, the log likelihood within 1e-12 (double) or 1e-5 (single) and the
 *    legacy gradient within 1e-10 or 1e-4 (relative to its largest entry)
 * T2 8-integer, _v5, 8-integer, _v5 on one instance: each result bit for bit as its first run
 * T3 the same values at shuffled, unaligned offsets with NaN everywhere else: the same results bit for bit
 * T4 sizes that change by interval: every coalescent partial is acc1 * acc2 / N[p + i], normalised, and the log
 *    likelihood is the reduction recomputed here from the partials and the integrals
 * T5 buffer flips (beagleSetBastaPopulationSizesBuffer, beagleSetCurrentPopulationSizeBuffer) give the results of
 *    their buffer bit for bit; beagleSetBastaPopulationSizes with another R regrows the current buffer
 * T6 the return codes of rejected calls; after each, the 8-integer and _v5 calls repeat their results bit for bit
 * It is C, so it also compiles beagle.h, and BastaOperation_v5, as C.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libhmsbeagle/beagle.h"

enum { TIPS = 40, MAX_INTERVALS = 2 * TIPS, MAX_OPS = 2 * TIPS * TIPS, MAX_BUFFERS = TIPS + 3 * MAX_OPS };
enum { EXTRA_INTERVALS = 2 }; /* intervalNumbers that no operation uses: bufferLength = intervals + EXTRA_INTERVALS */

static int failures = 0;
static char name[160];

static void check(int ok, const char* what) {
    if (!ok) {
        fprintf(stderr, "FAIL %s: %s\n", name, what);
        failures++;
    }
}

static void checkCode(int rc, int expected, const char* what) {
    if (rc != expected) {
        fprintf(stderr, "FAIL %s: %s returned %d, expected %d\n", name, what, rc, expected);
        failures++;
    }
}

/* ---- the problem ---- */

static uint64_t rngState;
static double unif(void) {
    uint64_t z = (rngState += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return (double) (z >> 11) * (1.0 / 9007199254740992.0);
}

static int K;                               /* demes (states) */
static int ops8[MAX_OPS * 8], nOps;         /* 8 integers per operation */
static BastaOperation_v5 ops9[MAX_OPS];     /* the same operations with BEAST's offsets */
static int starts[MAX_INTERVALS + 1], nIntervals, bufferCount;
static double lengths[MAX_INTERVALS];
static int tipState[TIPS];
static double sizes[8];                     /* N_k */
static double* matrices;                    /* nIntervals * K * K */
static int matrixIndices[MAX_INTERVALS];

static void addOperation(int dest, int c1, int m1, int c2, int m2, int a1, int a2, int interval) {
    int* o = ops8 + 8 * nOps++;
    o[0] = dest; o[1] = c1; o[2] = m1; o[3] = c2; o[4] = m2; o[5] = a1; o[6] = a2; o[7] = interval;
}

/* basta_check's problem: tips sampled at 0 or later, coalescences at rate 1.5 per pair; every interval propagates
 * each active lineage through its matrix; a coalescence writes the parent and the two accumulation buffers */
static void simulate(void) {
    double sampleTime[TIPS];
    int sampleTip[TIPS], active[TIPS + 1], nActive = 0, next = TIPS, interval = 0, si = 1;
    double t = 0.0;
    for (int k = 0; k < TIPS; k++) {
        double s = k == 0 ? 0.0 : (unif() < 0.5 ? 0.0 : 0.6 * unif());
        int j = k; /* stable insertion by time */
        while (j > 0 && sampleTime[j - 1] > s) { sampleTime[j] = sampleTime[j - 1]; sampleTip[j] = sampleTip[j - 1]; j--; }
        sampleTime[j] = s; sampleTip[j] = k;
    }
    for (int k = 0; k < TIPS; k++) tipState[k] = (int) (unif() * K);
    nOps = 0;
    starts[0] = 0;
    active[nActive++] = sampleTip[0];
    while (si < TIPS || nActive > 1) {
        const int k = nActive;
        const double tc = k >= 2 ? t - log(1.0 - unif()) / (1.5 * k * (k - 1) / 2.0) : INFINITY;
        const double ts = si < TIPS ? sampleTime[si] : INFINITY;
        if (ts <= tc) {
            if (ts - t > 0.0) {
                for (int b = 0; b < nActive; b++) {
                    const int out = next++;
                    addOperation(out, active[b], interval, -1, -1, out, -1, interval);
                    active[b] = out;
                }
                lengths[interval] = ts - t;
                starts[++interval] = nOps;
            }
            active[nActive++] = sampleTip[si++];
            t = ts;
        } else {
            int a = (int) (unif() * k), b = (int) (unif() * (k - 1));
            if (b >= a) b++;
            const int parent = next++, acc1 = next++, acc2 = next++;
            addOperation(parent, active[a], interval, active[b], interval, acc1, acc2, interval);
            int rest = 0;
            for (int j = 0; j < k; j++) {
                if (j == a || j == b) continue;
                const int out = next++;
                addOperation(out, active[j], interval, -1, -1, out, -1, interval);
                active[rest++] = out;
            }
            active[rest++] = parent;
            nActive = rest;
            lengths[interval] = tc - t;
            starts[++interval] = nOps;
            t = tc;
        }
    }
    nIntervals = interval;
    bufferCount = next;

    for (int k2 = 0; k2 < K; k2++) sizes[k2] = 0.5 + 1.5 * unif();
    free(matrices);
    matrices = (double*) malloc(sizeof(double) * nIntervals * K * K);
    for (int j = 0; j < nIntervals; j++) { /* a stochastic matrix per interval */
        matrixIndices[j] = j;
        for (int r = 0; r < K; r++) {
            double* row = matrices + (j * K + r) * K, sum = 0.0;
            for (int c = 0; c < K; c++) sum += (row[c] = (r == c ? 2.0 : 0.0) + 0.1 + unif());
            for (int c = 0; c < K; c++) row[c] /= sum;
        }
    }
    for (int o = 0; o < nOps; o++) {
        memcpy(&ops9[o], ops8 + 8 * o, 8 * sizeof(int));
        ops9[o].populationSizeOffset = K + ops8[8 * o + 7] * K;
    }
}

/* BEAST's layout: block 0 holds the base sizes (integrals 0), interval j's block is at K + j*K */
static int layoutSize(void) { return K + nIntervals * K; }

static double piecewiseSize(int j, int k) { return sizes[k] * (0.5 + ((j * 7 + k * 3) % 11) / 10.0); }

static double* layout(int piecewise) {
    const int R = layoutSize();
    double* v = (double*) calloc(2 * R, sizeof(double));
    for (int k = 0; k < K; k++) v[k] = sizes[k];
    for (int j = 0; j < nIntervals; j++) {
        for (int k = 0; k < K; k++) {
            const double n = piecewise ? piecewiseSize(j, k) : sizes[k];
            v[K + j * K + k] = n;
            v[R + K + j * K + k] = lengths[j] / n;
        }
    }
    return v;
}

/* ---- one evaluation ---- */

typedef struct {
    double logL;
    double partials[MAX_BUFFERS * 8];
    double coalescent[MAX_INTERVALS + EXTRA_INTERVALS];
} Result;

static int readResult(int instance, Result* r) {
    int rc = BEAGLE_SUCCESS;
    for (int b = 0; b < bufferCount; b++) rc |= beagleGetPartials(instance, b, BEAGLE_OP_NONE, r->partials + b * K);
    rc |= beagleGetBastaBuffer(instance, 0, r->coalescent);
    return rc;
}

static int forward8(int instance, Result* r) {
    int rc = beagleUpdateBastaPartials(instance, (const BastaOperation*) ops8, nOps, starts, nIntervals + 1, 0, 0);
    if (rc == BEAGLE_SUCCESS)
        rc = beagleAccumulateBastaPartials(instance, (const BastaOperation*) ops8, nOps, starts, nIntervals + 1,
                                           lengths, 0, 0, &r->logL);
    return rc == BEAGLE_SUCCESS ? readResult(instance, r) : rc;
}

static int forward5(int instance, const BastaOperation_v5* ops, Result* r) {
    int rc = beagleUpdateBastaPartials_v5(instance, ops, nOps, starts, nIntervals + 1, 0, 0);
    if (rc == BEAGLE_SUCCESS)
        rc = beagleAccumulateBastaPartials_v5(instance, ops, nOps, starts, nIntervals + 1, NULL, 0, 0, &r->logL);
    return rc == BEAGLE_SUCCESS ? readResult(instance, r) : rc;
}

/* the legacy gradient, after a forward pass with the same operations */
static int gradient8(int instance, double* out) {
    int rc = beagleUpdateTransitionMatricesGrad(instance, matrixIndices, lengths, nIntervals);
    rc |= beagleUpdateBastaPartialsGrad(instance, (const BastaOperation*) ops8, nOps, starts, nIntervals + 1, 0, 0);
    rc |= beagleAccumulateBastaPartialsGrad(instance, (const BastaOperation*) ops8, nOps, starts, nIntervals + 1,
                                            lengths, 0, 0, 0, -1, -1, NULL, out);
    return rc;
}

static int gradient5(int instance, const BastaOperation_v5* ops, double* out) {
    int rc = beagleUpdateTransitionMatricesGrad(instance, matrixIndices, lengths, nIntervals);
    rc |= beagleUpdateBastaPartialsGrad_v5(instance, ops, nOps, starts, nIntervals + 1, 0, 0);
    rc |= beagleAccumulateBastaPartialsGrad_v5(instance, ops, nOps, starts, nIntervals + 1, NULL, 0, 0, 0, -1, -1,
                                               NULL, out);
    return rc;
}

static int sameResult(const Result* a, const Result* b) {
    return memcmp(&a->logL, &b->logL, sizeof(double)) == 0 &&
           memcmp(a->partials, b->partials, sizeof(double) * bufferCount * K) == 0 &&
           memcmp(a->coalescent, b->coalescent, sizeof(double) * (nIntervals + EXTRA_INTERVALS)) == 0;
}

static double relative(double a, double b) { return fabs(a - b) / fabs(b); }

static double gradientDifference(const double* a, const double* b) {
    double scale = 0.0, worst = 0.0;
    for (int i = 0; i < K * K; i++) scale = fmax(scale, fabs(b[i]));
    for (int i = 0; i < K * K; i++) worst = fmax(worst, fabs(a[i] - b[i]) / scale);
    return worst;
}

/* ---- instances ---- */

static int createInstance(int single, int threads, int allocate, int gradient) {
    BeagleInstanceDetails details;
    const long long flags = BEAGLE_FLAG_PROCESSOR_CPU |
                            (single ? BEAGLE_FLAG_PRECISION_SINGLE : BEAGLE_FLAG_PRECISION_DOUBLE);
    const int instance = beagleCreateInstance(0, bufferCount + 1, 0, K, 1, 1, nIntervals, 1, 0, NULL, 0, 0, flags,
                                              &details);
    if (instance < 0) return instance;
    snprintf(name, sizeof(name), "%s, %d demes, %d BASTA thread%s", details.implName, K, threads,
             threads > 1 ? "s" : "");
    int rc = BEAGLE_SUCCESS;
    if (allocate)
        rc |= beagleAllocateBastaBuffers(instance, 1, nIntervals + EXTRA_INTERVALS, bufferCount + 1, 1, threads);
    if (gradient)
        rc |= beagleAllocateBastaGradBuffers(instance, gradient * (bufferCount + 1)); /* < 0: adjoint mode */
    for (int j = 0; j < nIntervals; j++)
        rc |= beagleSetTransitionMatrix(instance, j, matrices + j * K * K, 1.0);
    double tip[8];
    for (int t = 0; t < TIPS; t++) {
        for (int k = 0; k < K; k++) tip[k] = k == tipState[t] ? 1.0 : 0.0;
        rc |= beagleSetPartials(instance, t, tip);
    }
    rc |= beagleSetStateFrequencies(instance, 0, sizes);
    check(rc == BEAGLE_SUCCESS, "set-up returned an error");
    return instance;
}

/* every _v5 function on operations ops: each must return expected; then the instance must repeat its results */
static void expectRejected(int instance, const BastaOperation_v5* ops, const int* intervalStarts, int coalescentIndex,
                           int expected, const char* what, const Result* r8, const Result* r5) {
    char buf[200];
    double logL = 0.0, grad[64];
    snprintf(buf, sizeof(buf), "%s: updateBastaPartials_v5", what);
    checkCode(beagleUpdateBastaPartials_v5(instance, ops, nOps, intervalStarts, nIntervals + 1, 0, coalescentIndex),
              expected, buf);
    snprintf(buf, sizeof(buf), "%s: accumulateBastaPartials_v5", what);
    checkCode(beagleAccumulateBastaPartials_v5(instance, ops, nOps, intervalStarts, nIntervals + 1, NULL, 0,
                                               coalescentIndex, &logL), expected, buf);
    snprintf(buf, sizeof(buf), "%s: updateBastaPartialsGrad_v5", what);
    checkCode(beagleUpdateBastaPartialsGrad_v5(instance, ops, nOps, intervalStarts, nIntervals + 1, 0,
                                               coalescentIndex), expected, buf);
    snprintf(buf, sizeof(buf), "%s: accumulateBastaPartialsGrad_v5", what);
    checkCode(beagleAccumulateBastaPartialsGrad_v5(instance, ops, nOps, intervalStarts, nIntervals + 1, NULL, 0,
                                                   coalescentIndex, 0, -1, -1, NULL, grad), expected, buf);
    Result* again = (Result*) malloc(sizeof(Result));
    if (r8 != NULL) {
        snprintf(buf, sizeof(buf), "%s: the 8-integer functions do not repeat their result", what);
        check(forward8(instance, again) == BEAGLE_SUCCESS && sameResult(again, r8), buf);
    }
    if (r5 != NULL) {
        snprintf(buf, sizeof(buf), "%s: the _v5 functions do not repeat their result", what);
        check(forward5(instance, ops9, again) == BEAGLE_SUCCESS && sameResult(again, r5), buf);
    }
    free(again);
}

static void testConfiguration(int single, int threads) {
    const double logLTolerance = single ? 1e-5 : 1e-12, gradientTolerance = single ? 1e-4 : 1e-10;
    const double partialTolerance = single ? 1e-5 : 1e-13;
    const int R = layoutSize();
    double* constant = layout(0);
    double* piecewise = layout(1);
    Result *r8 = (Result*) malloc(sizeof(Result)), *r5 = (Result*) malloc(sizeof(Result));
    Result *rp = (Result*) malloc(sizeof(Result)), *again = (Result*) malloc(sizeof(Result));
    double g8[64], g5[64], gAgain[64];

    const int instance = createInstance(single, threads, 1, 1);
    if (instance < 0) {
        fprintf(stderr, "FAIL: no CPU instance (%s, %d demes): error %d\n", single ? "single" : "double", K, instance);
        failures++;
        goto done;
    }

    /* T1 */
    checkCode(beagleSetBastaPopulationSizes(instance, constant, R), BEAGLE_SUCCESS, "T1 beagleSetBastaPopulationSizes");
    checkCode(forward8(instance, r8), BEAGLE_SUCCESS, "T1 8-integer forward pass");
    checkCode(gradient8(instance, g8), BEAGLE_SUCCESS, "T1 8-integer legacy gradient");
    checkCode(forward5(instance, ops9, r5), BEAGLE_SUCCESS, "T1 _v5 forward pass");
    checkCode(gradient5(instance, ops9, g5), BEAGLE_SUCCESS, "T1 _v5 legacy gradient");
    check(memcmp(r8->partials, r5->partials, sizeof(double) * bufferCount * K) == 0,
          "T1 partials of _v5 differ from the 8-integer ones");
    check(memcmp(r8->coalescent, r5->coalescent, sizeof(double) * (nIntervals + EXTRA_INTERVALS)) == 0,
          "T1 coalescent probabilities of _v5 differ from the 8-integer ones");
    check(relative(r5->logL, r8->logL) <= logLTolerance, "T1 log likelihood of _v5 differs from the 8-integer one");
    check(gradientDifference(g5, g8) <= gradientTolerance, "T1 legacy gradient of _v5 differs from the 8-integer one");
    check(isfinite(r5->logL) && r5->logL < 0.0, "T1 log likelihood is not a finite negative number");
    printf("%s: logL 8-integer %.17g _v5 %.17g (relative %.2g), legacy gradient relative %.2g\n", name, r8->logL,
           r5->logL, relative(r5->logL, r8->logL), gradientDifference(g5, g8));

    /* T2 */
    check(forward8(instance, again) == BEAGLE_SUCCESS && sameResult(again, r8), "T2 8-integer repeat differs");
    check(gradient8(instance, gAgain) == BEAGLE_SUCCESS && memcmp(gAgain, g8, sizeof(double) * K * K) == 0,
          "T2 8-integer gradient repeat differs");
    check(forward5(instance, ops9, again) == BEAGLE_SUCCESS && sameResult(again, r5), "T2 _v5 repeat differs");
    check(gradient5(instance, ops9, gAgain) == BEAGLE_SUCCESS && memcmp(gAgain, g5, sizeof(double) * K * K) == 0,
          "T2 _v5 gradient repeat differs");

    /* T3: interval j's block at 1 + perm[j] * (K + 3), NaN elsewhere */
    {
        int perm[MAX_INTERVALS];
        for (int j = 0; j < nIntervals; j++) perm[j] = j;
        for (int j = nIntervals - 1; j > 0; j--) {
            const int i = (int) (unif() * (j + 1)), x = perm[i];
            perm[i] = perm[j]; perm[j] = x;
        }
        const int R3 = 1 + nIntervals * (K + 3) + 2;
        double* shuffled = (double*) malloc(sizeof(double) * 2 * R3);
        for (int i = 0; i < 2 * R3; i++) shuffled[i] = NAN;
        BastaOperation_v5* moved = (BastaOperation_v5*) malloc(sizeof(BastaOperation_v5) * nOps);
        for (int j = 0; j < nIntervals; j++) {
            const int p = 1 + perm[j] * (K + 3);
            for (int k = 0; k < K; k++) {
                shuffled[p + k] = constant[K + j * K + k];
                shuffled[R3 + p + k] = constant[R + K + j * K + k];
            }
        }
        for (int o = 0; o < nOps; o++) {
            moved[o] = ops9[o];
            moved[o].populationSizeOffset = 1 + perm[moved[o].intervalNumber] * (K + 3);
        }
        checkCode(beagleSetBastaPopulationSizes(instance, shuffled, R3), BEAGLE_SUCCESS, "T3 set shuffled layout");
        check(forward5(instance, moved, again) == BEAGLE_SUCCESS && sameResult(again, r5),
              "T3 shuffled layout gives other results");
        check(gradient5(instance, moved, gAgain) == BEAGLE_SUCCESS && memcmp(gAgain, g5, sizeof(double) * K * K) == 0,
              "T3 shuffled layout gives another gradient");

        /* T5 here too: the current buffer (0) regrew to R3; back to R, then to a size too small for the offsets */
        checkCode(beagleSetBastaPopulationSizes(instance, constant, R), BEAGLE_SUCCESS, "T5 set R again");
        check(forward5(instance, ops9, again) == BEAGLE_SUCCESS && sameResult(again, r5), "T5 R again differs");
        checkCode(beagleSetBastaPopulationSizes(instance, constant, R - K), BEAGLE_SUCCESS, "T5 set a smaller R");
        expectRejected(instance, ops9, starts, 0, BEAGLE_ERROR_OUT_OF_RANGE, "T5 offsets beyond a smaller R", r8, NULL);
        checkCode(beagleSetBastaPopulationSizes(instance, constant, R), BEAGLE_SUCCESS, "T5 set R once more");
        check(forward5(instance, ops9, again) == BEAGLE_SUCCESS && sameResult(again, r5), "T5 R once more differs");
        free(shuffled);
        free(moved);
    }

    /* T4: piecewise sizes; every coalescent partial, and the reduction recomputed from the partials */
    checkCode(beagleSetBastaPopulationSizes(instance, piecewise, R), BEAGLE_SUCCESS, "T4 set piecewise sizes");
    checkCode(forward5(instance, ops9, rp), BEAGLE_SUCCESS, "T4 _v5 forward pass");
    {
        double worst = 0.0, logL = 0.0;
        int coalescences = 0;
        for (int o = 0; o < nOps; o++) {
            const BastaOperation_v5* op = &ops9[o];
            if (op->child2Partials < 0) continue;
            const double* dest = rp->partials + op->destinationPartials * K;
            const double* a1 = rp->partials + op->accumulation1Partials * K;
            const double* a2 = rp->partials + op->accumulation2Partials2 * K;
            double sum = 0.0;
            for (int i = 0; i < K; i++) sum += a1[i] * a2[i] / piecewise[op->populationSizeOffset + i];
            for (int i = 0; i < K; i++)
                worst = fmax(worst, relative(dest[i], a1[i] * a2[i] / piecewise[op->populationSizeOffset + i] / sum));
            coalescences++;
        }
        for (int j = 0; j < nIntervals; j++) {
            double e[8] = {0}, f[8] = {0}, g[8] = {0}, h[8] = {0};
            for (int o = starts[j]; o < starts[j + 1]; o++) {
                const BastaOperation_v5* op = &ops9[o];
                const int begin[2] = {op->child1Partials, op->child2Partials};
                const int end[2] = {op->accumulation1Partials, op->accumulation2Partials2};
                for (int c = 0; c < (op->child2Partials >= 0 ? 2 : 1); c++) {
                    for (int k = 0; k < K; k++) {
                        const double x = rp->partials[begin[c] * K + k], y = rp->partials[end[c] * K + k];
                        e[k] += x; f[k] += x * x; g[k] += y; h[k] += y * y;
                    }
                }
            }
            const int p = ops9[starts[j]].populationSizeOffset;
            double sum = 0.0;
            for (int k = 0; k < K; k++) sum += (e[k] * e[k] - f[k] + g[k] * g[k] - h[k]) * piecewise[R + p + k];
            logL += -sum / 4;
            const double prob = rp->coalescent[ops9[starts[j]].intervalNumber];
            if (prob != 0.0) logL += log(prob);
        }
        check(coalescences == TIPS - 1, "T4 not every coalescence was checked");
        check(worst <= partialTolerance, "T4 a coalescent partial is not acc1 * acc2 / N[p + i], normalised");
        check(relative(rp->logL, logL) <= logLTolerance, "T4 log likelihood differs from the reduction recomputed here");
        check(relative(rp->logL, r5->logL) > 1e-3, "T4 piecewise sizes give the constant log likelihood");
        printf("%s: piecewise logL %.17g, recomputed %.17g; coalescent partials relative %.2g\n", name, rp->logL,
               logL, worst);
    }

    /* T5: A (constant) in buffer 0, B (piecewise) in buffer 1, then flips */
    checkCode(beagleSetBastaPopulationSizesBuffer(instance, constant, R, 0), BEAGLE_SUCCESS, "T5 set buffer 0");
    checkCode(beagleSetBastaPopulationSizesBuffer(instance, piecewise, R, 1), BEAGLE_SUCCESS, "T5 set buffer 1");
    check(forward5(instance, ops9, again) == BEAGLE_SUCCESS && sameResult(again, rp),
          "T5 buffer 1 (current after its set) does not give the piecewise results");
    checkCode(beagleSetCurrentPopulationSizeBuffer(instance, 0), BEAGLE_SUCCESS, "T5 setCurrent(0)");
    check(forward5(instance, ops9, again) == BEAGLE_SUCCESS && sameResult(again, r5),
          "T5 buffer 0 does not give the constant results");
    checkCode(beagleSetCurrentPopulationSizeBuffer(instance, 1), BEAGLE_SUCCESS, "T5 setCurrent(1)");
    check(forward5(instance, ops9, again) == BEAGLE_SUCCESS && sameResult(again, rp),
          "T5 buffer 1 does not give the piecewise results");
    checkCode(beagleSetBastaPopulationSizes(instance, constant, R), BEAGLE_SUCCESS, "T5 set constant into buffer 1");
    checkCode(beagleSetCurrentPopulationSizeBuffer(instance, 0), BEAGLE_SUCCESS, "T5 setCurrent(0) again");
    check(forward5(instance, ops9, again) == BEAGLE_SUCCESS && sameResult(again, r5),
          "T5 buffer 0 changed when buffer 1 was set");

    /* T6 on this instance: buffer 0 current, constant sizes in both */
    {
        BastaOperation_v5* bad = (BastaOperation_v5*) malloc(sizeof(BastaOperation_v5) * nOps);
        int badStarts[MAX_INTERVALS + 1];
        const int last = ops9[nOps - 1].intervalNumber;
#define RESET() memcpy(bad, ops9, sizeof(BastaOperation_v5) * nOps)
        /* an interval with at least two operations, for two different offsets */
        int two = 0;
        while (starts[two + 1] - starts[two] < 2) two++;

        RESET();
        for (int o = starts[0]; o < starts[1]; o++) bad[o].populationSizeOffset = -1;
        expectRejected(instance, bad, starts, 0, BEAGLE_ERROR_OUT_OF_RANGE, "T6 negative offset", r8, r5);
        RESET();
        for (int o = 0; o < nOps; o++)
            if (bad[o].intervalNumber == last) bad[o].populationSizeOffset = R - K + 1;
        expectRejected(instance, bad, starts, 0, BEAGLE_ERROR_OUT_OF_RANGE, "T6 offset + K > R", r8, r5);
        RESET();
        bad[starts[two] + 1].populationSizeOffset = 0; /* in range, but not the interval's offset */
        expectRejected(instance, bad, starts, 0, BEAGLE_ERROR_OUT_OF_RANGE, "T6 two offsets for one intervalNumber",
                       r8, r5);
        RESET();
        bad[nOps - 1].intervalNumber = nIntervals + EXTRA_INTERVALS;
        expectRejected(instance, bad, starts, 0, BEAGLE_ERROR_OUT_OF_RANGE, "T6 intervalNumber >= bufferLength", r8,
                       r5);
        RESET();
        bad[0].intervalNumber = -1;
        expectRejected(instance, bad, starts, 0, BEAGLE_ERROR_OUT_OF_RANGE, "T6 negative intervalNumber", r8, r5);
        memcpy(badStarts, starts, sizeof(int) * (nIntervals + 1));
        badStarts[1] = nOps;
        expectRejected(instance, ops9, badStarts, 0, BEAGLE_ERROR_OUT_OF_RANGE, "T6 interval start >= count", r8, r5);
        memcpy(badStarts, starts, sizeof(int) * (nIntervals + 1));
        badStarts[nIntervals] = nOps + 1;
        expectRejected(instance, ops9, badStarts, 0, BEAGLE_ERROR_OUT_OF_RANGE, "T6 last interval start > count", r8,
                       r5);
        expectRejected(instance, ops9, NULL, 0, BEAGLE_ERROR_OUT_OF_RANGE, "T6 NULL interval starts", r8, r5);
        expectRejected(instance, ops9, starts, 1, BEAGLE_ERROR_OUT_OF_RANGE, "T6 coalescentIndex 1 of 1", r8, r5);
#undef RESET
        free(bad);

        /* the setters: rejected calls change nothing */
        checkCode(beagleSetBastaPopulationSizesBuffer(instance, piecewise, R, 2), BEAGLE_ERROR_OUT_OF_RANGE,
                  "T6 setBastaPopulationSizesBuffer(buffer 2)");
        checkCode(beagleSetBastaPopulationSizesBuffer(instance, piecewise, R, -1), BEAGLE_ERROR_OUT_OF_RANGE,
                  "T6 setBastaPopulationSizesBuffer(buffer -1)");
        checkCode(beagleSetBastaPopulationSizesBuffer(instance, piecewise, 0, 1), BEAGLE_ERROR_OUT_OF_RANGE,
                  "T6 setBastaPopulationSizesBuffer(R 0)");
        checkCode(beagleSetBastaPopulationSizesBuffer(instance, NULL, R, 1), BEAGLE_ERROR_GENERAL,
                  "T6 setBastaPopulationSizesBuffer(NULL)");
        checkCode(beagleSetBastaPopulationSizes(instance, piecewise, -1), BEAGLE_ERROR_OUT_OF_RANGE,
                  "T6 setBastaPopulationSizes(R -1)");
        checkCode(beagleSetBastaPopulationSizes(instance, NULL, R), BEAGLE_ERROR_GENERAL,
                  "T6 setBastaPopulationSizes(NULL)");
        checkCode(beagleSetCurrentPopulationSizeBuffer(instance, 2), BEAGLE_ERROR_OUT_OF_RANGE,
                  "T6 setCurrentPopulationSizeBuffer(2)");
        checkCode(beagleSetCurrentPopulationSizeBuffer(instance, -1), BEAGLE_ERROR_OUT_OF_RANGE,
                  "T6 setCurrentPopulationSizeBuffer(-1)");
        check(forward5(instance, ops9, again) == BEAGLE_SUCCESS && sameResult(again, r5),
              "T6 a rejected setter changed the population sizes");

        double grad[64], popSizeGrad[8];
        checkCode(beagleAccumulateBastaPartialsGrad_v5(instance, ops9, nOps, starts, nIntervals + 1, NULL, 0, 0, 0,
                                                       -1, -1, popSizeGrad, grad),
                  BEAGLE_ERROR_NO_IMPLEMENTATION, "T6 accumulateBastaPartialsGrad_v5 with popSizeGradOut");
        check(forward8(instance, again) == BEAGLE_SUCCESS && sameResult(again, r8),
              "T6 8-integer result changed after popSizeGradOut");
        check(gradient5(instance, ops9, gAgain) == BEAGLE_SUCCESS && memcmp(gAgain, g5, sizeof(double) * K * K) == 0,
              "T6 _v5 gradient changed after the rejected calls");
    }
    beagleFinalizeInstance(instance);

    /* T6 on other instances */
    {
        Result* own = (Result*) malloc(sizeof(Result));
        double grad[64];
        /* no allocateBastaBuffers */
        int other = createInstance(single, threads, 0, 0);
        checkCode(beagleSetBastaPopulationSizes(other, constant, R), BEAGLE_SUCCESS, "T6 sizes before BASTA buffers");
        expectRejected(other, ops9, starts, 0, BEAGLE_ERROR_GENERAL, "T6 no BASTA buffers", NULL, NULL);
        beagleFinalizeInstance(other);

        /* BASTA buffers, no population sizes; then buffer 1 current and never set */
        other = createInstance(single, threads, 1, 1);
        checkCode(forward8(other, own), BEAGLE_SUCCESS, "T6 8-integer forward pass");
        expectRejected(other, ops9, starts, 0, BEAGLE_ERROR_GENERAL, "T6 no population sizes", own, NULL);
        checkCode(beagleSetBastaPopulationSizesBuffer(other, constant, R, 0), BEAGLE_SUCCESS, "T6 set buffer 0");
        checkCode(beagleSetCurrentPopulationSizeBuffer(other, 1), BEAGLE_SUCCESS, "T6 setCurrent(1)");
        expectRejected(other, ops9, starts, 0, BEAGLE_ERROR_GENERAL, "T6 current buffer never set", own, NULL);
        checkCode(beagleSetCurrentPopulationSizeBuffer(other, 0), BEAGLE_SUCCESS, "T6 setCurrent(0)");
        check(forward5(other, ops9, own) == BEAGLE_SUCCESS && sameResult(own, r5),
              "T6 a second instance gives other _v5 results");
        beagleFinalizeInstance(other);

        /* no gradient buffers: the _v5 gradient functions return BEAGLE_ERROR_GENERAL, the forward pass works */
        other = createInstance(single, threads, 1, 0);
        checkCode(beagleSetBastaPopulationSizes(other, constant, R), BEAGLE_SUCCESS, "T6 set sizes");
        checkCode(forward5(other, ops9, own), BEAGLE_SUCCESS, "T6 _v5 forward pass without gradient buffers");
        checkCode(beagleUpdateBastaPartialsGrad_v5(other, ops9, nOps, starts, nIntervals + 1, 0, 0),
                  BEAGLE_ERROR_GENERAL, "T6 updateBastaPartialsGrad_v5 without gradient buffers");
        checkCode(beagleAccumulateBastaPartialsGrad_v5(other, ops9, nOps, starts, nIntervals + 1, NULL, 0, 0, 0, -1,
                                                       -1, NULL, grad),
                  BEAGLE_ERROR_GENERAL, "T6 accumulateBastaPartialsGrad_v5 without gradient buffers");
        check(sameResult(own, r5), "T6 an instance without gradient buffers gives other _v5 results");
        beagleFinalizeInstance(other);

        /* adjoint mode */
        other = createInstance(single, threads, 1, -1);
        checkCode(beagleSetBastaPopulationSizes(other, constant, R), BEAGLE_SUCCESS, "T6 set sizes (adjoint)");
        checkCode(forward8(other, own), BEAGLE_SUCCESS, "T6 8-integer forward pass (adjoint)");
        expectRejected(other, ops9, starts, 0, BEAGLE_ERROR_NO_IMPLEMENTATION, "T6 adjoint mode", own, NULL);
        beagleFinalizeInstance(other);
        free(own);
    }

done:
    free(constant);
    free(piecewise);
    free(r8); free(r5); free(rp); free(again);
}

int main(void) {
    const int demes[3] = {3, 4, 5};
    for (int d = 0; d < 3; d++) {
        K = demes[d];
        for (int single = 0; single < 2; single++) {
            for (int threads = 1; threads <= 4; threads += 3) {
                rngState = 7919 * (uint64_t) K + 17;
                simulate();
                int widest = 0;
                for (int j = 0; j < nIntervals; j++) if (starts[j + 1] - starts[j] > widest) widest = starts[j + 1] - starts[j];
                if (d == 0 && single == 0 && threads == 1)
                    printf("problem: %d operations, %d intervals (up to %d operations), %d buffers\n", nOps, nIntervals,
                           widest, bufferCount);
                testConfiguration(single, threads);
            }
        }
    }
    free(matrices);
    if (failures) {
        fprintf(stderr, "%d failure%s\n", failures, failures > 1 ? "s" : "");
        return 1;
    }
    printf("bastav5test: all passed\n");
    return 0;
}
