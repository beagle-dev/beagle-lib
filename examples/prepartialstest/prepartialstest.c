/*
 * prepartialstest: beagleUpdatePrePartialsByPartition (v4.0.1's 3-argument form, BEAGLE_PARTIALS_BOTTOM) and
 * beagleUpdatePrePartialsByPartition_v5 against the whole-pattern pre-order updates, on CPU instances with two pattern
 * partitions, without and with C++ threading. Every by-partition pre-order partial must equal the whole-pattern one bit
 * for bit: the 3-argument function's and _v5's with BOTTOM those of beagleUpdatePrePartials, and _v5's with TOP those
 * of beagleUpdatePrePartials_v5(..., BEAGLE_PARTIALS_TOP). It is C, so it also checks that beagle.h compiles as C.
 */

#include <stdio.h>
#include <string.h>

#include "libhmsbeagle/beagle.h"

/* the rooted tree ((0,1)4,(2,3)5)6; tips 0 and 1 have partials, 2 and 3 states; node n's branch has matrix n */
enum { TIPS = 4, NODES = 7, ROOT = 6, STATES = 4, PATTERNS = 10, CATEGORIES = 2, PARTITIONS = 2 };
enum { PRE = NODES };                       /* node n's pre-order partials are in buffer PRE + n */
enum { OPS = NODES - 1 };                   /* one pre-order operation per non-root node */
enum { SIZE = STATES * PATTERNS * CATEGORIES };

static const int children[NODES][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 1}, {2, 3}, {4, 5}};
static const int internalPreOrder[3] = {ROOT, 4, 5}; /* parents first */
static const int patternPartitions[PATTERNS] = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};

static int failures = 0;

static void check(int ok, const char* what, const char* name) {
    if (!ok) {
        fprintf(stderr, "FAIL %s: %s\n", name, what);
        failures++;
    }
}

/* the pre-order operations, parents first, whole-pattern and per partition: BOTTOM partials include the node's own
 * branch, so child1TransitionMatrix is the node's; TOP partials do not, so it is the parent's (none at the root) */
static void preOrderOperations(int top, BeagleOperation ops[OPS], BeagleOperationByPartition byPartition[OPS * PARTITIONS]) {
    int count = 0;
    for (int i = 0; i < 3; i++) {
        const int n = internalPreOrder[i];
        for (int k = 0; k < 2; k++) {
            const int child = children[n][k], sibling = children[n][1 - k];
            BeagleOperation* op = &ops[count];
            op->destinationPartials = PRE + child;
            op->destinationScaleWrite = BEAGLE_OP_NONE;
            op->destinationScaleRead = BEAGLE_OP_NONE;
            op->child1Partials = PRE + n;
            op->child1TransitionMatrix = top ? (n == ROOT ? BEAGLE_OP_NONE : n) : child;
            op->child2Partials = sibling;
            op->child2TransitionMatrix = sibling;
            for (int p = 0; p < PARTITIONS; p++) {
                BeagleOperationByPartition* q = &byPartition[count * PARTITIONS + p];
                q->destinationPartials = op->destinationPartials;
                q->destinationScaleWrite = op->destinationScaleWrite;
                q->destinationScaleRead = op->destinationScaleRead;
                q->child1Partials = op->child1Partials;
                q->child1TransitionMatrix = op->child1TransitionMatrix;
                q->child2Partials = op->child2Partials;
                q->child2TransitionMatrix = op->child2TransitionMatrix;
                q->partition = p;
                q->cumulativeScaleIndex = BEAGLE_OP_NONE;
            }
            count++;
        }
    }
}

/* overwrite every non-root pre-order buffer, so that a result cannot be left over from an earlier update */
static void clearPreOrder(int instance) {
    static double sentinel[SIZE];
    for (int i = 0; i < SIZE; i++) sentinel[i] = -1.0;
    for (int n = 0; n < NODES; n++) {
        if (n != ROOT) beagleSetPartials(instance, PRE + n, sentinel);
    }
}

static void readPreOrder(int instance, double out[NODES][SIZE]) {
    for (int n = 0; n < NODES; n++) beagleGetPartials(instance, PRE + n, BEAGLE_OP_NONE, out[n]);
}

static void compare(double result[NODES][SIZE], double expected[NODES][SIZE], const char* what, const char* name) {
    for (int n = 0; n < NODES; n++) {
        if (memcmp(result[n], expected[n], sizeof(double) * SIZE) != 0) {
            for (int i = 0; i < SIZE; i++) {
                if (memcmp(&result[n][i], &expected[n][i], sizeof(double)) != 0) {
                    fprintf(stderr, "  node %d, entry %d: %.17g, expected %.17g\n", n, i, result[n][i], expected[n][i]);
                    break;
                }
            }
            check(0, what, name);
            return;
        }
    }
}

static void testInstance(int threaded) {
    static double bottom[NODES][SIZE], top[NODES][SIZE], result[NODES][SIZE];
    char name[128];
    BeagleInstanceDetails details;
    const int instance = beagleCreateInstance(TIPS, 2 * NODES, 2, STATES, PATTERNS, 1, NODES, CATEGORIES, 0, NULL, 0,
                                              BEAGLE_FLAG_SCALING_MANUAL,
                                              BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PRECISION_DOUBLE |
                                              BEAGLE_FLAG_VECTOR_NONE |
                                              (threaded ? BEAGLE_FLAG_THREADING_CPP : BEAGLE_FLAG_THREADING_NONE),
                                              &details);
    if (instance < 0) {
        fprintf(stderr, "FAIL: no CPU instance (%s threading): error %d\n", threaded ? "C++" : "no", instance);
        failures++;
        return;
    }
    snprintf(name, sizeof(name), "%s, %s", details.implName, threaded ? "THREADING_CPP" : "THREADING_NONE");
    check(threaded ? (details.flags & BEAGLE_FLAG_THREADING_CPP) != 0 : 1, "instance without THREADING_CPP", name);

    /* JC69 */
    const double evec[STATES * STATES] = {1.0,  2.0,  0.0,  0.5,
                                          1.0, -2.0,  0.5,  0.0,
                                          1.0,  2.0,  0.0, -0.5,
                                          1.0, -2.0, -0.5,  0.0};
    const double ivec[STATES * STATES] = {0.25,   0.25,  0.25,   0.25,
                                          0.125, -0.125, 0.125, -0.125,
                                          0.0,    1.0,   0.0,   -1.0,
                                          1.0,    0.0,  -1.0,    0.0};
    const double eval[STATES] = {0.0, -4.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0};
    const double frequencies[STATES] = {0.25, 0.25, 0.25, 0.25};
    const double rates[CATEGORIES] = {0.5, 1.5}, weights[CATEGORIES] = {0.5, 0.5};
    const double lengths[NODES - 1] = {0.1, 0.25, 0.15, 0.3, 0.05, 0.2};
    const int matrices[NODES - 1] = {0, 1, 2, 3, 4, 5};
    double patternWeights[PATTERNS], tipPartials[STATES * PATTERNS];
    int tipStates[PATTERNS];

    int rc = beagleSetPatternPartitions(instance, PARTITIONS, patternPartitions);
    for (int p = 0; p < PATTERNS; p++) patternWeights[p] = 1.0;
    rc |= beagleSetPatternWeights(instance, patternWeights);
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < STATES * PATTERNS; i++) tipPartials[i] = 0.05 + ((7 * t + 3 * i) % 11) / 11.0;
        rc |= beagleSetTipPartials(instance, t, tipPartials);
    }
    for (int t = 2; t < TIPS; t++) {
        for (int p = 0; p < PATTERNS; p++) tipStates[p] = (p == 2 * t) ? STATES : (t + 3 * p) % STATES; /* one gap */
        rc |= beagleSetTipStates(instance, t, tipStates);
    }
    rc |= beagleSetStateFrequencies(instance, 0, frequencies);
    rc |= beagleSetEigenDecomposition(instance, 0, evec, ivec, eval);
    rc |= beagleSetCategoryRates(instance, rates);
    rc |= beagleSetCategoryWeights(instance, 0, weights);
    rc |= beagleUpdateTransitionMatrices(instance, 0, matrices, NULL, NULL, lengths, NODES - 1);

    BeagleOperation post[3];
    for (int i = 0; i < 3; i++) {
        const int n = internalPreOrder[2 - i];
        post[i].destinationPartials = n;
        post[i].destinationScaleWrite = BEAGLE_OP_NONE;
        post[i].destinationScaleRead = BEAGLE_OP_NONE;
        post[i].child1Partials = post[i].child1TransitionMatrix = children[n][0];
        post[i].child2Partials = post[i].child2TransitionMatrix = children[n][1];
    }
    rc |= beagleUpdatePartials(instance, post, 3, BEAGLE_OP_NONE);
    const int rootPre = PRE + ROOT, frequencyIndex = 0;
    rc |= beagleSetRootPrePartials(instance, &rootPre, &frequencyIndex, 1);
    check(rc == BEAGLE_SUCCESS, "set-up returned an error", name);

    BeagleOperation bottomOps[OPS], topOps[OPS];
    BeagleOperationByPartition bottomByPartition[OPS * PARTITIONS], topByPartition[OPS * PARTITIONS];
    preOrderOperations(0, bottomOps, bottomByPartition);
    preOrderOperations(1, topOps, topByPartition);

    /* the references: whole-pattern BOTTOM and TOP */
    clearPreOrder(instance);
    check(beagleUpdatePrePartials(instance, bottomOps, OPS, BEAGLE_OP_NONE) == BEAGLE_SUCCESS,
          "beagleUpdatePrePartials returned an error", name);
    readPreOrder(instance, bottom);
    clearPreOrder(instance);
    check(beagleUpdatePrePartials_v5(instance, topOps, OPS, BEAGLE_OP_NONE, BEAGLE_PARTIALS_TOP) == BEAGLE_SUCCESS,
          "beagleUpdatePrePartials_v5 (TOP) returned an error", name);
    readPreOrder(instance, top);
    int computed = 1, topDiffers = 0;
    for (int n = 0; n < NODES; n++) {
        for (int i = 0; i < SIZE; i++) {
            computed = computed && bottom[n][i] >= 0.0 && top[n][i] >= 0.0;
            topDiffers = topDiffers || (n != ROOT && bottom[n][i] != top[n][i]);
        }
    }
    check(computed, "a whole-pattern pre-order partial was not computed", name);
    check(topDiffers, "TOP and BOTTOM pre-order partials are the same", name);

    clearPreOrder(instance);
    check(beagleUpdatePrePartialsByPartition(instance, bottomByPartition, OPS * PARTITIONS) == BEAGLE_SUCCESS,
          "beagleUpdatePrePartialsByPartition returned an error", name);
    readPreOrder(instance, result);
    compare(result, bottom, "beagleUpdatePrePartialsByPartition != beagleUpdatePrePartials", name);

    clearPreOrder(instance);
    check(beagleUpdatePrePartialsByPartition_v5(instance, bottomByPartition, OPS * PARTITIONS,
                                                BEAGLE_PARTIALS_BOTTOM) == BEAGLE_SUCCESS,
          "beagleUpdatePrePartialsByPartition_v5 (BOTTOM) returned an error", name);
    readPreOrder(instance, result);
    compare(result, bottom, "beagleUpdatePrePartialsByPartition_v5 (BOTTOM) != beagleUpdatePrePartials", name);

    clearPreOrder(instance);
    check(beagleUpdatePrePartialsByPartition_v5(instance, topByPartition, OPS * PARTITIONS,
                                                BEAGLE_PARTIALS_TOP) == BEAGLE_SUCCESS,
          "beagleUpdatePrePartialsByPartition_v5 (TOP) returned an error", name);
    readPreOrder(instance, result);
    compare(result, top, "beagleUpdatePrePartialsByPartition_v5 (TOP) != beagleUpdatePrePartials_v5 (TOP)", name);

    beagleFinalizeInstance(instance);
    printf("%s: checked\n", name);
}

int main(void) {
    testInstance(0);
    testInstance(1);
    if (failures) {
        fprintf(stderr, "prepartialstest: %d failure(s)\n", failures);
        return 1;
    }
    printf("prepartialstest: passed\n");
    return 0;
}
