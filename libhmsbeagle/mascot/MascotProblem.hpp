/*
 * MascotProblem.hpp -- internal interface between the MASCOT C API (mascot.cpp) and the kernels.
 *
 * A Problem is one evaluation's inputs in the caller's layout (K states, no padding). Kernels A and B each
 * implement Kernel; every kernel object owns its workspace, so distinct objects are independent.
 */
#ifndef MASCOT_PROBLEM_HPP
#define MASCOT_PROBLEM_HPP

#include <memory>

namespace mascot {

enum EventType { SAMPLE = 1, COALESCENT = 2 };

/* Same values as MascotStepRuleKind in mascot.h. */
enum StepRuleKind { STEPS_FIXED = 0, STEPS_ADAPTIVE = 1 };

/* Same values as MascotStatus in mascot.h. */
enum Status { OK = 0, ERROR_ARGUMENT = -1, ERROR_COALESCENT_RATE = -2, ERROR_PROBABILITIES = -3 };

struct Problem {
    int K;                          /* states (demes), >= 2                                                  */
    int lineageCount;               /* max lineage id + 1                                                    */
    int intervalCount;
    int initialSampleNode;
    int epochCount;
    int stepRuleKind;               /* StepRuleKind; the rules are those of BEAST's MascotStepRule:           */
    double maxStep;                 /*   FIXED: steps = max(1, ceil(len / maxStep)); ADAPTIVE: bound on h (<= 0: none) */
    double tau;                     /*   ADAPTIVE: target len * lambda per step                               */
    double minStep;                 /*   ADAPTIVE: lower bound on h (<= 0: none)                              */
    int maxStepsPerSegment;         /*   ADAPTIVE: upper bound on the step count (<= 0: none)                 */
    double initialTime;
    const double* boundaries;       /* epochCount + 1, first 0, last +inf                                    */
    const double* migration;        /* epochCount x K x K, row = source, diagonal = -row sum                  */
    const double* inversePopulation;/* epochCount x K                                                         */
    const double* branchRates;      /* lineageCount, or nullptr                                              */
    const double* intervalLengths;  /* intervalCount                                                         */
    const int* eventTypes;          /* EventType                                                             */
    const int* sampleNodes;
    const int* child1;
    const int* child2;
    const int* parent;
    const double* tipPartials;      /* lineageCount x K; only the rows of sampled nodes are read             */
    bool checkProbabilities;        /* after every segment and event, as MascotForwardEvaluator               */
    int checkpointReadSet;          /* forward() only: 0, 1 or -1 (none); see mascot.h, incremental restart    */
    int checkpointWriteSet;         /* forward() only: 0, 1 or -1 (none); may equal checkpointReadSet          */
};

class Kernel {
public:
    virtual ~Kernel() {}

    /*
     * Forward pass only. Returns a Status; *logL is set when OK. With problem.checkpointReadSet >= 0 it resumes from
     * the checkpoint before the first schedule interval that differs from that set's saved inputs (any other input
     * difference: from the beginning); with checkpointWriteSet >= 0 it saves a checkpoint at every interval boundary,
     * with the inputs. The result is bitwise that of a full pass. forwardAndReverse ignores both sets.
     */
    virtual int forward(const Problem& problem, double* logL) = 0;

    /*
     * Forward pass with tape, then the reverse pass. Returns a Status; outputs are set when OK.
     * thetaGradient (required): epochCount x (K(K-1) + K), migration rates then log Ne, zeroed by the kernel.
     * clockGradient: lineageCount, zeroed by the kernel; nullptr skips it (and must be nullptr without branch rates).
     * ancestralScores: lineageCount x K; NaN except the rows of coalescent parents; nullptr skips it.
     */
    virtual int forwardAndReverse(const Problem& problem, double* logL, double* thetaGradient,
                                  double* clockGradient, double* ancestralScores) = 0;

    /*
     * The step count of each integration segment (an interval split at epoch boundaries) of the last evaluation, in
     * order. Sets *count and writes min(count, capacity) entries; returns ERROR_ARGUMENT if capacity is too small.
     */
    virtual int lastStepCounts(int* steps, int capacity, int* count) const = 0;

    /* The interval the last forward() resumed from (0: the whole pass), or -1 if it read no checkpoint set. */
    virtual int lastRestartInterval() const = 0;

    virtual char name() const = 0;       /* 'A' or 'B'                                  */
    virtual int simdWidth() const = 0;   /* doubles per vector in B; 0 for A            */
    virtual bool fusedMultiplyAdd() const = 0;
};

/* Factories, one translation unit per kernel and FP mode. */
std::unique_ptr<Kernel> makeKernelAStrict();
std::unique_ptr<Kernel> makeKernelAFma();
std::unique_ptr<Kernel> makeKernelBStrict();
std::unique_ptr<Kernel> makeKernelBFma();

} // namespace mascot

#endif
